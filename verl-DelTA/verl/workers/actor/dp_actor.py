# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, get_global_score_top_mask, calculate_rhoed_adv
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _forward_micro_batch_with_h(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            h: #(bs,r_leb,h_s)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            captured_h = []
            def capture_last_h_hook(module, args):
                # args[0] 就是倒数第二层输出给 lm_head 的纯净 hidden_state
                captured_h.append(args[0].detach())

            hook_handle = None
            # 兼容 FSDP / DeepSpeed 包装，动态寻找 lm_head
            for name, module in self.actor_module.named_modules():
                if name.split('.')[-1] == "lm_head" or name.endswith("lm_head"):
                    hook_handle = module.register_forward_pre_hook(capture_last_h_hook)
                    break
            
            if hook_handle is None:
                raise RuntimeError("Failed to find 'lm_head'. Cannot capture hidden states.")

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                hook_handle.remove()
                h_rmpad = captured_h[0]
                if h_rmpad.dim() == 3:
                    h_rmpad = h_rmpad.squeeze(0)

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    h_rmpad = gather_outputs_and_unpad(h_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    h_rmpad = h_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_h = pad_input(hidden_states=h_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                h = full_h[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, h

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_h(self, data: DataProto, calculate_entropy: bool = False):
        """
        Returns:
            log_probs: [B, S]
            entropys: [B, S] or None
            W: [H]
            b: scalar
            preds: [B, S]
            weights: [B, S]
            weighted_advantages: [B, S]
        """

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "advantages"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)


        lam = 1e-6                
        eps = 1e-8
        disc_steps = self.config.get("K", 1)
        version = self.config.get("impl", "Normal")
        final_w_lo = self.config.get("lam_min", 0.8)
        final_w_hi = self.config.get("lam_max", 1.2)
        gamma_time = 1

        lam_w = 1.0



        if version=='Normal':

            log_probs_lst = []
            entropy_lst = []
            pred_lst = []
            weight_lst = []
            weighted_adv_lst = []


            sum_v_pos = None
            sum_v_neg = None
            sum_v2_pos = None
            sum_v2_neg = None
            S_p = None
            S_n = None

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    entropy, log_probs, h = self._forward_micro_batch_with_h(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    leng = log_probs.shape[-1]
                    token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()   # [B,S]
                    adv = micro_batch.batch["advantages"][:, 0].float()                   # [B]

                    probs = torch.exp(log_probs)                                          # [B,S]
                    v = (1.0 - probs).unsqueeze(-1) * h                                   # [B,S,H]
                    v2 = v ** 2
                    token_mask_3d = token_mask.unsqueeze(-1)                              # [B,S,1]

                    adv_pos = torch.clamp(adv, min=0.0)
                    adv_neg = torch.clamp(-adv, min=0.0)

                    w_pos0 = adv_pos[:, None, None] * token_mask_3d
                    w_neg0 = adv_neg[:, None, None] * token_mask_3d

                    if S_p is None:
                        hidden_dim = h.shape[-1]
                        device = h.device
                        dtype = h.dtype

                        sum_v_pos = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        sum_v_neg = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        sum_v2_pos = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        sum_v2_neg = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        S_p = torch.zeros((), device=device, dtype=dtype)
                        S_n = torch.zeros((), device=device, dtype=dtype)

                    sum_v_pos = sum_v_pos + (w_pos0 * v).sum(dim=(0, 1))
                    sum_v_neg = sum_v_neg + (w_neg0 * v).sum(dim=(0, 1))
                    sum_v2_pos = sum_v2_pos + (w_pos0 * v2).sum(dim=(0, 1))
                    sum_v2_neg = sum_v2_neg + (w_neg0 * v2).sum(dim=(0, 1))
                    S_p = S_p + w_pos0.sum()
                    S_n = S_n + w_neg0.sum()

                    del h, v, v2

                log_probs_lst.append(log_probs)
                if calculate_entropy:
                    entropy_lst.append(entropy)

            S_p = torch.clamp(S_p, min=eps)
            S_n = torch.clamp(S_n, min=eps)

            mu_pos = sum_v_pos / S_p
            mu_neg = sum_v_neg / S_n

            sig_pos = sum_v2_pos / S_p - mu_pos ** 2
            sig_neg = sum_v2_neg / S_n - mu_neg ** 2
            sig_pos = torch.clamp(sig_pos, min=0.0)
            sig_neg = torch.clamp(sig_neg, min=0.0)

            
            d = sig_pos + sig_neg + lam   # [H]
            d = torch.ones_like(mu_pos)


            m_pos_sumsq, m_pos_sum, m_pos_count = 0.0, 0.0, 0.0
            m_neg_sumsq, m_neg_sum, m_neg_count = 0.0, 0.0, 0.0

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    _, log_probs, h = self._forward_micro_batch_with_h(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                    )

                    leng = log_probs.shape[-1]
                    token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()
                    adv = micro_batch.batch["advantages"][:, 0].float()

                    probs = torch.exp(log_probs)
                    v = (1.0 - probs).unsqueeze(-1) * h

                    diff_pos = v - mu_pos.view(1, 1, -1)
                    diff_neg = v - mu_neg.view(1, 1, -1)

                    d_pos = ((diff_pos ** 2) / d.view(1, 1, -1)).sum(dim=-1)
                    d_neg = ((diff_neg ** 2) / d.view(1, 1, -1)).sum(dim=-1)

                    pos_seq_mask = (adv > 0).float().unsqueeze(1)
                    neg_seq_mask = (adv < 0).float().unsqueeze(1)
                    
                    valid_pos_mask = pos_seq_mask * token_mask
                    valid_neg_mask = neg_seq_mask * token_mask

                    
                    margin_pos = lam_w * d_neg - d_pos
                    margin_neg = lam_w * d_pos - d_neg

                    if isinstance(m_pos_sumsq, float):
                        device = mu_pos.device
                        dtype = mu_pos.dtype
                        m_pos_sumsq = torch.zeros((), device=device, dtype=dtype)
                        m_pos_sum = torch.zeros((), device=device, dtype=dtype)
                        m_pos_count = torch.zeros((), device=device, dtype=dtype)
                        m_neg_sumsq = torch.zeros((), device=device, dtype=dtype)
                        m_neg_sum = torch.zeros((), device=device, dtype=dtype)
                        m_neg_count = torch.zeros((), device=device, dtype=dtype)

                    m_pos_sum += (margin_pos * valid_pos_mask).sum()
                    m_pos_sumsq += (margin_pos ** 2 * valid_pos_mask).sum()
                    m_pos_count += valid_pos_mask.sum()

                    m_neg_sum += (margin_neg * valid_neg_mask).sum()
                    m_neg_sumsq += (margin_neg ** 2 * valid_neg_mask).sum()
                    m_neg_count += valid_neg_mask.sum()

                    del h, v, diff_pos, diff_neg, d_pos, d_neg, margin_pos, margin_neg

            m_pos_count = torch.clamp(m_pos_count, min=1.0)
            m_pos_mean = m_pos_sum / m_pos_count
            gamma_pos = gamma_time*torch.sqrt(torch.clamp(m_pos_sumsq / m_pos_count - m_pos_mean ** 2, min=eps))

            m_neg_count = torch.clamp(m_neg_count, min=1.0)
            m_neg_mean = m_neg_sum / m_neg_count
            gamma_neg = gamma_time*torch.sqrt(torch.clamp(m_neg_sumsq / m_neg_count - m_neg_mean ** 2, min=eps))


            mu_pos_star = mu_pos.clone()
            mu_neg_star = mu_neg.clone()

            num_bottom = 0
            num_top = 0
            total_valid = 0
            num_w = 0

            for _ in range(disc_steps):
                num_pos = torch.zeros_like(mu_pos_star)
                num_neg = torch.zeros_like(mu_neg_star)
                den_pos = torch.zeros((), device=mu_pos_star.device, dtype=mu_pos_star.dtype)
                den_neg = torch.zeros((), device=mu_neg_star.device, dtype=mu_neg_star.dtype)

                next_m_pos_sumsq, next_m_pos_sum, next_m_pos_count = 0.0, 0.0, 0.0
                next_m_neg_sumsq, next_m_neg_sum, next_m_neg_count = 0.0, 0.0, 0.0

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    with torch.no_grad():
                        _, log_probs, h = self._forward_micro_batch_with_h(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=False,
                        )

                        leng = log_probs.shape[-1]
                        token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()
                        adv = micro_batch.batch["advantages"][:, 0].float()

                        probs = torch.exp(log_probs)
                        v = (1.0 - probs).unsqueeze(-1) * h

                        diff_pos = v - mu_pos_star.view(1, 1, -1)
                        diff_neg = v - mu_neg_star.view(1, 1, -1)

                        d_pos = ((diff_pos ** 2) / d.view(1, 1, -1)).sum(dim=-1)
                        d_neg = ((diff_neg ** 2) / d.view(1, 1, -1)).sum(dim=-1)

                        pos_seq_mask = (adv > 0).float().unsqueeze(1)
                        neg_seq_mask = (adv < 0).float().unsqueeze(1)
                        
                        valid_pos_mask = pos_seq_mask * token_mask
                        valid_neg_mask = neg_seq_mask * token_mask

                        margin_pos = lam_w * d_neg - d_pos
                        margin_neg = lam_w * d_pos - d_neg

                        # Sigmoid posterior probability mapping
                        raw_w_pos = torch.sigmoid(margin_pos / gamma_pos)
                        raw_w_neg = torch.sigmoid(margin_neg / gamma_neg)

                        base_pos = torch.clamp(adv, min=0.0).unsqueeze(1)
                        base_neg = torch.clamp(-adv, min=0.0).unsqueeze(1)

                        w_pos = raw_w_pos * base_pos * valid_pos_mask
                        w_neg = raw_w_neg * base_neg * valid_neg_mask

                        num_pos += (w_pos.unsqueeze(-1) * v).sum(dim=(0, 1))
                        num_neg += (w_neg.unsqueeze(-1) * v).sum(dim=(0, 1))
                        den_pos += w_pos.sum()
                        den_neg += w_neg.sum()

                        
                        if isinstance(next_m_pos_sumsq, float):
                            device = mu_pos_star.device
                            dtype = mu_pos_star.dtype
                            next_m_pos_sumsq = torch.zeros((), device=device, dtype=dtype)
                            next_m_pos_sum = torch.zeros((), device=device, dtype=dtype)
                            next_m_pos_count = torch.zeros((), device=device, dtype=dtype)
                            next_m_neg_sumsq = torch.zeros((), device=device, dtype=dtype)
                            next_m_neg_sum = torch.zeros((), device=device, dtype=dtype)
                            next_m_neg_count = torch.zeros((), device=device, dtype=dtype)

                        next_m_pos_sum += (margin_pos * valid_pos_mask).sum()
                        next_m_pos_sumsq += (margin_pos ** 2 * valid_pos_mask).sum()
                        next_m_pos_count += valid_pos_mask.sum()

                        next_m_neg_sum += (margin_neg * valid_neg_mask).sum()
                        next_m_neg_sumsq += (margin_neg ** 2 * valid_neg_mask).sum()
                        next_m_neg_count += valid_neg_mask.sum()

                        del h, v, diff_pos, diff_neg, d_pos, d_neg, margin_pos, margin_neg
                        del raw_w_pos, raw_w_neg, w_pos, w_neg

                den_pos = torch.clamp(den_pos, min=eps)
                den_neg = torch.clamp(den_neg, min=eps)

                mu_pos_star = num_pos / den_pos
                mu_neg_star = num_neg / den_neg

               
                next_m_pos_count = torch.clamp(next_m_pos_count, min=1.0)
                next_m_pos_mean = next_m_pos_sum / next_m_pos_count
                gamma_pos = gamma_time*torch.sqrt(torch.clamp(next_m_pos_sumsq / next_m_pos_count - next_m_pos_mean ** 2, min=eps))

                next_m_neg_count = torch.clamp(next_m_neg_count, min=1.0)
                next_m_neg_mean = next_m_neg_sum / next_m_neg_count
                gamma_neg = gamma_time*torch.sqrt(torch.clamp(next_m_neg_sumsq / next_m_neg_count - next_m_neg_mean ** 2, min=eps))

            W = (mu_pos_star - mu_neg_star) / d
            b = -0.5 * ((mu_pos_star ** 2 - mu_neg_star ** 2) / d).sum()

            tk_sum = torch.zeros((), device=W.device, dtype=W.dtype)
            wht_sum = torch.zeros((), device=W.device, dtype=W.dtype)

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    entropy, log_probs, h = self._forward_micro_batch_with_h(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                    )

                    leng = log_probs.shape[-1]
                    token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()
                    adv = micro_batch.batch["advantages"][:, 0].float()

                    probs = torch.exp(log_probs)
                    v = (1.0 - probs).unsqueeze(-1) * h

                    score = (v * W.view(1, 1, -1)).sum(dim=-1) + b
                    y = torch.sign(adv).unsqueeze(1)
                    pred = y * score * token_mask

                    diff_pos = v - mu_pos_star.view(1, 1, -1)
                    diff_neg = v - mu_neg_star.view(1, 1, -1)

                    d_pos = ((diff_pos ** 2) / d.view(1, 1, -1)).sum(dim=-1)
                    d_neg = ((diff_neg ** 2) / d.view(1, 1, -1)).sum(dim=-1)

                    margin_pos = lam_w * d_neg - d_pos
                    margin_neg = lam_w * d_pos - d_neg

                   
                    w_pos_strict = torch.sigmoid(margin_pos / gamma_pos)
                    w_neg_strict = torch.sigmoid(margin_neg / gamma_neg)

                    w_pos_final = final_w_lo + (final_w_hi - final_w_lo) * w_pos_strict
                    w_neg_final = final_w_lo + (final_w_hi - final_w_lo) * w_neg_strict

                    pos_seq_mask = (adv > 0).float().unsqueeze(1)
                    neg_seq_mask = (adv < 0).float().unsqueeze(1)
                    
                    
                    zero_seq_mask = 1.0 - pos_seq_mask - neg_seq_mask

                    weight = (w_pos_final * pos_seq_mask + w_neg_final * neg_seq_mask + final_w_lo * zero_seq_mask) * token_mask

                    tk_sum += token_mask.sum()
                    wht_sum += weight.sum()

                    weighted_adv = adv.unsqueeze(1) * weight * token_mask


                    del h, v, score, y, diff_pos, diff_neg, d_pos, d_neg, margin_pos, margin_neg
                    del w_pos_strict, w_neg_strict, w_pos_final, w_neg_final

                pred_lst.append(pred)
                weight_lst.append(weight)
                weighted_adv_lst.append(weighted_adv)

            log_probs = torch.concat(log_probs_lst, dim=0)
            preds = torch.concat(pred_lst, dim=0)
            weights = torch.concat(weight_lst, dim=0)
            weighted_advantages = torch.concat(weighted_adv_lst, dim=0)

            entropys = None
            if calculate_entropy:
                entropys = torch.concat(entropy_lst, dim=0)


            if use_dynamic_bsz:
                log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
                preds = restore_dynamic_batch(preds, batch_idx_list)
                weights = restore_dynamic_batch(weights, batch_idx_list)
                weighted_advantages = restore_dynamic_batch(weighted_advantages, batch_idx_list)
                if calculate_entropy:
                    entropys = restore_dynamic_batch(entropys, batch_idx_list)


            gy_norm = tk_sum / wht_sum.clamp_min(1e-8)
            weights *= gy_norm
            weighted_advantages *= gy_norm


            return log_probs, entropys, W, b, preds, weights, weighted_advantages
        elif version=='Memory_efficient':
            log_probs_lst = []
            entropy_lst = []
            pred_lst = []
            weight_lst = []
            weighted_adv_lst = []

            sum_v_pos, sum_v_neg = None, None
            sum_v2_pos, sum_v2_neg = None, None
            S_p, S_n = None, None

            def compute_distances_memory_efficient(v, mu_pos_star, mu_neg_star, d):
                # v: [B, S, H], mu: [H], d: [H]
                # (v - mu)^2 / d  ==  v^2/d - 2v(mu/d) + mu^2/d
                v2_d_sum = ((v ** 2) / d.view(1, 1, -1)).sum(dim=-1) # [B, S]
                
                mu_pos_d = mu_pos_star / d
                mu_neg_d = mu_neg_star / d
                
                v_mu_pos = torch.matmul(v, mu_pos_d) # [B, S]
                v_mu_neg = torch.matmul(v, mu_neg_d) # [B, S]
                
                mu_pos_sq_d_sum = (mu_pos_star * mu_pos_d).sum()
                mu_neg_sq_d_sum = (mu_neg_star * mu_neg_d).sum()
                
                d_pos = (v2_d_sum - 2.0 * v_mu_pos + mu_pos_sq_d_sum).clamp_min(0.0)
                d_neg = (v2_d_sum - 2.0 * v_mu_neg + mu_neg_sq_d_sum).clamp_min(0.0)
                
                return d_pos, d_neg

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    entropy, log_probs, h = self._forward_micro_batch_with_h(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    leng = log_probs.shape[-1]
                    token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()
                    adv = micro_batch.batch["advantages"][:, 0].float()

                    probs = torch.exp(log_probs)
                    v = (1.0 - probs).unsqueeze(-1) * h
                    
                    del h, probs

                    adv_pos = torch.clamp(adv, min=0.0)
                    adv_neg = torch.clamp(-adv, min=0.0)

                    w_pos0 = adv_pos[:, None] * token_mask # [B, S]
                    w_neg0 = adv_neg[:, None] * token_mask # [B, S]

                    if S_p is None:
                        hidden_dim = v.shape[-1]
                        device, dtype = v.device, v.dtype
                        sum_v_pos = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        sum_v_neg = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        sum_v2_pos = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        sum_v2_neg = torch.zeros(hidden_dim, device=device, dtype=dtype)
                        S_p = torch.zeros((), device=device, dtype=dtype)
                        S_n = torch.zeros((), device=device, dtype=dtype)

                    w_pos0_flat = w_pos0.flatten().to(v.dtype) # [B*S]
                    w_neg0_flat = w_neg0.flatten().to(v.dtype)
                    
                    v_flat = v.flatten(0, 1)   
                    v2_flat = (v**2).flatten(0, 1)

                    sum_v_pos += torch.matmul(w_pos0_flat, v_flat)
                    sum_v_neg += torch.matmul(w_neg0_flat, v_flat)
                    sum_v2_pos += torch.matmul(w_pos0_flat, v2_flat)
                    sum_v2_neg += torch.matmul(w_neg0_flat, v2_flat)
                    
                    S_p += w_pos0_flat.sum()
                    S_n += w_neg0_flat.sum()

                    del v, v_flat, v2_flat, w_pos0, w_neg0, w_pos0_flat, w_neg0_flat



            S_p = torch.clamp(S_p, min=eps)
            S_n = torch.clamp(S_n, min=eps)

            mu_pos = sum_v_pos / S_p
            mu_neg = sum_v_neg / S_n

            sig_pos = torch.clamp(sum_v2_pos / S_p - mu_pos ** 2, min=0.0)
            sig_neg = torch.clamp(sum_v2_neg / S_n - mu_neg ** 2, min=0.0)

            d = sig_pos + sig_neg + lam
            d = torch.ones_like(mu_pos)

            m_pos_sumsq, m_pos_sum, m_pos_count = 0.0, 0.0, 0.0
            m_neg_sumsq, m_neg_sum, m_neg_count = 0.0, 0.0, 0.0

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    _, log_probs, h = self._forward_micro_batch_with_h(
                        model_inputs, temperature=temperature, calculate_entropy=False
                    )
                    leng = log_probs.shape[-1]
                    token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()
                    adv = micro_batch.batch["advantages"][:, 0].float()

                    probs = torch.exp(log_probs)
                    v = (1.0 - probs).unsqueeze(-1) * h
                    del h, probs

                    d_pos, d_neg = compute_distances_memory_efficient(v, mu_pos, mu_neg, d)

                    pos_seq_mask = (adv > 0).float().unsqueeze(1)
                    neg_seq_mask = (adv < 0).float().unsqueeze(1)
                    
                    valid_pos_mask = pos_seq_mask * token_mask
                    valid_neg_mask = neg_seq_mask * token_mask

                    margin_pos = lam_w * d_neg - d_pos
                    margin_neg = lam_w * d_pos - d_neg

                    if isinstance(m_pos_sumsq, float):
                        device, dtype = mu_pos.device, mu_pos.dtype
                        m_pos_sumsq = torch.zeros((), device=device, dtype=dtype)
                        m_pos_sum = torch.zeros((), device=device, dtype=dtype)
                        m_pos_count = torch.zeros((), device=device, dtype=dtype)
                        m_neg_sumsq = torch.zeros((), device=device, dtype=dtype)
                        m_neg_sum = torch.zeros((), device=device, dtype=dtype)
                        m_neg_count = torch.zeros((), device=device, dtype=dtype)

                    m_pos_sum += (margin_pos * valid_pos_mask).sum()
                    m_pos_sumsq += (margin_pos ** 2 * valid_pos_mask).sum()
                    m_pos_count += valid_pos_mask.sum()

                    m_neg_sum += (margin_neg * valid_neg_mask).sum()
                    m_neg_sumsq += (margin_neg ** 2 * valid_neg_mask).sum()
                    m_neg_count += valid_neg_mask.sum()

                    del v, d_pos, d_neg, margin_pos, margin_neg

            m_pos_count = torch.clamp(m_pos_count, min=1.0)
            m_pos_mean = m_pos_sum / m_pos_count
            gamma_pos = gamma_time * torch.sqrt(torch.clamp(m_pos_sumsq / m_pos_count - m_pos_mean ** 2, min=eps))

            m_neg_count = torch.clamp(m_neg_count, min=1.0)
            m_neg_mean = m_neg_sum / m_neg_count
            gamma_neg = gamma_time * torch.sqrt(torch.clamp(m_neg_sumsq / m_neg_count - m_neg_mean ** 2, min=eps))


            mu_pos_star = mu_pos.clone()
            mu_neg_star = mu_neg.clone()

            for _ in range(disc_steps):
                num_pos = torch.zeros_like(mu_pos_star)
                num_neg = torch.zeros_like(mu_neg_star)
                den_pos = torch.zeros((), device=mu_pos_star.device, dtype=mu_pos_star.dtype)
                den_neg = torch.zeros((), device=mu_neg_star.device, dtype=mu_neg_star.dtype)

                next_m_pos_sumsq, next_m_pos_sum, next_m_pos_count = 0.0, 0.0, 0.0
                next_m_neg_sumsq, next_m_neg_sum, next_m_neg_count = 0.0, 0.0, 0.0

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    with torch.no_grad():
                        _, log_probs, h = self._forward_micro_batch_with_h(
                            model_inputs, temperature=temperature, calculate_entropy=False
                        )
                        leng = log_probs.shape[-1]
                        token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()
                        adv = micro_batch.batch["advantages"][:, 0].float()

                        probs = torch.exp(log_probs)
                        v = (1.0 - probs).unsqueeze(-1) * h
                        del h, probs

                        d_pos, d_neg = compute_distances_memory_efficient(v, mu_pos_star, mu_neg_star, d)

                        pos_seq_mask = (adv > 0).float().unsqueeze(1)
                        neg_seq_mask = (adv < 0).float().unsqueeze(1)
                        
                        valid_pos_mask = pos_seq_mask * token_mask
                        valid_neg_mask = neg_seq_mask * token_mask

                        margin_pos = lam_w * d_neg - d_pos
                        margin_neg = lam_w * d_pos - d_neg

                        raw_w_pos = torch.sigmoid(margin_pos / gamma_pos)
                        raw_w_neg = torch.sigmoid(margin_neg / gamma_neg)

                        base_pos = torch.clamp(adv, min=0.0).unsqueeze(1)
                        base_neg = torch.clamp(-adv, min=0.0).unsqueeze(1)

                        w_pos = raw_w_pos * base_pos * valid_pos_mask
                        w_neg = raw_w_neg * base_neg * valid_neg_mask

                        
                        v_flat = v.flatten(0, 1)
                        num_pos += torch.matmul(w_pos.flatten().to(v.dtype), v_flat)
                        num_neg += torch.matmul(w_neg.flatten().to(v.dtype), v_flat)
                        
                        den_pos += w_pos.sum()
                        den_neg += w_neg.sum()

                        if isinstance(next_m_pos_sumsq, float):
                            device, dtype = mu_pos_star.device, mu_pos_star.dtype
                            next_m_pos_sumsq = torch.zeros((), device=device, dtype=dtype)
                            next_m_pos_sum = torch.zeros((), device=device, dtype=dtype)
                            next_m_pos_count = torch.zeros((), device=device, dtype=dtype)
                            next_m_neg_sumsq = torch.zeros((), device=device, dtype=dtype)
                            next_m_neg_sum = torch.zeros((), device=device, dtype=dtype)
                            next_m_neg_count = torch.zeros((), device=device, dtype=dtype)

                        next_m_pos_sum += (margin_pos * valid_pos_mask).sum()
                        next_m_pos_sumsq += (margin_pos ** 2 * valid_pos_mask).sum()
                        next_m_pos_count += valid_pos_mask.sum()

                        next_m_neg_sum += (margin_neg * valid_neg_mask).sum()
                        next_m_neg_sumsq += (margin_neg ** 2 * valid_neg_mask).sum()
                        next_m_neg_count += valid_neg_mask.sum()

                        del v, v_flat, d_pos, d_neg, margin_pos, margin_neg
                        del raw_w_pos, raw_w_neg, w_pos, w_neg

                den_pos = torch.clamp(den_pos, min=eps)
                den_neg = torch.clamp(den_neg, min=eps)

                mu_pos_star = num_pos / den_pos
                mu_neg_star = num_neg / den_neg

                next_m_pos_count = torch.clamp(next_m_pos_count, min=1.0)
                next_m_pos_mean = next_m_pos_sum / next_m_pos_count
                gamma_pos = gamma_time * torch.sqrt(torch.clamp(next_m_pos_sumsq / next_m_pos_count - next_m_pos_mean ** 2, min=eps))

                next_m_neg_count = torch.clamp(next_m_neg_count, min=1.0)
                next_m_neg_mean = next_m_neg_sum / next_m_neg_count
                gamma_neg = gamma_time * torch.sqrt(torch.clamp(next_m_neg_sumsq / next_m_neg_count - next_m_neg_mean ** 2, min=eps))


            W = (mu_pos_star - mu_neg_star) / d
            b = -0.5 * ((mu_pos_star ** 2 - mu_neg_star ** 2) / d).sum()

            tk_sum = torch.zeros((), device=W.device, dtype=W.dtype)
            wht_sum = torch.zeros((), device=W.device, dtype=W.dtype)

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    entropy, log_probs, h = self._forward_micro_batch_with_h(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    log_probs_lst.append(log_probs)
                    if calculate_entropy:
                        entropy_lst.append(entropy)

                    leng = log_probs.shape[-1]
                    token_mask = micro_batch.batch["attention_mask"][:, -leng:].float()
                    adv = micro_batch.batch["advantages"][:, 0].float()

                    probs = torch.exp(log_probs)
                    v = (1.0 - probs).unsqueeze(-1) * h
                    del h, probs

           
                    score = torch.matmul(v, W) + b
                    y = torch.sign(adv).unsqueeze(1)
                    pred = y * score * token_mask

                    d_pos, d_neg = compute_distances_memory_efficient(v, mu_pos_star, mu_neg_star, d)

                    margin_pos = lam_w * d_neg - d_pos
                    margin_neg = lam_w * d_pos - d_neg

                    w_pos_strict = torch.sigmoid(margin_pos / gamma_pos)
                    w_neg_strict = torch.sigmoid(margin_neg / gamma_neg)

                    w_pos_final = final_w_lo + (final_w_hi - final_w_lo) * w_pos_strict
                    w_neg_final = final_w_lo + (final_w_hi - final_w_lo) * w_neg_strict

                    pos_seq_mask = (adv > 0).float().unsqueeze(1)
                    neg_seq_mask = (adv < 0).float().unsqueeze(1)
                    zero_seq_mask = 1.0 - pos_seq_mask - neg_seq_mask

                    weight = (w_pos_final * pos_seq_mask + w_neg_final * neg_seq_mask + final_w_lo * zero_seq_mask) * token_mask

                    tk_sum += token_mask.sum()
                    wht_sum += weight.sum()

                    weighted_adv = adv.unsqueeze(1) * weight * token_mask

                    del v, score, y, d_pos, d_neg, margin_pos, margin_neg
                    del w_pos_strict, w_neg_strict, w_pos_final, w_neg_final

                pred_lst.append(pred)
                weight_lst.append(weight)
                weighted_adv_lst.append(weighted_adv)

            log_probs = torch.concat(log_probs_lst, dim=0)
            preds = torch.concat(pred_lst, dim=0)
            weights = torch.concat(weight_lst, dim=0)
            weighted_advantages = torch.concat(weighted_adv_lst, dim=0)

            entropys = None
            if calculate_entropy:
                entropys = torch.concat(entropy_lst, dim=0)

            if use_dynamic_bsz:
                log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
                preds = restore_dynamic_batch(preds, batch_idx_list)
                weights = restore_dynamic_batch(weights, batch_idx_list)
                weighted_advantages = restore_dynamic_batch(weighted_advantages, batch_idx_list)
                if calculate_entropy:
                    entropys = restore_dynamic_batch(entropys, batch_idx_list)

            gy_norm = tk_sum / wht_sum.clamp_min(1e-8)
            weights *= gy_norm
            weighted_advantages *= gy_norm

            return log_probs, entropys, W, b, preds, weights, weighted_advantages




    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        score_top_ratio = 0.2

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "pred",
            "weights",
            "weighted_adv"
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    score = model_inputs["pred"]
                    weighted_adv = model_inputs["weighted_adv"]
                    weights = model_inputs["weights"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    #loss_agg_mode = "weighted-mean"

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob  = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )



                    #print(log_prob.shape)
                    #print(h.shape)


                    score_top_mask = None
                    if score_top_ratio is not None:
                        score_top_mask = get_global_score_top_mask(score=score, response_mask=response_mask, top_ratio=score_top_ratio)

                    #rhoed_adv = calculate_rhoed_adv(score=score, response_mask=response_mask, advantage=advantages)

                    #print(score_top_mask)
                    #print("SCORE_MASK :"+str(score_top_mask.sum()))
                    #print('---')
                    #print(response_mask)
                    #print("RESPONSE_MASK :"+str(response_mask.sum()))





                    #exit(23333)

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        #advantages=advantages,
                        advantages=weighted_adv,
                        weights=weights,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
