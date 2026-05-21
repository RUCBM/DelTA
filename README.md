# DelTA
[![arXiv](https://img.shields.io/badge/arXiv-2602.12125-red.svg)](https://arxiv.org/abs/2605.21467v1)

## Installation
Our code is mainly based on [verl](https://github.com/volcengine/verl). To prepare the environment, please follow these steps:

```bash
conda create -n delta python==3.12
conda activate delta
pip install torch==2.9.1
pip install flash_attn==2.8.3
pip install sglang==0.5.6
cd verl-DelTA
pip install -e.
pip install math-verify
```


## Train

We provide an example for DelTA training in the script `verl-DelTA/recipe/dapo/srcs/run_DelTA.sh`.


## Citation
If you find our work helpful, please kindly cite as
```bibtex
@misc{zhang2026deltadiscriminativetokencredit,
      title={DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards}, 
      author={Kaiyi Zhang and Wei Wu and Yankai Lin},
      year={2026},
      eprint={2605.21467},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.21467}, 
}
```
