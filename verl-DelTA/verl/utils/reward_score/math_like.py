from verl.utils.reward_score.answer_extraction import extract_math_answer, strip_string
from verl.utils.reward_score.format_score import compute_format_score, extract_solution
from typing import Dict, Tuple, Optional
import json
import re
import numpy as np
from verl.utils.reward_score.math_g import boxed_reward_fn as math_v
from transformers import AutoTokenizer
import torch
alpha = 0.4



def mat_g(r,tc):
    new_label = math_v(r, strip_string(tc), fast=False)
    return new_label[1]

def run_math_eval(r, tc):
    try:
        pred = extract_math_answer(r)[-1]
        gold_answer = strip_string(tc)
        #print(pred)
        #print(gold_answer)
        if pred == gold_answer:
            accepted = True
        else:
            accepted = False
    except:
        accepted = False
    return int(accepted)


def compute_score(solution_str: str, ground_truth: str,**kwargs) :
    """
    Computes comprehensive score for model response.
    Args:
        solution_str: Raw model response string
        ground_truth: ground truth data
        answer_reward: Points awarded/deducted for answer correctness
    Returns:
        Total score (sum of format and answer rewards)
    """
    answer_reward = 1

    #answer_text, processed_str = extract_solution(solution_str)
    #print("\n\n" + "=" * 80)
    #print(" Processing New Sample ".center(80, '='))
    #print(f"[Ground Truth]\n{ground_truth}\n")
    #print(f"[Model Response]\n{solution_str}\n")
    
    format_score,query_pos,halt_f = compute_format_score(solution_str)

    len_scor = 0
    acc_i = 0
    pl = mat_g(solution_str, ground_truth)
    if pl>=1:
        acc_i = 1
        answer_score = answer_reward
    else:
        answer_score = 0
        acc_i = 0
    
    total_score = answer_score 

    sco = {'score':total_score, 'acc': acc_i}

    return sco
