from typing import Dict, Tuple, Optional
import re


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """
    Extracts the final answer from the model's response string.
    Args:
        solution_str: Raw response string from the language model
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    """
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant\n" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str
    """
    processed_str = solution_str
    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        #print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str):
    """
    Performs comprehensive validation of response structure.
    Args:
        processed_str: Processed response string from the model 
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    #print(f" Structure Validation ".center(80, '-'))
    validation_passed = True

    # Check required tags
    tags = {
        'think_end': ('</think>', 1),
    }

    positions = {}
    query_positions = [m.start() for m in re.finditer('<query>', processed_str)]
    qe_positions = [m.start() for m in re.finditer('</query>', processed_str)]
    #las_q = query_positions[-1]
    #print(query_positions)
    #print(processed_str)
    #exit(2333)
    if len(query_positions)==0:
        query_pos = -1
        las_q = -1
    else:
        query_pos = 0.0
        las_q = query_positions[-1]
        for i in range(len(query_positions)):
            query_pos += query_positions[i]
        query_pos /= len(query_positions)
        query_pos = query_pos/float(len(processed_str))
    is_hal = processed_str.find('<halt>')
    for tag_name, (tag_str, expected_count) in tags.items():
        #print(processed_str)
        #print('-----------')
        #exit(23333)
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        #print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            #print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False


    return validation_passed, query_pos, is_hal


def compute_format_score(solution_str: str, format_reward: float = 0.2):
    answer_text, processed_str = extract_solution(solution_str)
    format_correct,pos_st,is_hal = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else 0
    return format_score,pos_st,is_hal
