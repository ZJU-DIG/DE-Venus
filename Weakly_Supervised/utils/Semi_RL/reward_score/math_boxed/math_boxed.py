from ...math_utils import extract_boxed_answer, grade_answer_mathd, grade_answer_sympy

def compute_score(model_output: str, ground_truth: str) -> bool:
    ret_score = False

    if "\\boxed" in model_output:
        model_answer = extract_boxed_answer(model_output)
    else:
        return ret_score
    
    if "\\boxed" in ground_truth:
        ground_truth = extract_boxed_answer(ground_truth)
    
    ret_score = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)

    return ret_score
