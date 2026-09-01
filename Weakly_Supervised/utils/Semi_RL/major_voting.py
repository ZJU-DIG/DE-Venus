from verl import DataProto
from math_verify import parse, verify
from .math_utils import last_boxed_only_string, remove_boxed, grade_answer_mathd, grade_answer_sympy


def extract_pred_answers(data: DataProto, indices: list, tokenizer) -> list:
    responses = []
    for idx in indices:
        data_item = data[idx]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        responses.append(response_str)
    
    pred_answers = []
    for response in responses:
        boxed_pred = last_boxed_only_string(response)
        extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None
        pred_answers.append(extracted_pred)
    
    return pred_answers


def extract_pred_answers_mv(data: DataProto, indices: list, tokenizer) -> list:
    responses = []
    for idx in indices:
        data_item = data[idx]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        responses.append(response_str)
    
    pred_answers = list(map(parse, responses))
    
    return pred_answers


def find_majority(pred_answers: list) -> str | None:
    # filter out invalid answers
    pred_answers = [ans for ans in pred_answers if ans is not None]
    if not pred_answers:
        return None
    
    # count frequencies
    freq: dict[str, int] = {}
    for ans in pred_answers:
        freq[ans] = freq.get(ans, 0) + 1
    
    max_cnt = max(freq.values())
    major_answers = [ans for ans, cnt in freq.items() if cnt == max_cnt]
    if len(major_answers) > 1:
        return None

    return major_answers[0]


def find_majority_mv(pred_answers: list) -> str | None:
    freq = []
    for ans in pred_answers:
        found = False
        for k, (exist_ans, count) in enumerate(freq):
            try:
                if exist_ans[-1] == ans[-1] or verify(exist_ans, ans):
                    freq[k] = (exist_ans, count + 1)
                    found = True
                    break
            except Exception:
                pass
        if not found and len(ans) == 2:
            freq.append((ans, 1))
    
    counts = [count for _, count in freq]
    
    if len(counts) > 0:
        max_count = max(counts)
        majority_answer = next(rep[-1] for rep, cnt in freq if cnt == max_count)
    else:
        return None

    return majority_answer


def verify_answer(pred_answer: str, gold_answer: str) -> bool:
    return grade_answer_mathd(pred_answer, gold_answer) or grade_answer_sympy(pred_answer, gold_answer)


def verify_answer_mv(pred_answer, gold_answer: str) -> bool:
    if isinstance(pred_answer, str):
        pred_answer = parse("$" + pred_answer + "$")
    if isinstance(gold_answer, str):
        gold_answer = parse("$" + gold_answer + "$")
    
    try:
        if pred_answer[-1] == gold_answer[-1] or verify(pred_answer, gold_answer):
            return True
    except Exception:
        return False
    
    return False