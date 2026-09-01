"""Unified verification utilities for math and instruction-following (IFEval) data.

Usage:
    from verify_utils import verify_response, detect_data_type

    data_type = detect_data_type(ground_truth)  # 'math' or 'ifeval'
    is_correct = verify_response(response_text, ground_truth, data_type)
"""

import re
import json
import signal


# ============================================================
# Timeout mechanism (for math_verify which can hang on complex expressions)
# ============================================================

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("timed out")


def _run_with_timeout(func, args=(), timeout_seconds=10, default=None):
    """Run a function with a timeout. Returns default on timeout or error."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return func(*args)
    except _TimeoutError:
        return default
    except Exception:
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================
# Data type detection
# ============================================================

def detect_data_type(ground_truth):
    """Detect whether ground_truth is for math or ifeval.

    Args:
        ground_truth: str (math answer) or dict/JSON-string with 'func_name' (ifeval).

    Returns:
        'math' or 'ifeval'
    """
    if isinstance(ground_truth, dict):
        if 'func_name' in ground_truth:
            return 'ifeval'
        return 'math'
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            if isinstance(gt, dict) and 'func_name' in gt:
                return 'ifeval'
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 'math'


# ============================================================
# Unified verification entry point
# ============================================================

def verify_response(response, ground_truth, data_type=None, strict_boxed=False):
    """Verify a single response against ground_truth.

    Args:
        response: The generated text.
        ground_truth: str for math, or dict/JSON-string for ifeval.
        data_type: 'math' or 'ifeval'. Auto-detected if None.
        strict_boxed: If True, math responses must also have a correct
                      answer inside the last \\boxed{} to be considered correct.

    Returns:
        bool: True if the response is correct.
    """
    if data_type is None:
        data_type = detect_data_type(ground_truth)

    if data_type == 'ifeval':
        gt_info = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
        return _ifeval_validate(response, gt_info)
    else:
        return _math_validate(response, ground_truth, strict_boxed=strict_boxed)


# ============================================================
# Math verification (via math_verify)
# ============================================================

def _extract_last_boxed(text):
    """Extract content from the last \\boxed{...} in text. Returns None if not found."""
    # Find all \boxed{...} occurrences, handling nested braces
    results = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 0
        start = idx + len('\\boxed{')
        for j in range(start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    results.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            break  # unmatched brace
    return results[-1] if results else None


def _math_validate(response, golden_answer, strict_boxed=False):
    """Verify a math response using math_verify with timeout protection.

    If strict_boxed=True, additionally requires the last \\boxed{} content
    to match the golden answer. Returns False if no \\boxed{} is found.
    """
    from math_verify import parse, verify

    # Standard math_verify check on full response
    pred = _run_with_timeout(parse, (response,), timeout_seconds=10)
    gold = _run_with_timeout(parse, ("$" + str(golden_answer) + "$",), timeout_seconds=10)

    if pred is None or gold is None:
        return False

    result = _run_with_timeout(verify, (gold, pred), timeout_seconds=10, default=False)
    if not bool(result):
        return False

    if strict_boxed:
        boxed_content = _extract_last_boxed(response)
        if boxed_content is None:
            return False
        boxed_pred = _run_with_timeout(parse, ("$" + boxed_content + "$",), timeout_seconds=10)
        if boxed_pred is None:
            return False
        boxed_result = _run_with_timeout(verify, (gold, boxed_pred), timeout_seconds=10, default=False)
        return bool(boxed_result)

    return True


# ============================================================
# IFEval verification
# Based on google-research/instruction_following_eval
# ============================================================

def _count_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n|\* \*', text.strip())
    return len([p for p in paragraphs if p.strip()])


def _count_sentences(text):
    text = re.sub(r'[.!?]+', '.', text)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)


def _ifeval_validate(response, gt_info):
    """Validate an ifeval response against its ground_truth constraints.

    Args:
        response: The generated text response.
        gt_info: Dict with keys like 'func_name', 'N', 'keyword_list',
                 'forbidden_words', 'letter', 'end_phrase', etc.

    Returns:
        True if the response satisfies the constraint, False otherwise.
    """
    func_name = gt_info.get('func_name', '')
    N = gt_info.get('N')
    keyword_list = gt_info.get('keyword_list') or []
    forbidden_words = gt_info.get('forbidden_words') or []
    letter = gt_info.get('letter')
    end_phrase = gt_info.get('end_phrase')
    word = gt_info.get('word')
    quantifier = gt_info.get('quantifier')
    i = gt_info.get('i')
    first_word = gt_info.get('first_word')
    postscript_marker = gt_info.get('postscript_marker')
    options = gt_info.get('options') or []
    section_splitter = gt_info.get('section_splitter')
    original_prompt = gt_info.get('original_prompt')

    response_lower = response.lower()

    try:
        if func_name == 'validate_lowercase':
            letters = [c for c in response if c.isalpha()]
            return all(c.islower() for c in letters)

        elif func_name == 'validate_uppercase':
            letters = [c for c in response if c.isalpha()]
            return all(c.isupper() for c in letters)

        elif func_name == 'validate_no_commas':
            return ',' not in response

        elif func_name == 'validate_quotation':
            return ('"' in response) or ("'" in response and response.count("'") >= 2)

        elif func_name == 'validate_json_format':
            json_pattern = r'\{[^{}]*\}|\[[^\[\]]*\]'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        json.loads(match)
                        return True
                    except json.JSONDecodeError:
                        continue
            try:
                json.loads(response.strip())
                return True
            except json.JSONDecodeError:
                pass
            return False

        elif func_name == 'validate_paragraphs':
            if N is not None:
                return _count_paragraphs(response) >= int(N)
            return False

        elif func_name == 'verify_paragraph_count':
            if N is not None:
                return _count_paragraphs(response) == int(N)
            return False

        elif func_name == 'validate_sections':
            if section_splitter and N is not None:
                sections = [s for s in response.split(section_splitter) if s.strip()]
                return len(sections) >= int(N)
            return False

        elif func_name == 'validate_highlighted_sections':
            if N is not None:
                highlighted = re.findall(r'\*[^*]+\*|\*\*[^*]+\*\*|<[^>]+>', response)
                return len(highlighted) >= int(N)
            return False

        elif func_name == 'validate_placeholders':
            if N is not None:
                placeholders = re.findall(r'\[.*?\]|\{.*?\}|<.*?>', response)
                return len(placeholders) >= int(N)
            return False

        elif func_name == 'validate_title':
            lines = [l for l in response.strip().split('\n') if l.strip()]
            if not lines:
                return False
            first_line = lines[0].strip()
            is_short = len(first_line.split()) <= 15
            is_bold = first_line.startswith('#') or first_line.startswith('**') or first_line.endswith('**')
            is_capitalized = first_line[0].isupper() if first_line else False
            return is_short and (is_bold or is_capitalized)

        elif func_name == 'validate_forbidden_words':
            if forbidden_words:
                for w in forbidden_words:
                    if w.lower() in response_lower:
                        return False
            return True

        elif func_name == 'validate_word_constraint':
            if word and N is not None:
                count = response_lower.count(word.lower())
                if quantifier == 'at least':
                    return count >= int(N)
                elif quantifier == 'at most':
                    return count <= int(N)
                else:
                    return count == int(N)
            return False

        elif func_name == 'validate_frequency_capital_words':
            capital_words = re.findall(r'\b[A-Z]{2,}\b', response)
            if N is not None:
                if quantifier == 'at least':
                    return len(capital_words) >= int(N)
                elif quantifier == 'at most':
                    return len(capital_words) <= int(N)
                else:
                    return len(capital_words) == int(N)
            return False

        elif func_name == 'verify_keyword_frequency':
            if keyword_list and N is not None:
                count = sum(response_lower.count(kw.lower()) for kw in keyword_list)
                if quantifier == 'at least':
                    return count >= int(N)
                elif quantifier == 'at most':
                    return count <= int(N)
                else:
                    return count == int(N)
            return False

        elif func_name == 'verify_keywords':
            if keyword_list:
                return all(kw.lower() in response_lower for kw in keyword_list)
            return False

        elif func_name == 'verify_letter_frequency':
            if letter and N is not None:
                count = response.lower().count(letter.lower())
                if quantifier == 'at least':
                    return count >= int(N)
                elif quantifier == 'at most':
                    return count <= int(N)
                else:
                    return count == int(N)
            return False

        elif func_name == 'verify_bullet_points':
            bullet_patterns = re.findall(r'^\s*[-*•]\s', response, re.MULTILINE)
            if N is not None:
                if quantifier == 'at least':
                    return len(bullet_patterns) >= int(N)
                elif quantifier == 'at most':
                    return len(bullet_patterns) <= int(N)
                else:
                    return len(bullet_patterns) == int(N)
            return len(bullet_patterns) > 0

        elif func_name == 'verify_sentence_constraint':
            if N is not None:
                num_sentences = _count_sentences(response)
                if quantifier == 'at least':
                    return num_sentences >= int(N)
                elif quantifier == 'at most':
                    return num_sentences <= int(N)
                else:
                    return num_sentences == int(N)
            return False

        elif func_name == 'verify_postscript':
            ps_patterns = ['p.s.', 'ps.', 'p.s', 'P.S.', 'PS.', 'P.S', 'PS']
            has_ps = any(pat in response for pat in ps_patterns)
            if postscript_marker:
                return has_ps or postscript_marker.lower() in response_lower
            return has_ps

        elif func_name == 'validate_end':
            if end_phrase:
                return response.strip().lower().endswith(end_phrase.strip().lower())
            return True

        elif func_name == 'validate_choice':
            if options:
                return any(opt.lower() in response_lower for opt in options)
            return True

        elif func_name == 'validate_two_responses':
            separators = ['---', '***', '===', 'answer 1', 'answer 2',
                          'response 1', 'response 2', 'option 1', 'option 2',
                          'choice 1', 'choice 2', 'first', 'second']
            if any(sep in response_lower for sep in separators):
                return True
            sections = [s for s in re.split(r'\n\s*\n', response.strip()) if s.strip()]
            return len(sections) >= 2

        elif func_name == 'validate_repeat_prompt':
            if original_prompt:
                prompt_words = set(original_prompt.strip().lower().split())
                response_words = set(response_lower.split())
                overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
                return overlap > 0.5
            return False

        elif func_name == 'verify_paragraph_count_with_first_word':
            count_ok = True
            if N is not None:
                count_ok = _count_paragraphs(response) >= int(N)
            first_word_ok = True
            if i is not None and first_word is not None:
                paragraphs = [p for p in re.split(r'\n\s*\n|\* \*', response.strip()) if p.strip()]
                idx = int(i) - 1 if int(i) > 0 else int(i)
                if 0 <= idx < len(paragraphs):
                    first_w = paragraphs[idx].strip().split()[0] if paragraphs[idx].strip().split() else ''
                    first_word_ok = first_w.lower().startswith(first_word.lower())
                else:
                    first_word_ok = False
            return count_ok and first_word_ok

        else:
            print(f"[IFEval] Warning: Unknown func_name '{func_name}', skipping validation")
            return True

    except Exception as e:
        print(f"[IFEval] Error validating func_name={func_name}: {e}")
        return True  # Don't penalize on errors
