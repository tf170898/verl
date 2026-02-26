import re


def _extract_action(text: str) -> str:
    if text is None:
        return ""
    upper = str(text).upper()
    if re.search(r"\bSTAND\b", upper):
        return "STAND"
    if re.search(r"\bHIT\b", upper):
        return "HIT"
    return ""


def bench_reward(data_source, solution_str, ground_truth, extra_info=None):
    pred = _extract_action(solution_str)
    target = str(ground_truth).upper().strip()
    return 1.0 if pred == target else -1.0

