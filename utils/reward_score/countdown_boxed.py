"""Reward scoring utilities for Countdown with \\boxed{} output format.

Designed for Math-series models (e.g. Qwen2.5-Math-*) that naturally emit
\\boxed{expr} instead of <answer>...</answer>.
"""

import re
from typing import List, Optional


def extract_boxed(response: str) -> Optional[str]:
    """Return content of last \\boxed{} in response, or None."""
    matches = re.findall(r"\\boxed\{([^{}]*)\}", response)
    return matches[-1].strip() if matches else None


def format_reward(response: str) -> float:
    """1.0 if response contains at least one \\boxed{}."""
    return 1.0 if re.search(r"\\boxed\{", response) else 0.0


def answer_reward(response: str, numbers: List[int], target: float) -> float:
    """1.0 if last \\boxed{expr} uses each number exactly once and evaluates to target."""
    content = extract_boxed(response)
    if not content:
        return 0.0
    if not re.match(r"^[0-9+\-*/() ]+$", content):
        return 0.0

    used = [int(n) for n in re.findall(r"\d+", content)]
    if sorted(used) != sorted([int(n) for n in numbers]):
        return 0.0

    try:
        result = eval(content, {"__builtins__": None}, {})  # noqa: S307
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def is_answer_correct(response: str, ground_truth: dict) -> bool:
    return answer_reward(response, ground_truth["numbers"], ground_truth["target"]) > 0.5


def compute_score(response: str, ground_truth: dict) -> float:
    """Blended score: 0.1 * format + answer correctness."""
    fmt = format_reward(response)
    ans = answer_reward(response, ground_truth["numbers"], ground_truth["target"])
    return fmt * 0.1 + ans
