"""
MBPP (Mostly Basic Python Programming) dataset loader.

Loads the sanitized MBPP benchmark (427 problems in the test split).
Each problem has: task_id, text (prompt), code (solution), test_list.
"""

import json
import os
import urllib.request
from typing import Dict, List

MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/mbpp/sanitized-mbpp.json"
)
MBPP_CACHE = os.path.join(os.path.dirname(__file__), ".cache", "mbpp_sanitized.json")


def _download_mbpp() -> str:
    """Download sanitized MBPP dataset if not cached."""
    if os.path.exists(MBPP_CACHE):
        return MBPP_CACHE
    
    os.makedirs(os.path.dirname(MBPP_CACHE), exist_ok=True)
    print(f"[MBPP] Downloading from {MBPP_URL} ...")
    urllib.request.urlretrieve(MBPP_URL, MBPP_CACHE)
    print(f"[MBPP] Cached at {MBPP_CACHE}")
    return MBPP_CACHE


def _build_prompt(problem: Dict) -> str:
    """Build a code generation prompt from MBPP problem description."""
    text = problem["text"]
    # Include the first test case as an example
    test_examples = problem.get("test_list", [])
    example_str = ""
    if test_examples:
        example_str = f"\n\nExamples:\n" + "\n".join(
            f"  >>> {t}" for t in test_examples[:2]
        )
    
    prompt = (
        f'"""\n{text}{example_str}\n"""\n'
    )
    return prompt


def _build_test_code(problem: Dict) -> str:
    """Build executable test code from MBPP test_list."""
    tests = problem.get("test_list", [])
    # Each test is an assert statement string
    test_code = "\n".join(tests)
    return test_code


def load_mbpp() -> List[Dict]:
    """
    Load sanitized MBPP problems.
    
    Returns a list of dicts, each with:
        - task_id: str           (e.g. "MBPP/1")
        - prompt: str            (natural language description + examples)
        - canonical_solution: str
        - test: str              (assert-based unit tests)
        - entry_point: str       (extracted function name)
        - dataset: str           (always "mbpp")
    """
    path = _download_mbpp()
    
    with open(path, "r", encoding="utf-8") as f:
        raw_problems = json.load(f)
    
    problems = []
    for p in raw_problems:
        # Extract function name from the solution code
        code = p.get("code", "")
        entry_point = ""
        for line in code.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("def "):
                entry_point = line_stripped.split("(")[0].replace("def ", "").strip()
                break
        
        problem = {
            "task_id": f"MBPP/{p.get('task_id', p.get('source_file', 'unknown'))}",
            "prompt": _build_prompt(p),
            "canonical_solution": code,
            "test": _build_test_code(p),
            "entry_point": entry_point,
            "dataset": "mbpp",
        }
        problems.append(problem)
    
    print(f"[MBPP] Loaded {len(problems)} problems")
    return problems
