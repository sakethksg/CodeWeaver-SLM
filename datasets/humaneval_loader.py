"""
HumanEval dataset loader.

Loads the OpenAI HumanEval benchmark (164 problems).
Each problem has: task_id, prompt, canonical_solution, test, entry_point.
"""

import json
import os
import urllib.request
from typing import Dict, List

HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)
HUMANEVAL_CACHE = os.path.join(os.path.dirname(__file__), ".cache", "HumanEval.jsonl")


def _download_humaneval() -> str:
    """Download HumanEval dataset if not cached."""
    if os.path.exists(HUMANEVAL_CACHE):
        return HUMANEVAL_CACHE
    
    os.makedirs(os.path.dirname(HUMANEVAL_CACHE), exist_ok=True)
    
    # Download gzipped file
    import gzip
    gz_path = HUMANEVAL_CACHE + ".gz"
    print(f"[HumanEval] Downloading from {HUMANEVAL_URL} ...")
    urllib.request.urlretrieve(HUMANEVAL_URL, gz_path)
    
    # Decompress
    with gzip.open(gz_path, "rb") as f_in:
        with open(HUMANEVAL_CACHE, "wb") as f_out:
            f_out.write(f_in.read())
    
    os.remove(gz_path)
    print(f"[HumanEval] Cached at {HUMANEVAL_CACHE}")
    return HUMANEVAL_CACHE


def load_humaneval() -> List[Dict]:
    """
    Load HumanEval problems.
    
    Returns a list of dicts, each with:
        - task_id: str           (e.g. "HumanEval/0")
        - prompt: str            (function signature + docstring)
        - canonical_solution: str
        - test: str              (unit test code)
        - entry_point: str       (function name to call)
        - dataset: str           (always "humaneval")
    """
    path = _download_humaneval()
    
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problem = json.loads(line)
            problem["dataset"] = "humaneval"
            problems.append(problem)
    
    print(f"[HumanEval] Loaded {len(problems)} problems")
    return problems
