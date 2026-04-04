"""
Candidate ranking and selection.

Ranks candidates based on execution results (test pass rate).
Implements the final selection strategy for the best solution.
"""

from typing import Dict, List, Tuple
from pipeline.executor import ExecutionResult


def rank_candidates(
    candidates: List[str],
    results: List[ExecutionResult],
) -> List[Tuple[str, ExecutionResult, float]]:
    """
    Rank candidates by their execution performance.
    
    Returns list of (code, result, score) sorted by score descending.
    Score = tests_passed / tests_total (1.0 = all tests pass).
    """
    scored = []
    for code, result in zip(candidates, results):
        if result.tests_total > 0:
            score = result.tests_passed / result.tests_total
        else:
            score = 1.0 if result.passed else 0.0
        scored.append((code, result, score))
    
    # Sort: passed first, then by score, then by execution time (faster = better)
    scored.sort(key=lambda x: (
        -int(x[1].passed),     # passed solutions first
        -x[2],                  # higher score
        x[1].execution_time,    # faster execution
    ))
    
    return scored


def select_best(
    candidates: List[str],
    results: List[ExecutionResult],
) -> Tuple[str, ExecutionResult, float]:
    """Select the single best candidate."""
    ranked = rank_candidates(candidates, results)
    if ranked:
        return ranked[0]
    return ("", ExecutionResult(), 0.0)
