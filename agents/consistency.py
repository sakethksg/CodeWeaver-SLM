"""
Consistency checks extracted from the original Orchestrator Agent.

These are standalone functions that validate cross-agent output consistency.
Used by the analysis graph's aggregation node.
"""

from typing import Any, Dict, List


def run_consistency_checks(
    gen_results: Dict,
    repair_results: Dict,
    oracle_results: Dict,
    exec_results: Dict,
    error_results: Dict,
    cost_results: Dict,
) -> Dict[str, Any]:
    """
    Run the 6 mandatory consistency checks from the prompt specification.

    Returns:
        Dict with "passed" bool and "inconsistencies" list
    """
    inconsistencies = []

    # 1. repair_pass@k >= generation_pass@k
    gen_pk = gen_results.get("pass_at_k_before_repair", {})
    rep_pk = repair_results.get("pass_at_k_after_repair", {})
    for k_str in ["pass@1", "pass@5", "pass@10"]:
        g = gen_pk.get(k_str)
        r = rep_pk.get(k_str)
        if g is None or r is None:
            continue
        if r < g - 1e-9:
            inconsistencies.append(
                f"[CHECK 1] repair {k_str}={r:.4f} < generation {k_str}={g:.4f}"
            )

    # 2. oracle_pass@1 >= repair_pass@1
    oracle_p1 = oracle_results.get("oracle_pass_at_1", 0)
    repair_p1 = rep_pk.get("pass@1", 0)
    if oracle_p1 < repair_p1 - 1e-9:
        inconsistencies.append(
            f"[CHECK 2] oracle_pass@1={oracle_p1:.4f} < repair_pass@1={repair_p1:.4f}"
        )

    # 3. error distribution sums to ~100%
    dist = error_results.get("distribution", {})
    if dist:
        total_pct = sum(v.get("percentage", 0) for v in dist.values())
        if abs(total_pct - 100.0) > 0.5:
            inconsistencies.append(
                f"[CHECK 3] Error distribution sums to {total_pct:.2f}%, expected ~100%"
            )

    # 4. total executions consistency (checked internally by ExecMetrics agent)

    # 5. problems_solved <= total_problems
    solved = gen_results.get("problems_solved", 0)
    total = gen_results.get("total_problems", 0)
    if solved > total:
        inconsistencies.append(
            f"[CHECK 5] problems_solved={solved} > total_problems={total}"
        )

    # 6. cost_per_solved >= cost_per_problem
    cps = cost_results.get("cost_per_solved_problem", 0)
    cpp = cost_results.get("cost_per_problem", 0)
    if cps < cpp - 1e-9 and cps > 0:
        inconsistencies.append(
            f"[CHECK 6] cost_per_solved={cps:.4f} < cost_per_problem={cpp:.4f}"
        )

    return {
        "passed": len(inconsistencies) == 0,
        "inconsistencies": inconsistencies,
    }


def collect_agent_anomalies(agent_results: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Collect anomalies from all agent result dicts."""
    anomalies = {}
    for name, result in agent_results.items():
        agent_anomalies = result.get("anomalies", [])
        if agent_anomalies:
            anomalies[name] = agent_anomalies
    return anomalies
