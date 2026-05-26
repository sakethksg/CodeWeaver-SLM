"""
Parallel analysis agent execution via fan-out/fan-in.

Runs 8 analysis agents concurrently using ThreadPoolExecutor,
then aggregates results and runs consistency checks.

This module is used by eval_graph.py rather than being a standalone
LangGraph subgraph, because the analysis agents are pure computation
(no LLM calls) and don't need LangGraph's state management overhead.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from agents.generation_eval import GenerationEvalAgent
from agents.execution_metrics import ExecutionMetricsAgent
from agents.repair_analysis import RepairAnalysisAgent
from agents.oracle_analysis import OracleAnalysisAgent
from agents.efficiency_cost import EfficiencyCostAgent
from agents.error_analysis import ErrorAnalysisAgent
from agents.dataset_analysis import DatasetAnalysisAgent
from agents.tradeoff_analysis import TradeoffAnalysisAgent
from agents.consistency import run_consistency_checks, collect_agent_anomalies
from agents.base_agent import BaseAgent


# Agent registry: name → class
AGENT_REGISTRY = {
    "generation_eval": GenerationEvalAgent,
    "execution_metrics": ExecutionMetricsAgent,
    "repair_analysis": RepairAnalysisAgent,
    "oracle_analysis": OracleAnalysisAgent,
    "efficiency_cost": EfficiencyCostAgent,
    "error_analysis": ErrorAnalysisAgent,
    "dataset_analysis": DatasetAnalysisAgent,
    "tradeoff_analysis": TradeoffAnalysisAgent,
}


def _run_single_agent(
    agent_name: str,
    problems: List[Dict],
    all_errors: List[Dict],
    agent_results_so_far: Dict[str, Dict] = None,
) -> Dict[str, Any]:
    """
    Run a single analysis agent.

    Args:
        agent_name: Key from AGENT_REGISTRY
        problems: All pipeline results
        all_errors: All error records
        agent_results_so_far: Results from previously-completed agents
            (used by oracle_analysis to get pass@1 values)

    Returns:
        {agent_name: result_dict}
    """
    agent_cls = AGENT_REGISTRY[agent_name]
    agent = agent_cls()

    if agent_name == "error_analysis":
        # Error agent takes errors, not problems
        data = {"errors": all_errors}
    elif agent_name == "oracle_analysis":
        # Oracle needs pass@1 from generation and repair
        # Compute inline to avoid dependency on other agents' outputs
        gen_agent = GenerationEvalAgent()
        gen_result = gen_agent.analyze({"problems": problems})
        repair_agent = RepairAnalysisAgent()
        repair_result = repair_agent.analyze({"problems": problems})

        data = {
            "problems": problems,
            "generation_pass_at_1": gen_result.get(
                "pass_at_k_before_repair", {}
            ).get("pass@1", 0.0),
            "after_repair_pass_at_1": repair_result.get(
                "pass_at_k_after_repair", {}
            ).get("pass@1", 0.0),
        }
    else:
        data = {"problems": problems}

    result = agent.analyze(data)
    return {agent_name: result}


def run_all_agents_parallel(
    problems: List[Dict],
    all_errors: List[Dict],
    max_workers: int = 8,
) -> Dict[str, Dict]:
    """
    Run all 8 analysis agents in parallel.

    Uses ThreadPoolExecutor since all agents are pure computation (no I/O).
    Oracle agent computes its own pass@1 dependencies inline.

    Args:
        problems: All pipeline result dicts
        all_errors: All error records
        max_workers: Number of parallel workers

    Returns:
        Dict of {agent_name: result_dict}
    """
    agent_results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_single_agent, name, problems, all_errors
            ): name
            for name in AGENT_REGISTRY
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                agent_results.update(result)
            except Exception as e:
                print(f"[Analysis] Agent {name} failed: {e}")
                agent_results[name] = {"agent": name, "error": str(e)}

    return agent_results


def run_analysis_and_checks(
    problems: List[Dict],
    all_errors: List[Dict],
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Run all agents in parallel, then run consistency checks.

    Returns:
        {
            "agent_results": {...},
            "consistency_checks": {"passed": bool, "inconsistencies": [...]},
            "agent_anomalies": {...}
        }
    """
    print("\n[Analysis] Running 8 analysis agents in parallel...")
    agent_results = run_all_agents_parallel(problems, all_errors, max_workers)

    # Print agent completion
    for name in AGENT_REGISTRY:
        status = "[OK]" if "error" not in agent_results.get(name, {}) else "[FAIL]"
        print(f"  {status} {name}")

    # Run consistency checks
    print("\n[Analysis] Running consistency checks...")
    checks = run_consistency_checks(
        gen_results=agent_results.get("generation_eval", {}),
        repair_results=agent_results.get("repair_analysis", {}),
        oracle_results=agent_results.get("oracle_analysis", {}),
        exec_results=agent_results.get("execution_metrics", {}),
        error_results=agent_results.get("error_analysis", {}),
        cost_results=agent_results.get("efficiency_cost", {}),
    )

    if checks["passed"]:
        print("  [OK] All consistency checks passed")
    else:
        for ic in checks["inconsistencies"]:
            print(f"  [WARN] {ic}")

    # Collect anomalies
    anomalies = collect_agent_anomalies(agent_results)
    if anomalies:
        total = sum(len(v) for v in anomalies.values())
        print(f"  [WARN] {total} agent anomalies detected")

    return {
        "agent_results": agent_results,
        "consistency_checks": checks,
        "agent_anomalies": anomalies,
    }
