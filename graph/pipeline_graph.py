"""
Per-problem pipeline subgraph.

LangGraph StateGraph: generate → execute → [repair loop] → rank

Each node receives PipelineState and returns a partial state update.
The repair loop uses conditional edges (should_repair → repair → re_execute → should_repair).
"""

import time
from typing import Dict, List

from langgraph.graph import StateGraph, START, END

from graph.state import PipelineState
from pipeline.generator import _build_generation_prompt, CODE_STOP_SEQUENCES
from pipeline.executor import (
    execute_code_batch_parallel, ErrorType, ExecutionResult
)
from pipeline.ranker import rank_candidates


# ──────────────────────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────────────────────

def generate_node(state: PipelineState) -> dict:
    """
    Generate k candidates using batched LLM calls.

    One API call per temperature, with n= samples per call.
    """
    from pipeline.llm import create_llm, generate_batch

    problem = state["problem"]
    config = state["config"]
    k = config.get("k", 10)
    temperatures = config.get("temperatures", [0.3, 0.8])

    system_msg, user_msg = _build_generation_prompt(problem)

    candidates = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0

    # Distribute k across temperatures
    per_temp = k // len(temperatures)
    remainder = k % len(temperatures)

    for i, temp in enumerate(temperatures):
        n_samples = per_temp + (1 if i < remainder else 0)
        if n_samples == 0:
            continue

        llm = create_llm(temperature=temp)
        completions, stats = generate_batch(
            llm, system_msg, user_msg,
            n=n_samples, temperature=temp,
            stop=CODE_STOP_SEQUENCES,
        )
        total_time += stats["elapsed"]
        total_input_tokens += stats["input_tokens"]
        total_output_tokens += stats["output_tokens"]

        for code in completions:
            if problem.get("dataset") == "humaneval":
                full_code = problem["prompt"] + code
            else:
                full_code = code
            candidates.append(full_code)

    return {
        "candidates": candidates,
        "generation_time": total_time,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "repair_round": 0,
        "all_repairs": [],
        "all_repair_results": [],
        "per_round_stats": [],
        "repair_time": 0.0,
        "repair_input_tokens": 0,
        "repair_output_tokens": 0,
    }


def execute_node(state: PipelineState) -> dict:
    """
    Execute initial candidates against tests.

    Uses parallel execution via ThreadPoolExecutor.
    Identifies failures for repair and collects errors.
    """
    problem = state["problem"]
    config = state["config"]
    candidates = state["candidates"]
    test_code = problem.get("test", "")
    entry_point = problem.get("entry_point", "")
    timeout = config.get("execution_timeout", 5.0)
    max_workers = config.get("executor_workers", 4)

    start = time.perf_counter()
    results = execute_code_batch_parallel(
        candidates, test_code, entry_point, timeout, max_workers
    )
    exec_time = time.perf_counter() - start

    # Compute test pass rates
    test_pass_rates = []
    for r in results:
        rate = r.tests_passed / r.tests_total if r.tests_total > 0 else 0.0
        test_pass_rates.append(rate)

    # Identify failures for repair
    failures = []
    errors = []
    for i, r in enumerate(results):
        if not r.passed:
            failures.append({
                "slot_index": i,
                "code": candidates[i],
                "error_result": r,
            })
            errors.append({
                "task_id": problem.get("task_id", ""),
                "dataset": problem.get("dataset", ""),
                "error_type": r.error_type.value,
                "error_message": r.error_message[:200],
                "was_repaired": False,
                "stage": "generation",
                "candidate_index": i,
            })

    exec_results_dicts = [r.to_dict() for r in results]

    return {
        "execution_results": exec_results_dicts,
        "execution_time": exec_time,
        "current_failures": failures,
        "errors": errors,  # Annotated[list, add] — appended
    }


def should_repair(state: PipelineState) -> str:
    """Conditional edge: decide whether to repair or rank."""
    config = state["config"]
    max_rounds = config.get("repair_rounds", 2)
    current_round = state.get("repair_round", 0)
    failures = state.get("current_failures", [])

    if failures and current_round < max_rounds:
        return "repair"
    return "rank"


def repair_node(state: PipelineState) -> dict:
    """
    Repair all current failures via batched LLM calls.

    Builds repair prompts for all failures, sends as one batch.
    """
    from pipeline.llm import create_llm, repair_batch
    from pipeline.repairer import _build_repair_prompt, REPAIR_STOP_SEQUENCES

    problem = state["problem"]
    config = state["config"]
    failures = state["current_failures"]
    fixes_per_failure = config.get("fixes_per_failure", 2)
    repair_temp = config.get("repair_temperature", 0.3)

    # Build prompts for all failures
    prompts = []
    for f in failures:
        error_result = f["error_result"]
        if isinstance(error_result, dict):
            # Reconstruct ExecutionResult from dict
            er = ExecutionResult(
                passed=error_result.get("passed", False),
                error_type=ErrorType(error_result.get("error_type", "runtime_type")),
                error_message=error_result.get("error_message", ""),
            )
        else:
            er = error_result
        sys_msg, usr_msg = _build_repair_prompt(f["code"], er, problem)
        prompts.append((sys_msg, usr_msg))

    llm = create_llm(temperature=repair_temp)
    start = time.perf_counter()
    batch_results, stats = repair_batch(
        llm, prompts,
        n_per_prompt=fixes_per_failure,
        temperature=repair_temp,
        stop=REPAIR_STOP_SEQUENCES,
    )
    repair_time = time.perf_counter() - start

    repairs = list(state.get("all_repairs", []))
    for i, repairs_for_failure in enumerate(batch_results):
        for repaired in repairs_for_failure:
            if repaired:
                if problem.get("dataset") == "humaneval":
                    if not repaired.strip().startswith("def "):
                        repaired = problem["prompt"] + repaired
                repairs.append(repaired)

    return {
        "all_repairs": repairs,
        "repair_time": state.get("repair_time", 0.0) + repair_time,
        "repair_input_tokens": state.get("repair_input_tokens", 0) + stats["input_tokens"],
        "repair_output_tokens": state.get("repair_output_tokens", 0) + stats["output_tokens"],
    }


def re_execute_node(state: PipelineState) -> dict:
    """
    Execute repair candidates and update failure tracking.

    Identifies remaining failures for the next repair round.
    """
    problem = state["problem"]
    config = state["config"]
    all_repairs = state["all_repairs"]
    test_code = problem.get("test", "")
    entry_point = problem.get("entry_point", "")
    timeout = config.get("execution_timeout", 5.0)
    max_workers = config.get("executor_workers", 4)

    # Only execute new repairs (from the latest round)
    prev_repair_results = state.get("all_repair_results", [])
    new_repairs = all_repairs[len(prev_repair_results):]

    if not new_repairs:
        return {
            "repair_round": state.get("repair_round", 0) + 1,
            "current_failures": [],
            "per_round_stats": state.get("per_round_stats", []) + [{
                "round": state.get("repair_round", 0) + 1,
                "candidates_fixed": 0,
                "candidates_attempted": 0,
            }],
        }

    start = time.perf_counter()
    results = execute_code_batch_parallel(
        new_repairs, test_code, entry_point, timeout, max_workers
    )
    exec_time = time.perf_counter() - start

    fixed_count = sum(1 for r in results if r.passed)
    repair_round = state.get("repair_round", 0) + 1

    # New failures for next round
    new_failures = []
    errors = []
    for i, r in enumerate(results):
        if not r.passed:
            new_failures.append({
                "slot_index": i,
                "code": new_repairs[i],
                "error_result": r,
            })
            errors.append({
                "task_id": problem.get("task_id", ""),
                "dataset": problem.get("dataset", ""),
                "error_type": r.error_type.value,
                "error_message": r.error_message[:200],
                "was_repaired": False,
                "stage": f"repair_round_{repair_round}",
                "candidate_index": i,
            })

    # Accumulate repair results
    all_repair_results = prev_repair_results + [r.to_dict() for r in results]

    per_round_stats = list(state.get("per_round_stats", []))
    per_round_stats.append({
        "round": repair_round,
        "candidates_fixed": fixed_count,
        "candidates_attempted": len(new_repairs),
    })

    return {
        "all_repair_results": all_repair_results,
        "current_failures": new_failures,
        "repair_round": repair_round,
        "per_round_stats": per_round_stats,
        "execution_time": state.get("execution_time", 0.0) + exec_time,
        "errors": errors,  # Annotated[list, add] — appended
    }


def rank_node(state: PipelineState) -> dict:
    """
    Rank all candidates and build the final result dict.

    Aggregates data from all pipeline stages into the result dict
    that analysis agents consume.
    """
    problem = state["problem"]
    config = state["config"]
    candidates = state["candidates"]
    exec_results_dicts = state["execution_results"]
    all_repairs = state.get("all_repairs", [])
    all_repair_result_dicts = state.get("all_repair_results", [])

    # Reconstruct ExecutionResults for ranking
    all_codes = candidates + all_repairs

    initial_results = []
    for d in exec_results_dicts:
        r = ExecutionResult(
            passed=d["passed"],
            error_type=ErrorType(d["error_type"]),
            error_message=d.get("error_message", ""),
            execution_time=d.get("execution_time", 0),
            tests_passed=d.get("tests_passed", 0),
            tests_total=d.get("tests_total", 0),
        )
        initial_results.append(r)

    repair_results = []
    for d in all_repair_result_dicts:
        r = ExecutionResult(
            passed=d["passed"],
            error_type=ErrorType(d["error_type"]),
            error_message=d.get("error_message", ""),
            execution_time=d.get("execution_time", 0),
            tests_passed=d.get("tests_passed", 0),
            tests_total=d.get("tests_total", 0),
        )
        repair_results.append(r)

    all_results = initial_results + repair_results
    ranked = rank_candidates(all_codes, all_results)
    best_code, best_result, best_score = ranked[0] if ranked else ("", None, 0.0)

    # Compute metrics
    num_correct_before = sum(1 for r in initial_results if r.passed)
    num_fixed = sum(1 for r in repair_results if r.passed)
    num_correct_after = num_correct_before + num_fixed
    num_correct_total = sum(1 for r in all_results if r.passed)

    # Test pass rates (initial candidates only)
    test_pass_rates = []
    for r in initial_results:
        rate = r.tests_passed / r.tests_total if r.tests_total > 0 else 0.0
        test_pass_rates.append(rate)

    result = {
        "task_id": problem.get("task_id", ""),
        "dataset": problem.get("dataset", ""),
        "num_candidates": len(candidates),
        "num_correct_before_repair": num_correct_before,
        "num_correct_after_repair": num_correct_after,
        "num_failed_before_repair": len(candidates) - num_correct_before,
        "num_fixed_by_repair": num_fixed,
        "num_candidates_total": len(all_codes),
        "num_correct_total": num_correct_total,
        "any_candidate_passed": num_correct_total > 0,
        "solved": best_result.passed if best_result else False,
        "best_score": best_score,
        "generation_time": state.get("generation_time", 0),
        "execution_time": state.get("execution_time", 0),
        "repair_time": state.get("repair_time", 0),
        "total_executions": len(exec_results_dicts) + len(all_repair_result_dicts),
        "test_pass_rates": test_pass_rates,
        "execution_results": exec_results_dicts + all_repair_result_dicts,
        "per_round_stats": state.get("per_round_stats", []),
        "total_repair_rounds": config.get("repair_rounds", 2),
        "input_tokens": state.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0),
        "repair_input_tokens": state.get("repair_input_tokens", 0),
        "repair_output_tokens": state.get("repair_output_tokens", 0),
    }

    return {"result": result}


# ──────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────

def build_pipeline_graph():
    """
    Build and compile the per-problem pipeline graph.

    Graph:
        START → generate → execute → should_repair
        should_repair --[repair]--> repair → re_execute → should_repair
        should_repair --[rank]--> rank → END

    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("repair", repair_node)
    graph.add_node("re_execute", re_execute_node)
    graph.add_node("rank", rank_node)

    # Add edges
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", should_repair, {
        "repair": "repair",
        "rank": "rank",
    })
    graph.add_edge("repair", "re_execute")
    graph.add_conditional_edges("re_execute", should_repair, {
        "repair": "repair",
        "rank": "rank",
    })
    graph.add_edge("rank", END)

    return graph.compile()
