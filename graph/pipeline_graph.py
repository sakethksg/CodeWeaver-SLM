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
from pipeline.generator import (
    _build_generation_prompt, _finalize_humaneval_code, _finalize_mbpp_code,
    CODE_STOP_SEQUENCES,
)
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
    from pipeline.llm import create_llm, generate_batch, normalize_code_output

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
            normalized = normalize_code_output(code)
            if problem.get("dataset") == "humaneval":
                full_code = _finalize_humaneval_code(problem["prompt"], normalized)
            else:
                full_code = _finalize_mbpp_code(normalized)
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
    from pipeline.llm import create_llm, repair_batch, normalize_code_output
    from pipeline.repairer import (
        _build_repair_prompt, _finalize_humaneval_repair, REPAIR_STOP_SEQUENCES,
    )

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
                normalized = normalize_code_output(repaired)
                if problem.get("dataset") == "humaneval":
                    normalized = _finalize_humaneval_repair(problem["prompt"], normalized)
                else:
                    normalized = _finalize_mbpp_code(normalized)
                repairs.append(normalized)

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

    repair_round = state.get("repair_round", 0) + 1

    # Track which initial-failure slot indices were fixed this round.
    # current_failures carries the slot_index from the original candidate list.
    current_failures_list = state.get("current_failures", [])
    # Build a mapping: repair candidate index -> original slot index
    # Each failure spawns fixes_per_failure repair candidates
    fixes_per_failure = config.get("fixes_per_failure", 2)
    slot_for_repair = []
    for f in current_failures_list:
        slot_idx = f.get("slot_index", 0)
        for _ in range(fixes_per_failure):
            slot_for_repair.append(slot_idx)
    # Trim to actual number of new repairs (in case of mismatch)
    slot_for_repair = slot_for_repair[:len(results)]

    # Collect previously-fixed slots from earlier rounds
    already_fixed_slots = set()
    for prev_stat in state.get("per_round_stats", []):
        for idx in prev_stat.get("fixed_slot_indices", []):
            already_fixed_slots.add(idx)

    # Identify unique slots fixed THIS round (not already fixed)
    fixed_slots_this_round = set()
    for i, r in enumerate(results):
        if r.passed and i < len(slot_for_repair):
            slot_idx = slot_for_repair[i]
            if slot_idx not in already_fixed_slots:
                fixed_slots_this_round.add(slot_idx)

    # New failures for next round (only slots not yet fixed)
    all_fixed = already_fixed_slots | fixed_slots_this_round
    new_failures = []
    errors = []
    for i, r in enumerate(results):
        if not r.passed:
            slot_idx = slot_for_repair[i] if i < len(slot_for_repair) else i
            if slot_idx not in all_fixed:
                new_failures.append({
                    "slot_index": slot_idx,
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
                "candidate_index": slot_idx if i < len(slot_for_repair) else i,
            })

    # Accumulate repair results
    all_repair_results = prev_repair_results + [r.to_dict() for r in results]

    per_round_stats = list(state.get("per_round_stats", []))
    per_round_stats.append({
        "round": repair_round,
        "candidates_fixed": len(fixed_slots_this_round),
        "candidates_attempted": len(set(slot_for_repair) - already_fixed_slots),
        "fixed_slot_indices": list(fixed_slots_this_round),
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

    # Compute metrics — slot-level deduplication
    num_correct_before = sum(1 for r in initial_results if r.passed)

    # Determine which initially-failed slots were fixed by repair.
    # Use per_round_stats which now tracks fixed_slot_indices.
    per_round_stats = state.get("per_round_stats", [])
    fixed_slots = set()
    for rs in per_round_stats:
        for idx in rs.get("fixed_slot_indices", []):
            fixed_slots.add(idx)
    num_fixed_by_repair = len(fixed_slots)
    num_correct_after = num_correct_before + num_fixed_by_repair
    num_correct_total = sum(1 for r in all_results if r.passed)

    # Test pass rates (initial candidates only)
    test_pass_rates = []
    for r in initial_results:
        rate = r.tests_passed / r.tests_total if r.tests_total > 0 else 0.0
        test_pass_rates.append(rate)

    # Rebuild errors with correct was_repaired flags.
    # Errors from state are append-only and always have was_repaired=False.
    # We retroactively mark generation-stage errors for fixed slots.
    raw_errors = list(state.get("errors", []))
    for err in raw_errors:
        if (err.get("stage") == "generation"
                and err.get("candidate_index") in fixed_slots):
            err["was_repaired"] = True

    result = {
        "task_id": problem.get("task_id", ""),
        "dataset": problem.get("dataset", ""),
        "num_candidates": len(candidates),
        "num_correct_before_repair": num_correct_before,
        "num_correct_after_repair": num_correct_after,
        "num_failed_before_repair": len(candidates) - num_correct_before,
        "num_fixed_by_repair": num_fixed_by_repair,
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
        "per_round_stats": per_round_stats,
        "total_repair_rounds": config.get("repair_rounds", 2),
        "input_tokens": state.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0),
        "repair_input_tokens": state.get("repair_input_tokens", 0),
        "repair_output_tokens": state.get("repair_output_tokens", 0),
    }

    # Override errors in result so eval_graph gets corrected was_repaired flags
    result["_corrected_errors"] = raw_errors

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
