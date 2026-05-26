"""
Typed state definitions for LangGraph evaluation graphs.

Two state schemas:
  1. PipelineState: Per-problem pipeline (generate → execute → repair → rank)
  2. EvalState: Main evaluation orchestration (load → process → analyze → report)
"""

import operator
from typing import TypedDict, Annotated, Optional


def _merge_dicts(left: dict, right: dict) -> dict:
    """Custom reducer: merge dicts (for collecting agent results)."""
    merged = left.copy() if left else {}
    if right:
        merged.update(right)
    return merged


class PipelineState(TypedDict):
    """State for the per-problem pipeline subgraph."""

    # ── Input ──
    problem: dict                                    # problem dict (task_id, prompt, test, ...)
    config: dict                                     # serialized ExperimentConfig fields

    # ── Generation ──
    candidates: list                                 # generated code candidates
    generation_time: float                           # seconds for generation
    input_tokens: int                                # generation prompt tokens
    output_tokens: int                               # generation completion tokens

    # ── Execution ──
    execution_results: list                          # ExecutionResult.to_dict() for initial candidates
    execution_time: float                            # total execution time (initial + repair)

    # ── Repair loop ──
    repair_round: int                                # current repair round (0-indexed)
    current_failures: list                           # [{slot_index, code, error_result}] for repair
    all_repairs: list                                # accumulated repaired code strings
    all_repair_results: list                         # accumulated repair execution results
    per_round_stats: list                            # [{round, candidates_fixed, candidates_attempted}]
    repair_time: float                               # total repair time
    repair_input_tokens: int
    repair_output_tokens: int

    # ── Errors (append-only) ──
    errors: Annotated[list, operator.add]            # error records collected across stages

    # ── Final ──
    result: dict                                     # aggregated per-problem result dict


class EvalState(TypedDict):
    """State for the main evaluation graph."""

    # ── Config ──
    config: dict                                     # serialized ExperimentConfig

    # ── Data loading ──
    problems: list                                   # loaded dataset problems

    # ── Pipeline results (append-only for parallel problem processing) ──
    pipeline_results: Annotated[list, operator.add]  # per-problem result dicts
    all_errors: Annotated[list, operator.add]        # all error records

    # ── Analysis ──
    agent_results: Annotated[dict, _merge_dicts]     # {agent_key: result_dict}
    consistency_checks: dict                         # orchestrator validation results

    # ── Report ──
    report: str                                      # final markdown report
    report_path: str                                 # saved report file path
