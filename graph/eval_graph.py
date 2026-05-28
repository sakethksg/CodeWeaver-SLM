"""
Main evaluation graph.

LangGraph StateGraph: load_data → process_problems → analyze → report

Supports:
  - Parallel per-problem pipeline processing via ThreadPoolExecutor
  - Parallel analysis agents via analysis_graph.py
  - Optional checkpointing via SqliteSaver for research reproducibility
  - Dry-run mode (bypass pipeline with mock data)
  - Pre-computed data loading (--from-data)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from tqdm import tqdm

from config import ExperimentConfig, CONFIG
from graph.state import EvalState
from graph.pipeline_graph import build_pipeline_graph
from graph.analysis_graph import run_analysis_and_checks
from datasets import load_humaneval, load_mbpp
from report.report_generator import ReportGenerator


# ──────────────────────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────────────────────

def load_data_node(state: EvalState) -> dict:
    """Load datasets based on config."""
    config = state["config"]
    datasets = config.get("datasets", ["humaneval", "mbpp"])

    all_problems = []
    for ds_name in datasets:
        if ds_name == "humaneval":
            all_problems.extend(load_humaneval())
        elif ds_name == "mbpp":
            all_problems.extend(load_mbpp())
        else:
            print(f"[EvalGraph] Unknown dataset: {ds_name}")

    print(f"\n[EvalGraph] Loaded {len(all_problems)} problems from {datasets}")
    return {"problems": all_problems}


def _run_pipeline_for_problem(
    pipeline_graph, problem: Dict, config: Dict
) -> Dict:
    """
    Run the pipeline subgraph for a single problem.

    Returns the problem result dict.
    """
    initial_state = {
        "problem": problem,
        "config": config,
        "candidates": [],
        "generation_time": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "execution_results": [],
        "execution_time": 0.0,
        "repair_round": 0,
        "current_failures": [],
        "all_repairs": [],
        "all_repair_results": [],
        "per_round_stats": [],
        "repair_time": 0.0,
        "repair_input_tokens": 0,
        "repair_output_tokens": 0,
        "errors": [],
        "result": {},
    }

    final_state = pipeline_graph.invoke(initial_state)
    result = final_state.get("result", {})
    # Prefer corrected errors (with was_repaired flags) from rank_node
    errors = result.pop("_corrected_errors", None) or final_state.get("errors", [])
    return result, errors


def process_problems_node(state: EvalState) -> dict:
    """
    Process all problems through the pipeline.

    Uses ThreadPoolExecutor for parallel problem processing.
    Each problem gets its own pipeline subgraph invocation.
    """
    config = state["config"]
    problems = state["problems"]
    max_concurrency = config.get("max_concurrency", 8)

    print(f"\n{'='*60}")
    print(f"[Pipeline] Processing {len(problems)} problems")
    print(f"  k={config.get('k', 10)}, temps={config.get('temperatures', [0.3, 0.8])}")
    print(f"  repair_rounds={config.get('repair_rounds', 2)}, "
          f"concurrency={max_concurrency}")
    print(f"{'='*60}\n")

    # Build pipeline graph (shared, stateless compiled graph)
    pipeline_graph = build_pipeline_graph()

    all_results = []
    all_errors = []

    # Process problems in parallel
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_problem = {
            pool.submit(
                _run_pipeline_for_problem, pipeline_graph, problem, config
            ): problem
            for problem in problems
        }

        with tqdm(total=len(problems), desc="Evaluating problems") as pbar:
            for future in as_completed(future_to_problem):
                try:
                    result, errors = future.result()
                    all_results.append(result)
                    all_errors.extend(errors)
                except Exception as e:
                    problem = future_to_problem[future]
                    print(f"\n[Pipeline] Error for {problem.get('task_id', '?')}: {e}")
                    all_results.append({
                        "task_id": problem.get("task_id", "?"),
                        "dataset": problem.get("dataset", ""),
                        "error": str(e),
                        "solved": False,
                    })
                pbar.update(1)

    return {
        "pipeline_results": all_results,
        "all_errors": all_errors,
    }


def analyze_node(state: EvalState) -> dict:
    """
    Run all 8 analysis agents in parallel, then consistency checks.
    """
    problems = state["pipeline_results"]
    all_errors = state["all_errors"]

    result = run_analysis_and_checks(problems, all_errors)

    return {
        "agent_results": result["agent_results"],
        "consistency_checks": result["consistency_checks"],
    }


def report_node(state: EvalState) -> dict:
    """
    Generate the evaluation report and save all outputs.
    """
    config = state["config"]
    output_dir = config.get("output_dir", "results")
    report_file = config.get("report_file", "evaluation_report.md")

    # Build the full results dict for the report generator
    results = {
        "config": {
            "model": config.get("model_name", "N/A"),
            "k": config.get("k", 10),
            "temperatures": config.get("temperatures", []),
            "repair_rounds": config.get("repair_rounds", 2),
            "fixes_per_failure": config.get("fixes_per_failure", 2),
            "execution_timeout": config.get("execution_timeout", 5.0),
        },
        "agent_results": state["agent_results"],
        "consistency_checks": state["consistency_checks"],
        "agent_anomalies": collect_agent_anomalies_from_state(state),
    }

    # Generate report
    report_gen = ReportGenerator(results)
    report_path = os.path.join(output_dir, report_file)
    report = report_gen.save(report_path)

    # Save raw JSON outputs
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(output_dir, "pipeline_data.json"), "w") as f:
        json.dump(state["pipeline_results"], f, indent=2, default=str)

    for name, agent_result in state["agent_results"].items():
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(agent_result, f, indent=2, default=str)

    print(f"[EvalGraph] Results saved to {output_dir}/")

    return {
        "report": report,
        "report_path": report_path,
    }


def collect_agent_anomalies_from_state(state: EvalState) -> Dict:
    """Collect anomalies from agent results in state."""
    from agents.consistency import collect_agent_anomalies
    return collect_agent_anomalies(state.get("agent_results", {}))


# ──────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────

def build_eval_graph(config: ExperimentConfig = None):
    """
    Build and compile the main evaluation graph.

    Graph:
        START → load_data → process_problems → analyze → report → END

    Args:
        config: ExperimentConfig (defaults to global CONFIG)

    Returns:
        Compiled LangGraph StateGraph
    """
    if config is None:
        config = CONFIG

    graph = StateGraph(EvalState)

    # Add nodes
    graph.add_node("load_data", load_data_node)
    graph.add_node("process_problems", process_problems_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("report", report_node)

    # Add edges
    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "process_problems")
    graph.add_edge("process_problems", "analyze")
    graph.add_edge("analyze", "report")
    graph.add_edge("report", END)

    # Compile with optional checkpointing
    if config.enable_checkpointing:
        try:
            # pyrefly: ignore [missing-import]
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string(config.checkpoint_db)
            print(f"[EvalGraph] Checkpointing enabled: {config.checkpoint_db}")
            return graph.compile(checkpointer=checkpointer)
        except ImportError:
            print("[EvalGraph] Warning: langgraph-checkpoint-sqlite not installed, "
                  "checkpointing disabled")

    return graph.compile()


def run_evaluation(
    config: ExperimentConfig = None,
    problems: List[Dict] = None,
    dry_run_data: Dict = None,
) -> Dict:
    """
    Convenience function: build graph and run the full evaluation.

    Args:
        config: ExperimentConfig
        problems: Pre-loaded problems (skips load_data)
        dry_run_data: Dict with "problems" and "errors" for dry-run mode

    Returns:
        Final EvalState dict
    """
    if config is None:
        config = CONFIG

    graph = build_eval_graph(config)

    # Serialize config for graph state
    config_dict = {
        "model_name": config.model_name,
        "api_base": config.api_base,
        "api_key": config.api_key,
        "k": config.k,
        "temperatures": config.temperatures,
        "max_tokens": config.max_tokens,
        "repair_rounds": config.repair_rounds,
        "fixes_per_failure": config.fixes_per_failure,
        "repair_temperature": config.repair_temperature,
        "execution_timeout": config.execution_timeout,
        "max_concurrency": config.max_concurrency,
        "executor_workers": config.executor_workers,
        "datasets": config.datasets,
        "output_dir": config.output_dir,
        "report_file": config.report_file,
        "gpu_cost_per_hour": config.gpu_cost_per_hour,
        "cost_per_1k_input_tokens": config.cost_per_1k_input_tokens,
        "cost_per_1k_output_tokens": config.cost_per_1k_output_tokens,
        "tradeoff_k_values": config.tradeoff_k_values,
        "tradeoff_repair_rounds": config.tradeoff_repair_rounds,
    }

    # Build initial state
    initial_state: EvalState = {
        "config": config_dict,
        "problems": problems or [],
        "pipeline_results": [],
        "all_errors": [],
        "agent_results": {},
        "consistency_checks": {},
        "report": "",
        "report_path": "",
    }

    # Handle dry-run or pre-computed data
    if dry_run_data:
        initial_state["pipeline_results"] = dry_run_data.get("problems", [])
        initial_state["all_errors"] = dry_run_data.get("errors", [])
        # Skip load_data and process_problems — go directly to analyze
        from graph.analysis_graph import run_analysis_and_checks
        analysis = run_analysis_and_checks(
            initial_state["pipeline_results"],
            initial_state["all_errors"],
        )
        initial_state["agent_results"] = analysis["agent_results"]
        initial_state["consistency_checks"] = analysis["consistency_checks"]

        # Generate report
        results = {
            "config": initial_state["config"],
            "agent_results": initial_state["agent_results"],
            "consistency_checks": initial_state["consistency_checks"],
            "agent_anomalies": collect_agent_anomalies_from_state(initial_state),
        }
        report_gen = ReportGenerator(results)
        report_path = os.path.join(config.output_dir, config.report_file)
        report = report_gen.save(report_path)

        # Save JSON outputs
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, "full_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(os.path.join(config.output_dir, "pipeline_data.json"), "w") as f:
            json.dump(initial_state["pipeline_results"], f, indent=2, default=str)
        for name, agent_result in initial_state["agent_results"].items():
            with open(os.path.join(config.output_dir, f"{name}.json"), "w") as f:
                json.dump(agent_result, f, indent=2, default=str)

        initial_state["report"] = report
        initial_state["report_path"] = report_path
        return initial_state

    # Full pipeline run via graph
    invoke_config = {}
    if config.enable_checkpointing and config.experiment_id:
        invoke_config = {"configurable": {"thread_id": config.experiment_id}}

    final_state = graph.invoke(initial_state, invoke_config or None)
    return final_state
