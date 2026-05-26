"""
Multi-Agent Evaluation System — Main Entry Point

LangGraph-based evaluation pipeline for code generation architectures.
Runs the full evaluation pipeline on HumanEval and MBPP benchmarks.

Usage:
    python main.py                           # Full evaluation
    python main.py --k 5                     # Override k
    python main.py --datasets humaneval      # Single dataset
    python main.py --dry-run                 # Test pipeline with mock data
    python main.py --from-data results/pipeline_data.json  # Re-analyze
    python main.py --checkpoint              # Enable checkpointing
    python main.py --visualize               # Print graph diagram
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Evaluation System for Code Generation (LangGraph)"
    )

    parser.add_argument(
        "--k", type=int, default=None,
        help=f"Number of candidate samples per problem (default: {CONFIG.k})"
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=None,
        help=f"Sampling temperatures (default: {CONFIG.temperatures})"
    )
    parser.add_argument(
        "--repair-rounds", type=int, default=None,
        help=f"Number of repair rounds (default: {CONFIG.repair_rounds})"
    )
    parser.add_argument(
        "--fixes-per-failure", type=int, default=None,
        help=f"Repair candidates per failure (default: {CONFIG.fixes_per_failure})"
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help=f"Execution timeout in seconds (default: {CONFIG.execution_timeout})"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        choices=["humaneval", "mbpp"],
        help=f"Datasets to evaluate (default: {CONFIG.datasets})"
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help=f"vLLM API base URL (default: {CONFIG.api_base})"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Model name (default: {CONFIG.model_name})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {CONFIG.output_dir})"
    )
    parser.add_argument(
        "--gpu-cost", type=float, default=None,
        help=f"GPU cost per hour in $ (default: {CONFIG.gpu_cost_per_hour})"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=None,
        help=f"Max parallel problems (default: {CONFIG.max_concurrency})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run pipeline with mock data (no LLM calls)"
    )
    parser.add_argument(
        "--from-data", type=str, default=None,
        help="Path to pre-computed pipeline_data.json to skip pipeline execution"
    )
    parser.add_argument(
        "--checkpoint", action="store_true",
        help="Enable graph checkpointing for research reproducibility"
    )
    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="Experiment ID for checkpoint isolation"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Print the graph diagram in Mermaid format and exit"
    )

    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """Create experiment config from CLI arguments."""
    config = ExperimentConfig()

    if args.k is not None:
        config.k = args.k
    if args.temperatures is not None:
        config.temperatures = args.temperatures
    if args.repair_rounds is not None:
        config.repair_rounds = args.repair_rounds
    if args.fixes_per_failure is not None:
        config.fixes_per_failure = args.fixes_per_failure
    if args.timeout is not None:
        config.execution_timeout = args.timeout
    if args.datasets is not None:
        config.datasets = args.datasets
    if args.api_base is not None:
        config.api_base = args.api_base
    if args.model is not None:
        config.model_name = args.model
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.gpu_cost is not None:
        config.gpu_cost_per_hour = args.gpu_cost
    if args.max_concurrency is not None:
        config.max_concurrency = args.max_concurrency
    if args.checkpoint:
        config.enable_checkpointing = True
    if args.experiment_id is not None:
        config.experiment_id = args.experiment_id

    return config


def run_dry_run(config: ExperimentConfig) -> dict:
    """Run a dry-run with synthetic mock data to test the full system."""
    import numpy as np

    print("=" * 60)
    print("[DRY RUN] Generating synthetic evaluation data...")
    print("=" * 60)

    np.random.seed(42)

    mock_problems = []
    mock_errors = []

    # Generate mock data for both datasets
    datasets = {
        "humaneval": 164,
        "mbpp": 427,
    }

    for ds_name in config.datasets:
        n_problems = datasets.get(ds_name, 100)

        for i in range(n_problems):
            k = config.k

            # Simulate generation: some pass, some fail
            n_correct_before = np.random.binomial(k, 0.35)
            n_failed = k - n_correct_before

            # Simulate repair: fix some failures
            n_fixed = 0
            per_round = []
            remaining_failures = n_failed

            for r in range(config.repair_rounds):
                fixes = np.random.binomial(
                    remaining_failures * config.fixes_per_failure, 0.2
                )
                fixes = min(fixes, remaining_failures)
                per_round.append({
                    "round": r + 1,
                    "candidates_fixed": fixes,
                    "candidates_attempted": remaining_failures,
                })
                n_fixed += fixes
                remaining_failures -= fixes

            n_correct_after = n_correct_before + n_fixed
            n_total_candidates = k + n_failed * config.fixes_per_failure * config.repair_rounds
            n_correct_total = n_correct_after

            # Simulate test pass rates
            test_rates = []
            for j in range(k):
                if j < n_correct_before:
                    test_rates.append(1.0)
                else:
                    test_rates.append(np.random.uniform(0, 0.8))

            # Simulate timings
            gen_time = np.random.uniform(1, 5)
            exec_time = np.random.uniform(0.1, 2) * k
            repair_time = (
                np.random.uniform(0.5, 3) * config.repair_rounds
                if n_failed > 0 else 0
            )

            # Simulate execution results with granular error taxonomy
            exec_results = []
            error_types = [
                "syntax_error", "runtime_type", "runtime_value",
                "runtime_resource", "logical_error", "timeout"
            ]
            error_weights = [0.12, 0.18, 0.15, 0.05, 0.40, 0.10]

            for j in range(k):
                if j < n_correct_before:
                    exec_results.append({
                        "passed": True,
                        "error_type": "none",
                        "error_message": "",
                        "execution_time": np.random.uniform(0.01, 0.5),
                        "tests_passed": 5,
                        "tests_total": 5,
                    })
                else:
                    etype = np.random.choice(error_types, p=error_weights)
                    exec_results.append({
                        "passed": False,
                        "error_type": etype,
                        "error_message": f"Mock {etype}",
                        "execution_time": np.random.uniform(0.01, 0.5),
                        "tests_passed": np.random.randint(0, 4),
                        "tests_total": 5,
                    })
                    mock_errors.append({
                        "task_id": f"{ds_name.upper()}/{i}",
                        "dataset": ds_name,
                        "error_type": etype,
                        "was_repaired": False,
                        "stage": "generation",
                    })

            # Mark some errors as repaired
            repaired_count = 0
            for err in mock_errors:
                if (err["task_id"] == f"{ds_name.upper()}/{i}"
                        and not err["was_repaired"]
                        and repaired_count < n_fixed):
                    err["was_repaired"] = True
                    repaired_count += 1

            problem = {
                "task_id": f"{ds_name.upper()}/{i}",
                "dataset": ds_name,
                "num_candidates": k,
                "num_correct_before_repair": n_correct_before,
                "num_correct_after_repair": n_correct_after,
                "num_failed_before_repair": n_failed,
                "num_fixed_by_repair": n_fixed,
                "num_candidates_total": n_total_candidates,
                "num_correct_total": n_correct_total,
                "any_candidate_passed": n_correct_total > 0,
                "solved": n_correct_after > 0,
                "best_score": 1.0 if n_correct_after > 0 else max(test_rates),
                "generation_time": gen_time,
                "execution_time": exec_time,
                "repair_time": repair_time,
                "total_executions": len(exec_results),
                "test_pass_rates": test_rates,
                "execution_results": exec_results,
                "per_round_stats": per_round,
                "total_repair_rounds": config.repair_rounds,
                "input_tokens": np.random.randint(500, 2000),
                "output_tokens": np.random.randint(200, 1000),
                "repair_input_tokens": (
                    np.random.randint(500, 3000) if n_failed > 0 else 0
                ),
                "repair_output_tokens": (
                    np.random.randint(200, 1500) if n_failed > 0 else 0
                ),
            }
            mock_problems.append(problem)

    return {"problems": mock_problems, "errors": mock_errors}


def main():
    args = parse_args()
    config = create_config_from_args(args)

    # Handle --visualize
    if args.visualize:
        from graph.eval_graph import build_eval_graph
        from graph.pipeline_graph import build_pipeline_graph

        print("\n=== Main Evaluation Graph ===")
        eval_graph = build_eval_graph(config)
        try:
            print(eval_graph.get_graph().draw_mermaid())
        except Exception:
            print("(Mermaid diagram not available)")

        print("\n=== Pipeline Subgraph ===")
        pipeline_graph = build_pipeline_graph()
        try:
            print(pipeline_graph.get_graph().draw_mermaid())
        except Exception:
            print("(Mermaid diagram not available)")
        return

    # Import graph runner
    from graph.eval_graph import run_evaluation

    print("\n" + "=" * 60)
    print("  [*] Multi-Agent Evaluation System (LangGraph)")
    print("  Execution-Guided Code Generation Evaluation")
    print("=" * 60)
    print(f"\n  Model:          {config.model_name}")
    print(f"  Datasets:       {config.datasets}")
    print(f"  k:              {config.k}")
    print(f"  Temperatures:   {config.temperatures}")
    print(f"  Repair Rounds:  {config.repair_rounds}")
    print(f"  Fixes/Failure:  {config.fixes_per_failure}")
    print(f"  Timeout:        {config.execution_timeout}s")
    print(f"  Concurrency:    {config.max_concurrency}")
    print(f"  Checkpointing:  {'ON' if config.enable_checkpointing else 'OFF'}")
    print(f"  Output:         {config.output_dir}/")
    print()

    start_time = time.perf_counter()

    if args.dry_run:
        # Dry-run: generate mock data, skip pipeline
        dry_data = run_dry_run(config)
        final_state = run_evaluation(config, dry_run_data=dry_data)
    elif args.from_data:
        # Load pre-computed pipeline data
        print(f"[Main] Loading pre-computed data from {args.from_data}")
        with open(args.from_data, "r") as f:
            pipeline_data = json.load(f)
        final_state = run_evaluation(
            config, dry_run_data={"problems": pipeline_data, "errors": []}
        )
    else:
        # Full pipeline execution via LangGraph
        final_state = run_evaluation(config)

    elapsed = time.perf_counter() - start_time

    report_path = final_state.get("report_path", "")
    report = final_state.get("report", "")

    print(f"\n{'='*60}")
    print(f"  [OK] Evaluation complete in {elapsed:.1f}s")
    print(f"  [REPORT] Report:  {report_path}")
    print(f"  [DATA]   Data:    {config.output_dir}/")
    print(f"{'='*60}\n")

    # Print summary to console (ASCII-safe for Windows)
    if report:
        safe_report = report.encode("ascii", errors="replace").decode("ascii")
        print(safe_report[:3000])
        if len(report) > 3000:
            print(f"\n  ... (full report saved to {report_path})")


if __name__ == "__main__":
    main()
