"""
Agent 3: Execution Metrics Agent

Tracks execution behavior across ALL pipeline stages (initial + repair).
Computes total executions, success/failure counts, average test pass rates.

Output schema matches EXECUTION_METRICS_PROMPT exactly.
"""

from typing import Any, Dict, List
import numpy as np

from agents.base_agent import BaseAgent
from agents.prompts import EXECUTION_METRICS_PROMPT


class ExecutionMetricsAgent(BaseAgent):
    """Tracks execution statistics across the entire pipeline."""
    
    SYSTEM_PROMPT = EXECUTION_METRICS_PROMPT
    
    def __init__(self):
        super().__init__("Execution Metrics Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze execution data.
        
        Input: {"problems": [{task_id, dataset, execution_results: [{passed, error_type, execution_time, tests_passed, tests_total}]}]}
        Output: matches EXECUTION_METRICS_PROMPT output schema.
        """
        self.anomalies = []  # reset
        problems = data.get("problems", [])
        
        if not problems:
            self.results = {"agent": self.name, "error": "No data", "anomalies": []}
            return self.results
        
        total_executions = 0
        total_successes = 0
        total_failures = 0
        per_problem_stats = []
        all_test_pass_rates = []
        
        # Validate: count from execution_results, not from stored totals
        # (per prompt: "count from actual execution_results")
        for p in problems:
            execs = p.get("execution_results", [])
            n_exec = len(execs)
            n_pass = sum(1 for e in execs if e.get("passed", False))
            n_fail = n_exec - n_pass
            
            total_executions += n_exec
            total_successes += n_pass
            total_failures += n_fail
            
            # Per-problem test pass rate
            test_rates = []
            for e in execs:
                total_t = e.get("tests_total", 1)
                passed_t = e.get("tests_passed", 0)
                test_rates.append(self.safe_div(passed_t, total_t))
            
            avg_rate = float(np.mean(test_rates)) if test_rates else 0.0
            all_test_pass_rates.append(avg_rate)
            
            per_problem_stats.append({
                "task_id": p["task_id"],
                "dataset": p.get("dataset", ""),
                "total_executions": n_exec,
                "successes": n_pass,
                "failures": n_fail,
                "avg_test_pass_rate": avg_rate,
            })
        
        # Validate total_executions consistency
        reported_total = sum(p.get("total_executions", 0) for p in problems)
        if reported_total != total_executions and reported_total > 0:
            self.flag_inconsistency(
                f"Counted {total_executions} executions from results, "
                f"but problems report {reported_total} total"
            )
        
        self.results = {
            "agent": self.name,
            "total_executions": total_executions,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": self.safe_div(total_successes, total_executions) * 100,
            "avg_executions_per_problem": self.safe_div(
                total_executions, len(problems)
            ),
            "avg_test_pass_rate": float(np.mean(all_test_pass_rates)) if all_test_pass_rates else 0.0,
            "per_problem": per_problem_stats,
        }
        self._attach_anomalies(self.results)
        
        return self.results
