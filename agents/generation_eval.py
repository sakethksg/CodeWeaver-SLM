"""
Agent 2: Generation Evaluation Agent

Evaluates initial k candidates BEFORE repair.
Computes pass@1, pass@5, pass@10, % problems solved, average test pass rate.

Output schema matches GENERATION_EVAL_PROMPT exactly.
"""

from typing import Any, Dict, List
import numpy as np

from agents.base_agent import BaseAgent
from agents.prompts import GENERATION_EVAL_PROMPT


class GenerationEvalAgent(BaseAgent):
    """Evaluates generation quality before repair."""
    
    SYSTEM_PROMPT = GENERATION_EVAL_PROMPT
    
    def __init__(self):
        super().__init__("Generation Evaluation Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze generation results.
        
        Input: {"problems": [{task_id, dataset, num_candidates, num_correct_before_repair, test_pass_rates}]}
        Output: matches GENERATION_EVAL_PROMPT output schema.
        """
        self.anomalies = []  # reset
        problems = data.get("problems", [])
        
        if not problems:
            self.results = {"agent": self.name, "error": "No problem data provided", "anomalies": []}
            return self.results
        
        # ── pass@k before repair ──
        k_values = [1, 5, 10]
        pass_at_k = self.compute_pass_at_k_for_problems(
            problems, k_values,
            key_n="num_candidates",
            key_c="num_correct_before_repair",
        )
        
        # Validate monotonicity: pass@1 <= pass@5 <= pass@10
        pk_values = [pass_at_k.get(k) for k in k_values]
        pk_labels = [f"pass@{k}" for k in k_values]
        for anomaly in self.validate_monotonic_non_decreasing(pk_values, pk_labels):
            self.flag_anomaly(anomaly)
        
        # ── % problems solved (at least 1 candidate passes) ──
        problems_solved = sum(
            1 for p in problems if p["num_correct_before_repair"] > 0
        )
        pct_solved = self.safe_div(problems_solved, len(problems)) * 100
        
        # ── Average test pass rate ──
        all_rates = []
        for p in problems:
            rates = p.get("test_pass_rates", [])
            if rates:
                all_rates.extend(rates)
        avg_test_pass_rate = float(np.mean(all_rates)) if all_rates else 0.0
        
        # ── Per-problem breakdown ──
        per_problem = []
        for p in problems:
            n = p["num_candidates"]
            c = p["num_correct_before_repair"]
            per_problem.append({
                "task_id": p["task_id"],
                "dataset": p.get("dataset", ""),
                "n": n,
                "c": c,
                "pass@1": self.pass_at_k(n, c, 1) if n >= 1 else 0.0,
                "pass@5": self.pass_at_k(n, c, min(5, n)) if n >= 5 else None,
                "pass@10": self.pass_at_k(n, c, min(10, n)) if n >= 10 else None,
                "avg_test_pass_rate": float(np.mean(p.get("test_pass_rates", [0]))),
            })
        
        self.results = {
            "agent": self.name,
            "pass_at_k_before_repair": {f"pass@{k}": v for k, v in pass_at_k.items()},
            "pct_problems_solved": pct_solved,
            "problems_solved": problems_solved,
            "total_problems": len(problems),
            "avg_test_pass_rate": avg_test_pass_rate,
            "per_problem": per_problem,
        }
        self._attach_anomalies(self.results)
        
        return self.results
