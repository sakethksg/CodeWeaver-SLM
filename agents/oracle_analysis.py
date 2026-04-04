"""
Agent 5: Oracle Analysis Agent

Computes upper bound performance. A problem is solved if ANY candidate
(before or after repair) passes all tests.
Validates oracle >= repair (upper bound invariant).

Output schema matches ORACLE_ANALYSIS_PROMPT exactly.
"""

from typing import Any, Dict, List
import numpy as np

from agents.base_agent import BaseAgent
from agents.prompts import ORACLE_ANALYSIS_PROMPT


class OracleAnalysisAgent(BaseAgent):
    """Computes oracle (upper bound) performance with invariant checking."""
    
    SYSTEM_PROMPT = ORACLE_ANALYSIS_PROMPT
    
    def __init__(self):
        super().__init__("Oracle Analysis Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute oracle performance with upper-bound validation.
        
        Input: {"problems": [{task_id, dataset, any_candidate_passed, num_candidates_total, num_correct_total}],
                "generation_pass_at_1": float, "after_repair_pass_at_1": float}
        """
        self.anomalies = []  # reset
        problems = data.get("problems", [])
        
        if not problems:
            self.results = {"agent": self.name, "error": "No data", "anomalies": []}
            return self.results
        
        # ── Oracle pass@1 ──
        oracle_solved = sum(1 for p in problems if p.get("any_candidate_passed", False))
        oracle_pass_1 = self.safe_div(oracle_solved, len(problems))
        
        # ── Oracle pass@k (using full candidate pool) ──
        k_values = [1, 5, 10]
        oracle_pass_at_k = self.compute_pass_at_k_for_problems(
            problems, k_values,
            key_n="num_candidates_total",
            key_c="num_correct_total",
        )
        
        # ── Gap analysis ──
        generation_pass1 = data.get("generation_pass_at_1", 0.0)
        after_repair_pass1 = data.get("after_repair_pass_at_1", 0.0)
        
        # ── Validate upper bound invariant ──
        err = self.validate_upper_bound(
            oracle_pass_1, after_repair_pass1,
            "oracle_pass@1", "after_repair_pass@1"
        )
        if err:
            self.flag_inconsistency(err)
        
        err2 = self.validate_upper_bound(
            oracle_pass_1, generation_pass1,
            "oracle_pass@1", "generation_pass@1"
        )
        if err2:
            self.flag_inconsistency(err2)
        
        # ── Per-problem oracle data ──
        per_problem = []
        for p in problems:
            per_problem.append({
                "task_id": p["task_id"],
                "dataset": p.get("dataset", ""),
                "oracle_solved": p.get("any_candidate_passed", False),
                "num_candidates_total": p.get("num_candidates_total", 0),
                "num_correct_total": p.get("num_correct_total", 0),
            })
        
        self.results = {
            "agent": self.name,
            "oracle_pass_at_1": oracle_pass_1,
            "oracle_pass_at_k": {f"pass@{k}": v for k, v in oracle_pass_at_k.items()},
            "oracle_solved_count": oracle_solved,
            "total_problems": len(problems),
            "oracle_solve_rate": oracle_pass_1 * 100,
            "gap_to_oracle": {
                "from_generation_pass1": oracle_pass_1 - generation_pass1,
                "from_after_repair_pass1": oracle_pass_1 - after_repair_pass1,
            },
            "per_problem": per_problem,
        }
        self._attach_anomalies(self.results)
        
        return self.results
