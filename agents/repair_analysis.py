"""
Agent 4: Repair Analysis Agent

Evaluates repair loop effectiveness with anomaly detection.
Computes pass@k after repair, delta improvement, diminishing returns.
Flags anomalies if repair degrades performance (delta < 0).

Output schema matches REPAIR_ANALYSIS_PROMPT exactly.
"""

from typing import Any, Dict, List
import numpy as np

from agents.base_agent import BaseAgent
from agents.prompts import REPAIR_ANALYSIS_PROMPT


class RepairAnalysisAgent(BaseAgent):
    """Evaluates the effectiveness of the repair loop."""
    
    SYSTEM_PROMPT = REPAIR_ANALYSIS_PROMPT
    
    def __init__(self):
        super().__init__("Repair Analysis Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze repair results with anomaly detection.
        
        Input: {"problems": [{task_id, dataset, num_candidates, num_correct_before_repair,
                num_correct_after_repair, num_failed_before_repair, num_fixed_by_repair,
                per_round_stats: [{round, candidates_fixed, candidates_attempted}]}]}
        """
        self.anomalies = []  # reset
        problems = data.get("problems", [])
        
        if not problems:
            self.results = {"agent": self.name, "error": "No data", "anomalies": []}
            return self.results
        
        # ── pass@k before and after repair ──
        k_values = [1, 5, 10]
        pass_at_k_after = self.compute_pass_at_k_for_problems(
            problems, k_values,
            key_n="num_candidates",
            key_c="num_correct_after_repair",
        )
        
        pass_at_k_before = self.compute_pass_at_k_for_problems(
            problems, k_values,
            key_n="num_candidates",
            key_c="num_correct_before_repair",
        )
        
        # ── Delta improvement with anomaly detection ──
        delta = {}
        for k in k_values:
            before = pass_at_k_before.get(k)
            after = pass_at_k_after.get(k)
            if before is None or after is None:
                delta[f"delta_pass@{k}"] = None
                continue

            d = after - before
            delta[f"delta_pass@{k}"] = d

            # Anomaly: repair should NOT decrease performance
            err = self.validate_upper_bound(after, before, f"pass@{k}_after", f"pass@{k}_before")
            if err:
                self.flag_anomaly(f"Repair degradation: {err}")
        
        # ── Repair success rate ──
        total_failed = sum(p.get("num_failed_before_repair", 0) for p in problems)
        total_fixed = sum(p.get("num_fixed_by_repair", 0) for p in problems)
        repair_success_rate = self.safe_div(total_fixed, total_failed) * 100
        
        # ── Per-round analysis (diminishing returns) ──
        max_rounds = max(
            len(p.get("per_round_stats", [])) for p in problems
        ) if problems else 0
        
        per_round_agg = []
        for r in range(max_rounds):
            round_fixed = 0
            round_attempted = 0
            for p in problems:
                rounds = p.get("per_round_stats", [])
                if r < len(rounds):
                    round_fixed += rounds[r].get("candidates_fixed", 0)
                    round_attempted += rounds[r].get("candidates_attempted", 0)
            
            per_round_agg.append({
                "round": r + 1,
                "candidates_fixed": round_fixed,
                "candidates_attempted": round_attempted,
                "fix_rate": self.safe_div(round_fixed, round_attempted) * 100,
            })
        
        # ── Diminishing returns ──
        diminishing_returns = []
        for i in range(1, len(per_round_agg)):
            prev_rate = per_round_agg[i - 1]["fix_rate"]
            curr_rate = per_round_agg[i]["fix_rate"]
            diminishing_returns.append({
                "from_round": i,
                "to_round": i + 1,
                "rate_change": curr_rate - prev_rate,
            })
        
        self.results = {
            "agent": self.name,
            "pass_at_k_after_repair": {f"pass@{k}": v for k, v in pass_at_k_after.items()},
            "pass_at_k_before_repair": {f"pass@{k}": v for k, v in pass_at_k_before.items()},
            "delta_improvement": delta,
            "repair_success_rate": repair_success_rate,
            "total_failed_candidates": total_failed,
            "total_fixed_by_repair": total_fixed,
            "candidates_fixed_ratio": f"{total_fixed}/{total_failed}",
            "per_round_analysis": per_round_agg,
            "diminishing_returns": diminishing_returns,
        }
        self._attach_anomalies(self.results)
        
        return self.results
