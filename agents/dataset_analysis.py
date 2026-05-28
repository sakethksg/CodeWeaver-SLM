"""
Agent 8: Dataset Analysis Agent

Provides per-dataset (HumanEval, MBPP) breakdown of all key metrics.
Validates: after_repair >= before_repair, oracle_upper_bound >= pass@1_after.
Does NOT compute an "overall" row (Orchestrator's responsibility).

Output schema matches DATASET_ANALYSIS_PROMPT exactly.
"""

from typing import Any, Dict, List
import numpy as np

from agents.base_agent import BaseAgent
from agents.prompts import DATASET_ANALYSIS_PROMPT


class DatasetAnalysisAgent(BaseAgent):
    """Aggregates metrics per dataset with cross-checks."""
    
    SYSTEM_PROMPT = DATASET_ANALYSIS_PROMPT
    
    def __init__(self):
        super().__init__("Dataset Analysis Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze per-dataset performance with validation.
        
        Input: {"problems": [{task_id, dataset, num_candidates, num_correct_before_repair,
            num_correct_after_repair, any_candidate_passed, test_pass_rates}]}
        """
        self.anomalies = []  # reset
        problems = data.get("problems", [])
        
        if not problems:
            self.results = {"agent": self.name, "error": "No data", "anomalies": []}
            return self.results
        
        # Group by dataset
        datasets = {}
        for p in problems:
            ds = p.get("dataset", "unknown")
            if ds not in datasets:
                datasets[ds] = []
            datasets[ds].append(p)
        
        k_values = [1, 5, 10]
        per_dataset = {}
        
        for ds_name, ds_problems in datasets.items():
            # ── pass@k before repair ──
            pass_before = self.compute_pass_at_k_for_problems(
                ds_problems, k_values,
                key_n="num_candidates",
                key_c="num_correct_before_repair",
            )
            
            # ── pass@k after repair ──
            pass_after = self.compute_pass_at_k_for_problems(
                ds_problems, k_values,
                key_n="num_candidates",
                key_c="num_correct_after_repair",
            )
            
            # ── Oracle solve-rate upper bound ──
            oracle_solved = sum(
                1 for p in ds_problems if p.get("any_candidate_passed", False)
            )
            oracle_upper_bound = self.safe_div(oracle_solved, len(ds_problems))
            
            # ── Validate invariants per dataset ──
            for k in k_values:
                b = pass_before.get(k)
                a = pass_after.get(k)
                o = oracle_upper_bound
                if a is None or b is None:
                    continue

                err = self.validate_upper_bound(a, b, f"{ds_name} pass@{k}_after", f"{ds_name} pass@{k}_before")
                if err:
                    self.flag_anomaly(err)
                
                if k != 1:
                    continue

                err2 = self.validate_upper_bound(o, a, f"{ds_name} oracle_upper_bound", f"{ds_name} pass@{k}_after")
                if err2:
                    self.flag_anomaly(err2)
            
            # ── Average score (test pass rate) ──
            all_rates = []
            for p in ds_problems:
                rates = p.get("test_pass_rates", [])
                if rates:
                    all_rates.extend(rates)
            avg_score = float(np.mean(all_rates)) if all_rates else 0.0
            
            per_dataset[ds_name] = {
                "num_problems": len(ds_problems),
                "pass_at_k_before_repair": {
                    f"pass@{k}": v for k, v in pass_before.items()
                },
                "pass_at_k_after_repair": {
                    f"pass@{k}": v for k, v in pass_after.items()
                },
                "oracle_upper_bound": oracle_upper_bound,
                "oracle_solve_rate": self.safe_div(
                    oracle_solved, len(ds_problems)
                ) * 100,
                "avg_score": avg_score,
            }
        
        self.results = {
            "agent": self.name,
            "per_dataset": per_dataset,
        }
        self._attach_anomalies(self.results)
        
        return self.results
