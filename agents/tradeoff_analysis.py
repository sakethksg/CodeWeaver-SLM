"""
Agent 9: Tradeoff Analysis Agent

Analyzes latency vs accuracy and cost vs performance tradeoffs.
Uses explicit methodology:
  - Best k = argmax(efficiency) where efficiency = accuracy / latency
  - Optimal repair rounds = last round where marginal_gain > 0.01
Validates accuracy monotonicity with k.

Output schema matches TRADEOFF_ANALYSIS_PROMPT exactly.
"""

from typing import Any, Dict, List
import numpy as np

from agents.base_agent import BaseAgent
from agents.prompts import TRADEOFF_ANALYSIS_PROMPT
from config import CONFIG


class TradeoffAnalysisAgent(BaseAgent):
    """Analyzes scaling behavior and tradeoffs with validation."""
    
    SYSTEM_PROMPT = TRADEOFF_ANALYSIS_PROMPT
    MARGINAL_GAIN_THRESHOLD = 0.01  # per prompt specification
    
    def __init__(self):
        super().__init__("Tradeoff Analysis Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze tradeoffs with anomaly detection for non-monotonic accuracy.
        
        Input: {"problems": [{task_id, num_candidates, num_correct_before_repair,
                generation_time, execution_time, repair_time, total_repair_rounds,
                per_round_stats}]}
        """
        self.anomalies = []  # reset
        problems = data.get("problems", [])
        sweep_results = data.get("sweep_results", None)
        
        if not problems:
            self.results = {"agent": self.name, "error": "No data", "anomalies": []}
            return self.results
        
        if sweep_results:
            return self._analyze_sweep(sweep_results)
        
        return self._analyze_from_problems(problems)
    
    def _analyze_from_problems(self, problems: List[Dict]) -> Dict[str, Any]:
        """Simulate tradeoffs by sub-sampling existing results."""
        k_values = CONFIG.tradeoff_k_values
        repair_values = CONFIG.tradeoff_repair_rounds
        
        # ── Vary k (number of samples) ──
        k_curve = []
        for k in k_values:
            pass_at_k_scores = []
            latencies = []
            
            for p in problems:
                n = p.get("num_candidates", 0)
                c = p.get("num_correct_before_repair", 0)
                if n >= k:
                    score = self.pass_at_k(n, c, k)
                    pass_at_k_scores.append(score)
                
                # Scale latency linearly with k (per prompt methodology)
                full_gen_time = p.get("generation_time", 0)
                full_k = p.get("num_candidates", 1)
                scaled_gen_time = full_gen_time * (k / max(full_k, 1))
                
                full_exec_time = p.get("execution_time", 0)
                scaled_exec_time = full_exec_time * (k / max(full_k, 1))
                
                latencies.append(scaled_gen_time + scaled_exec_time)
            
            avg_accuracy = float(np.mean(pass_at_k_scores)) if pass_at_k_scores else 0.0
            avg_latency = float(np.mean(latencies)) if latencies else 0.0
            
            k_curve.append({
                "k": k,
                "accuracy": avg_accuracy,
                "avg_latency": avg_latency,
                "efficiency": self.safe_div(avg_accuracy, avg_latency),
            })
        
        # ── Validate accuracy monotonicity with k ──
        k_accuracies = [pt["accuracy"] for pt in k_curve]
        k_labels = [f"k={pt['k']}" for pt in k_curve]
        for anomaly in self.validate_monotonic_non_decreasing(k_accuracies, k_labels):
            self.flag_anomaly(f"Accuracy vs k: {anomaly}")
        
        # ── Vary repair rounds ──
        repair_curve = []
        for r in repair_values:
            pass_at_k_scores = []
            latencies = []
            
            for p in problems:
                n = p.get("num_candidates", 0)
                c_before = p.get("num_correct_before_repair", 0)
                per_round = p.get("per_round_stats", [])
                
                # Cumulative fixes up to round r
                c_at_r = c_before
                for round_stat in per_round[:r]:
                    c_at_r += round_stat.get("candidates_fixed", 0)
                
                if n >= 1:
                    pass_at_k_scores.append(self.pass_at_k(n, min(c_at_r, n), 1))
                
                # Latency: gen + exec + scaled repair
                gen_time = p.get("generation_time", 0)
                exec_time = p.get("execution_time", 0)
                repair_time = p.get("repair_time", 0)
                
                total_rounds = p.get("total_repair_rounds", CONFIG.repair_rounds)
                scaled_repair = repair_time * (r / max(total_rounds, 1)) if r > 0 else 0
                
                latencies.append(gen_time + exec_time + scaled_repair)
            
            avg_accuracy = float(np.mean(pass_at_k_scores)) if pass_at_k_scores else 0.0
            avg_latency = float(np.mean(latencies)) if latencies else 0.0
            
            repair_curve.append({
                "repair_rounds": r,
                "accuracy": avg_accuracy,
                "avg_latency": avg_latency,
                "marginal_gain": 0.0,
            })
        
        # Compute marginal gains
        for i in range(1, len(repair_curve)):
            repair_curve[i]["marginal_gain"] = (
                repair_curve[i]["accuracy"] - repair_curve[i - 1]["accuracy"]
            )
        
        # ── Optimal operating point (per prompt methodology) ──
        # Best k = argmax(efficiency)
        if k_curve:
            best_k = max(k_curve, key=lambda x: x["efficiency"])
        else:
            best_k = {"k": 1, "accuracy": 0, "avg_latency": 0, "efficiency": 0}
        
        # Optimal repair = last round where marginal_gain > threshold
        optimal_repair = 0
        for r in repair_curve:
            if r["repair_rounds"] > 0 and r["marginal_gain"] > self.MARGINAL_GAIN_THRESHOLD:
                optimal_repair = r["repair_rounds"]
        
        self.results = {
            "agent": self.name,
            "accuracy_vs_k": k_curve,
            "accuracy_vs_repair_rounds": repair_curve,
            "optimal_operating_point": {
                "best_k": best_k,
                "optimal_repair_rounds": optimal_repair,
            },
        }
        self._attach_anomalies(self.results)
        
        return self.results
    
    def _analyze_sweep(self, sweep_results: Dict) -> Dict[str, Any]:
        """Analyze pre-computed sweep results."""
        self.results = {
            "agent": self.name,
            "sweep_results": sweep_results,
        }
        self._attach_anomalies(self.results)
        return self.results
