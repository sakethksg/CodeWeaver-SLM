"""
Agent 6: Efficiency & Cost Agent

Analyzes system efficiency: executions per solution, latency breakdown,
cost per problem, marginal cost of repair.
Validates cost_per_solved >= cost_per_problem.

Output schema matches EFFICIENCY_COST_PROMPT exactly.
"""

from typing import Any, Dict, List
import numpy as np

from agents.base_agent import BaseAgent
from agents.prompts import EFFICIENCY_COST_PROMPT
from config import CONFIG


class EfficiencyCostAgent(BaseAgent):
    """Analyzes efficiency and cost metrics with validation."""
    
    SYSTEM_PROMPT = EFFICIENCY_COST_PROMPT
    
    def __init__(self):
        super().__init__("Efficiency & Cost Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze efficiency and cost with anomaly detection.
        
        Input: {"problems": [{task_id, dataset, solved, generation_time, execution_time,
                repair_time, total_executions, input_tokens, output_tokens, ...}]}
        """
        self.anomalies = []  # reset
        problems = data.get("problems", [])
        
        if not problems:
            self.results = {"agent": self.name, "error": "No data", "anomalies": []}
            return self.results
        
        solved_problems = [p for p in problems if p.get("solved", False)]
        
        # ── Latency ──
        gen_times = [p.get("generation_time", 0) for p in problems]
        exec_times = [p.get("execution_time", 0) for p in problems]
        repair_times = [p.get("repair_time", 0) for p in problems]
        total_times = [g + e + r for g, e, r in zip(gen_times, exec_times, repair_times)]
        
        avg_latency = float(np.mean(total_times)) if total_times else 0.0
        
        # ── Executions per solution (ONLY solved problems) ──
        solved_exec_counts = [
            p.get("total_executions", 0) for p in solved_problems
        ]
        avg_exec_per_solution = (
            float(np.mean(solved_exec_counts)) if solved_exec_counts else 0.0
        )
        
        # ── Cost model ──
        total_input_tokens = sum(p.get("input_tokens", 0) for p in problems)
        total_output_tokens = sum(p.get("output_tokens", 0) for p in problems)
        total_compute_time = sum(total_times)
        
        # Token-based cost (0 for self-hosted, per prompt constraint)
        token_cost = (
            total_input_tokens / 1000 * CONFIG.cost_per_1k_input_tokens
            + total_output_tokens / 1000 * CONFIG.cost_per_1k_output_tokens
        )
        # GPU compute cost
        gpu_cost = (total_compute_time / 3600) * CONFIG.gpu_cost_per_hour
        total_cost = token_cost + gpu_cost
        
        cost_per_problem = self.safe_div(total_cost, len(problems))
        cost_per_solved = self.safe_div(total_cost, len(solved_problems))
        
        # ── Validate: cost_per_solved >= cost_per_problem ──
        err = self.validate_upper_bound(
            cost_per_solved, cost_per_problem,
            "cost_per_solved", "cost_per_problem"
        )
        if err:
            self.flag_anomaly(err)
        
        # ── Marginal cost of repair ──
        total_repair_time = sum(repair_times)
        repair_gpu_cost = (total_repair_time / 3600) * CONFIG.gpu_cost_per_hour
        repair_token_cost = sum(
            p.get("repair_input_tokens", 0) for p in problems
        ) / 1000 * CONFIG.cost_per_1k_input_tokens + sum(
            p.get("repair_output_tokens", 0) for p in problems
        ) / 1000 * CONFIG.cost_per_1k_output_tokens
        marginal_repair_cost = repair_gpu_cost + repair_token_cost
        
        # ── Problems solved ONLY by repair ──
        solved_only_by_repair = sum(
            1 for p in problems
            if p.get("solved", False)
            and p.get("num_correct_before_repair", 0) == 0
        )
        
        self.results = {
            "agent": self.name,
            "avg_executions_per_successful_solution": avg_exec_per_solution,
            "avg_latency_per_problem": avg_latency,
            "latency_breakdown": {
                "generation": float(np.mean(gen_times)) if gen_times else 0.0,
                "execution": float(np.mean(exec_times)) if exec_times else 0.0,
                "repair": float(np.mean(repair_times)) if repair_times else 0.0,
            },
            "total_compute_time": total_compute_time,
            "cost_per_problem": cost_per_problem,
            "cost_per_solved_problem": cost_per_solved,
            "total_cost": total_cost,
            "marginal_cost_of_repair": marginal_repair_cost,
            "solved_only_by_repair": solved_only_by_repair,
            "token_usage": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            },
        }
        self._attach_anomalies(self.results)
        
        return self.results
