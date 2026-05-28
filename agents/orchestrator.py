"""
Agent 1: Orchestrator Agent

Coordinates all agents, runs the full evaluation pipeline,
aggregates outputs, and produces the final structured report.

Performs MANDATORY consistency checks per the prompt specification:
  1. repair_pass@k >= generation_pass@k
  2. oracle_pass@1 >= repair_pass@1
  3. sum(error_distribution) ≈ 100%
  4. total_executions consistency
  5. problems_solved <= total_problems
  6. cost_per_solved >= cost_per_problem
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm

from config import CONFIG
from datasets import load_humaneval, load_mbpp
from pipeline.generator import generate_candidates
from pipeline.executor import execute_code, execute_code_batch, ErrorType
from pipeline.repairer import repair_candidates
from pipeline.ranker import rank_candidates

from agents.base_agent import BaseAgent
from agents.prompts import ORCHESTRATOR_PROMPT
from agents.generation_eval import GenerationEvalAgent
from agents.execution_metrics import ExecutionMetricsAgent
from agents.repair_analysis import RepairAnalysisAgent
from agents.oracle_analysis import OracleAnalysisAgent
from agents.efficiency_cost import EfficiencyCostAgent
from agents.error_analysis import ErrorAnalysisAgent
from agents.dataset_analysis import DatasetAnalysisAgent
from agents.tradeoff_analysis import TradeoffAnalysisAgent


class OrchestratorAgent(BaseAgent):
    """Coordinates all agents and runs the full evaluation pipeline."""
    
    SYSTEM_PROMPT = ORCHESTRATOR_PROMPT
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__("Orchestrator Agent")
        self.config = config or CONFIG
        self.client = OpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
        )
        
        # Initialize all agents
        self.agents = {
            "generation_eval": GenerationEvalAgent(),
            "execution_metrics": ExecutionMetricsAgent(),
            "repair_analysis": RepairAnalysisAgent(),
            "oracle_analysis": OracleAnalysisAgent(),
            "efficiency_cost": EfficiencyCostAgent(),
            "error_analysis": ErrorAnalysisAgent(),
            "dataset_analysis": DatasetAnalysisAgent(),
            "tradeoff_analysis": TradeoffAnalysisAgent(),
        }
        
        # Raw pipeline results
        self.pipeline_results: List[Dict] = []
        self.all_errors: List[Dict] = []
    
    def load_datasets(self) -> List[Dict]:
        """Load all configured datasets."""
        all_problems = []
        for ds_name in self.config.datasets:
            if ds_name == "humaneval":
                all_problems.extend(load_humaneval())
            elif ds_name == "mbpp":
                all_problems.extend(load_mbpp())
            else:
                print(f"[Orchestrator] Unknown dataset: {ds_name}")
        return all_problems
    
    def run_pipeline_for_problem(self, problem: Dict) -> Dict:
        """
        Run the full pipeline for a single problem:
        1. Generate k candidates
        2. Execute all candidates
        3. Repair failed candidates
        4. Re-execute repaired candidates
        5. Rank and select
        """
        task_id = problem.get("task_id", "?")
        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")
        
        result = {
            "task_id": task_id,
            "dataset": problem.get("dataset", ""),
        }
        
        # ── Step 1: Generate candidates ──
        candidates, gen_stats = generate_candidates(
            problem, k=self.config.k, client=self.client
        )
        result["generation_time"] = gen_stats["generation_time"]
        result["input_tokens"] = gen_stats.get("input_tokens", 0)
        result["output_tokens"] = gen_stats.get("output_tokens", 0)
        
        if not candidates:
            result.update({
                "num_candidates": 0,
                "num_correct_before_repair": 0,
                "num_correct_after_repair": 0,
                "num_candidates_total": 0,
                "num_correct_total": 0,
                "any_candidate_passed": False,
                "solved": False,
                "execution_time": 0,
                "repair_time": 0,
                "total_executions": 0,
                "test_pass_rates": [],
                "execution_results": [],
                "num_failed_before_repair": 0,
                "num_fixed_by_repair": 0,
                "per_round_stats": [],
                "total_repair_rounds": 0,
            })
            return result
        
        # ── Step 2: Execute initial candidates ──
        exec_start = time.perf_counter()
        initial_results = execute_code_batch(
            candidates, test_code, entry_point, self.config.execution_timeout
        )
        exec_time_initial = time.perf_counter() - exec_start
        
        # Track test pass rates
        test_pass_rates = []
        for er in initial_results:
            rate = self.safe_div(er.tests_passed, er.tests_total)
            test_pass_rates.append(rate)
        
        num_correct_before = sum(1 for r in initial_results if r.passed)
        num_failed_before = len(initial_results) - num_correct_before
        
        # Collect execution results
        all_exec_results = [r.to_dict() for r in initial_results]
        
        # Track errors
        for i, r in enumerate(initial_results):
            if not r.passed:
                self.all_errors.append({
                    "task_id": task_id,
                    "dataset": problem.get("dataset", ""),
                    "error_type": r.error_type.value,
                    "error_message": r.error_message[:200],
                    "was_repaired": False,
                    "stage": "generation",
                    "candidate_index": i,
                })
        
        # ── Step 3: Repair loop ──
        all_repaired = []
        all_repair_results = []
        total_repair_time = 0.0
        per_round_stats = []
        repair_input_tokens = 0
        repair_output_tokens = 0
        
        # Slot semantics: each initial candidate index defines one slot.
        slot_success = [r.passed for r in initial_results]
        current_failures = [
            (i, candidates[i], initial_results[i])
            for i in range(len(candidates))
            if not initial_results[i].passed
        ]
        
        for round_idx in range(self.config.repair_rounds):
            if not current_failures:
                break
            
            repaired_codes, repair_slot_indices, repair_stats = repair_candidates(
                current_failures, problem,
                repair_rounds=1,
                fixes_per_failure=self.config.fixes_per_failure,
                client=self.client,
            )
            total_repair_time += repair_stats.get("repair_time", 0)
            repair_input_tokens += repair_stats.get("input_tokens", 0)
            repair_output_tokens += repair_stats.get("output_tokens", 0)
            
            if not repaired_codes:
                per_round_stats.append({
                    "round": round_idx + 1,
                    "candidates_fixed": 0,
                    "candidates_attempted": len(current_failures),
                })
                continue
            
            exec_start = time.perf_counter()
            repair_results = execute_code_batch(
                repaired_codes, test_code, entry_point, self.config.execution_timeout
            )
            exec_time_repair = time.perf_counter() - exec_start
            exec_time_initial += exec_time_repair
            
            attempted_slots = len({slot_idx for slot_idx, _, _ in current_failures})
            fixed_slots_this_round = set()
            for i, r in enumerate(repair_results):
                slot_idx = repair_slot_indices[i]
                if r.passed and not slot_success[slot_idx]:
                    fixed_slots_this_round.add(slot_idx)
                if r.passed:
                    slot_success[slot_idx] = True

            per_round_stats.append({
                "round": round_idx + 1,
                "candidates_fixed": len(fixed_slots_this_round),
                "candidates_attempted": attempted_slots,
            })
            
            for i, r in enumerate(repair_results):
                slot_idx = repair_slot_indices[i]
                if r.passed:
                    for err in self.all_errors:
                        if (err["task_id"] == task_id
                                and not err["was_repaired"]
                                and err["stage"] == "generation"
                                and err.get("candidate_index") == slot_idx):
                            err["was_repaired"] = True
                            break
                else:
                    self.all_errors.append({
                        "task_id": task_id,
                        "dataset": problem.get("dataset", ""),
                        "error_type": r.error_type.value,
                        "error_message": r.error_message[:200],
                        "was_repaired": False,
                        "stage": f"repair_round_{round_idx + 1}",
                        "candidate_index": slot_idx,
                    })
            
            all_repaired.extend(repaired_codes)
            all_repair_results.extend(repair_results)
            all_exec_results.extend([r.to_dict() for r in repair_results])
            
            current_failures = [
                (repair_slot_indices[i], repaired_codes[i], repair_results[i])
                for i in range(len(repaired_codes))
                if not repair_results[i].passed and not slot_success[repair_slot_indices[i]]
            ]
        
        # ── Step 4: Aggregate results ──
        num_correct_after = sum(1 for ok in slot_success if ok)
        num_fixed_by_repair = num_correct_after - num_correct_before
        
        all_candidates = candidates + all_repaired
        all_results_combined = initial_results + all_repair_results
        num_correct_total = num_correct_after
        any_passed = num_correct_after > 0
        
        # ── Step 5: Rank and select ──
        ranked = rank_candidates(all_candidates, all_results_combined)
        best_code, best_result, best_score = ranked[0] if ranked else ("", None, 0.0)
        
        result.update({
            "num_candidates": len(candidates),
            "num_correct_before_repair": num_correct_before,
            "num_correct_after_repair": num_correct_after,
            "num_failed_before_repair": num_failed_before,
            "num_fixed_by_repair": num_fixed_by_repair,
            "num_candidates_total": len(all_candidates),
            "num_correct_total": num_correct_total,
            "any_candidate_passed": any_passed,
            "solved": best_result.passed if best_result else False,
            "best_score": best_score,
            "execution_time": exec_time_initial,
            "repair_time": total_repair_time,
            "total_executions": len(all_exec_results),
            "test_pass_rates": test_pass_rates,
            "execution_results": all_exec_results,
            "per_round_stats": per_round_stats,
            "total_repair_rounds": self.config.repair_rounds,
            "repair_input_tokens": repair_input_tokens,
            "repair_output_tokens": repair_output_tokens,
        })
        
        return result
    
    def _run_consistency_checks(
        self,
        gen_results: Dict,
        repair_results: Dict,
        oracle_results: Dict,
        exec_results: Dict,
        error_results: Dict,
        cost_results: Dict,
    ) -> List[str]:
        """
        Run the 6 mandatory consistency checks from the prompt.
        Returns list of inconsistency messages.
        """
        inconsistencies = []
        
        # 1. repair_pass@k >= generation_pass@k
        gen_pk = gen_results.get("pass_at_k_before_repair", {})
        rep_pk = repair_results.get("pass_at_k_after_repair", {})
        for k_str in ["pass@1", "pass@5", "pass@10"]:
            g = gen_pk.get(k_str)
            r = rep_pk.get(k_str)
            if g is None or r is None:
                continue
            if r < g - 1e-9:
                inconsistencies.append(
                    f"[CHECK 1] repair {k_str}={r:.4f} < generation {k_str}={g:.4f}"
                )
        
        # 2. oracle_pass@1 >= repair_pass@1
        oracle_p1 = oracle_results.get("oracle_pass_at_1", 0)
        repair_p1 = rep_pk.get("pass@1", 0)
        if oracle_p1 < repair_p1 - 1e-9:
            inconsistencies.append(
                f"[CHECK 2] oracle_pass@1={oracle_p1:.4f} < repair_pass@1={repair_p1:.4f}"
            )
        
        # 3. error distribution sums to ~100%
        dist = error_results.get("distribution", {})
        if dist:
            total_pct = sum(v.get("percentage", 0) for v in dist.values())
            if abs(total_pct - 100.0) > 0.5:
                inconsistencies.append(
                    f"[CHECK 3] Error distribution sums to {total_pct:.2f}%, expected ~100%"
                )
        
        # 4. total executions consistency
        # (Execution Metrics Agent already checks this internally)
        
        # 5. problems_solved <= total_problems
        solved = gen_results.get("problems_solved", 0)
        total = gen_results.get("total_problems", 0)
        if solved > total:
            inconsistencies.append(
                f"[CHECK 5] problems_solved={solved} > total_problems={total}"
            )
        
        # 6. cost_per_solved >= cost_per_problem
        cps = cost_results.get("cost_per_solved_problem", 0)
        cpp = cost_results.get("cost_per_problem", 0)
        if cps < cpp - 1e-9 and cps > 0:
            inconsistencies.append(
                f"[CHECK 6] cost_per_solved={cps:.4f} < cost_per_problem={cpp:.4f}"
            )
        
        return inconsistencies
    
    def analyze(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the full multi-agent evaluation with consistency checks.
        """
        self.anomalies = []  # reset
        
        if data and "problems" in data:
            self.pipeline_results = data["problems"]
        else:
            problems = self.load_datasets()
            print(f"\n{'='*60}")
            print(f"[Orchestrator] Starting evaluation of {len(problems)} problems")
            print(f"  k={self.config.k}, temps={self.config.temperatures}")
            print(f"  repair_rounds={self.config.repair_rounds}, "
                  f"fixes_per_failure={self.config.fixes_per_failure}")
            print(f"{'='*60}\n")
            
            self.pipeline_results = []
            for problem in tqdm(problems, desc="Evaluating problems"):
                result = self.run_pipeline_for_problem(problem)
                self.pipeline_results.append(result)
        
        # ── Run all agents in sequence ──
        print("\n[Orchestrator] Running analysis agents...")
        
        problem_data = {"problems": self.pipeline_results}
        
        # 1. Generation Evaluation Agent
        print("  → Generation Evaluation Agent")
        gen_results = self.agents["generation_eval"].analyze(problem_data)
        
        # 2. Execution Metrics Agent
        print("  → Execution Metrics Agent")
        exec_results = self.agents["execution_metrics"].analyze(problem_data)
        
        # 3. Repair Analysis Agent
        print("  → Repair Analysis Agent")
        repair_results = self.agents["repair_analysis"].analyze(problem_data)
        
        # 4. Oracle Analysis Agent
        print("  → Oracle Analysis Agent")
        oracle_data = dict(problem_data)
        oracle_data["generation_pass_at_1"] = gen_results.get(
            "pass_at_k_before_repair", {}
        ).get("pass@1", 0.0)
        oracle_data["after_repair_pass_at_1"] = repair_results.get(
            "pass_at_k_after_repair", {}
        ).get("pass@1", 0.0)
        oracle_results = self.agents["oracle_analysis"].analyze(oracle_data)
        
        # 5. Error Analysis Agent
        print("  → Error Analysis Agent")
        error_results = self.agents["error_analysis"].analyze({
            "errors": self.all_errors
        })
        
        # 6. Efficiency & Cost Agent
        print("  → Efficiency & Cost Agent")
        cost_results = self.agents["efficiency_cost"].analyze(problem_data)
        
        # 7. Dataset Analysis Agent
        print("  → Dataset Analysis Agent")
        dataset_results = self.agents["dataset_analysis"].analyze(problem_data)
        
        # 8. Tradeoff Analysis Agent
        print("  → Tradeoff Analysis Agent")
        tradeoff_results = self.agents["tradeoff_analysis"].analyze(problem_data)
        
        # ── MANDATORY consistency checks ──
        print("\n[Orchestrator] Running consistency checks...")
        inconsistencies = self._run_consistency_checks(
            gen_results, repair_results, oracle_results,
            exec_results, error_results, cost_results,
        )
        
        # Collect anomalies from all agents
        all_agent_anomalies = {}
        for name, agent in self.agents.items():
            agent_anomalies = agent.get_results().get("anomalies", [])
            if agent_anomalies:
                all_agent_anomalies[name] = agent_anomalies
        
        if inconsistencies:
            for ic in inconsistencies:
                self.flag_inconsistency(ic)
                print(f"  ⚠ {ic}")
        else:
            print("  ✓ All consistency checks passed")
        
        if all_agent_anomalies:
            print(f"  ⚠ {sum(len(v) for v in all_agent_anomalies.values())} agent anomalies detected")
        
        # ── Compile final report ──
        print("\n[Orchestrator] Compiling final report...")
        
        self.results = {
            "orchestrator": self.name,
            "config": {
                "model": self.config.model_name,
                "k": self.config.k,
                "temperatures": self.config.temperatures,
                "repair_rounds": self.config.repair_rounds,
                "fixes_per_failure": self.config.fixes_per_failure,
                "execution_timeout": self.config.execution_timeout,
            },
            "agent_results": {
                "generation_eval": gen_results,
                "execution_metrics": exec_results,
                "repair_analysis": repair_results,
                "oracle_analysis": oracle_results,
                "error_analysis": error_results,
                "efficiency_cost": cost_results,
                "dataset_analysis": dataset_results,
                "tradeoff_analysis": tradeoff_results,
            },
            "consistency_checks": {
                "passed": len(inconsistencies) == 0,
                "inconsistencies": inconsistencies,
            },
            "agent_anomalies": all_agent_anomalies,
        }
        self._attach_anomalies(self.results)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        """Save raw results to JSON."""
        if output_dir is None:
            output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "full_results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        for name, agent in self.agents.items():
            with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
                json.dump(agent.get_results(), f, indent=2, default=str)
        
        with open(os.path.join(output_dir, "pipeline_data.json"), "w") as f:
            json.dump(self.pipeline_results, f, indent=2, default=str)
        
        print(f"[Orchestrator] Results saved to {output_dir}/")
