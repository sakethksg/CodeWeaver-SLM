"""
Report Generator

Produces the final structured markdown report from all agent outputs.
Follows the strict output format specified in the evaluation protocol.
"""

import os
import json
from typing import Any, Dict
from datetime import datetime


class ReportGenerator:
    """Generates the final evaluation report."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.config = results.get("config", {})
        self.agents = results.get("agent_results", {})
    
    def generate(self) -> str:
        """Generate the full markdown report."""
        sections = [
            self._header(),
            self._consistency_status(),
            self._summary_table(),
            self._detailed_metrics(),
            self._tradeoff_analysis(),
            self._error_analysis(),
            self._insights(),
            self._conclusions(),
        ]
        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _fmt(value: Any, digits: int = 3) -> str:
        """Format numeric values with fixed precision or return N/A."""
        if value is None:
            return "N/A"
        return f"{value:.{digits}f}"
    
    def _header(self) -> str:
        """Report header with configuration."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# 📊 Multi-Agent Evaluation Report

**Generated**: {ts}  
**Model**: `{self.config.get('model', 'N/A')}`  
**Experiment Protocol**:
| Parameter | Value |
|-----------|-------|
| k (samples) | {self.config.get('k', 'N/A')} |
| Temperature | {self.config.get('temperatures', 'N/A')} |
| Repair Rounds | {self.config.get('repair_rounds', 'N/A')} |
| Fixes per Failure | {self.config.get('fixes_per_failure', 'N/A')} |
| Execution Timeout | {self.config.get('execution_timeout', 'N/A')}s |

---"""
    
    def _consistency_status(self) -> str:
        """Consistency check results from the Orchestrator."""
        checks = self.results.get("consistency_checks", {})
        agent_anomalies = self.results.get("agent_anomalies", {})
        
        if not checks and not agent_anomalies:
            return ""
        
        passed = checks.get("passed", True)
        inconsistencies = checks.get("inconsistencies", [])
        
        status_icon = "✅" if passed else "⚠️"
        status_text = "All checks passed" if passed else f"{len(inconsistencies)} issue(s) found"
        
        lines = [f"> **Consistency Checks**: {status_icon} {status_text}"]
        
        if inconsistencies:
            lines.append("")
            lines.append("> **Issues:**")
            for ic in inconsistencies:
                lines.append(f"> - {ic}")
        
        if agent_anomalies:
            total_anomalies = sum(len(v) for v in agent_anomalies.values())
            lines.append("")
            lines.append(f"> **Agent Anomalies**: {total_anomalies} detected")
            for agent_name, anomalies in agent_anomalies.items():
                for a in anomalies:
                    lines.append(f"> - `{agent_name}`: {a}")
        
        return "\n".join(lines)
    
    def _summary_table(self) -> str:
        """Section 1: Summary Table."""
        ds_results = self.agents.get("dataset_analysis", {}).get("per_dataset", {})
        gen_results = self.agents.get("generation_eval", {})
        oracle_results = self.agents.get("oracle_analysis", {})
        
        header = (
            "## 1. Summary Table\n\n"
            "| Dataset | pass@1 | pass@5 | pass@10 | "
            "pass@10 (after repair) | Oracle solve rate | Avg Score |"
        )
        separator = "|---------|--------|--------|---------|" \
                     "-----------------------|---------------|-----------|"
        
        rows = []
        for ds_name, ds_data in ds_results.items():
            before = ds_data.get("pass_at_k_before_repair", {})
            after = ds_data.get("pass_at_k_after_repair", {})
            oracle_upper = ds_data.get("oracle_upper_bound", 0)
            
            p1 = before.get("pass@1")
            p5 = before.get("pass@5")
            p10 = before.get("pass@10")
            p10_repair = after.get("pass@10")
            oracle_k = oracle_upper
            avg = ds_data.get("avg_score", 0)
            
            rows.append(
                f"| **{ds_name.upper()}** | {self._fmt(p1)} | {self._fmt(p5)} | {self._fmt(p10)} | "
                f"{self._fmt(p10_repair)} | {self._fmt(oracle_k)} | {self._fmt(avg)} |"
            )
        
        # Aggregate row
        gen_pass = gen_results.get("pass_at_k_before_repair", {})
        repair_pass = self.agents.get("repair_analysis", {}).get(
            "pass_at_k_after_repair", {}
        )
        oracle_upper = self.agents.get("oracle_analysis", {}).get("oracle_pass_at_1", 0)
        
        agg_p1 = gen_pass.get("pass@1")
        agg_p5 = gen_pass.get("pass@5")
        agg_p10 = gen_pass.get("pass@10")
        agg_p10r = repair_pass.get("pass@10")
        agg_oracle = oracle_upper
        agg_avg = gen_results.get("avg_test_pass_rate", 0)
        
        rows.append(
            f"| **OVERALL** | {self._fmt(agg_p1)} | {self._fmt(agg_p5)} | {self._fmt(agg_p10)} | "
            f"{self._fmt(agg_p10r)} | {self._fmt(agg_oracle)} | {self._fmt(agg_avg)} |"
        )
        
        return f"{header}\n{separator}\n" + "\n".join(rows)
    
    def _detailed_metrics(self) -> str:
        """Section 2: Detailed Metrics."""
        gen = self.agents.get("generation_eval", {})
        exec_m = self.agents.get("execution_metrics", {})
        repair = self.agents.get("repair_analysis", {})
        cost = self.agents.get("efficiency_cost", {})
        
        gen_section = f"""### 2.1 Generation Performance

| Metric | Value |
|--------|-------|
| pass@1 (before repair) | {self._fmt(gen.get('pass_at_k_before_repair', {}).get('pass@1'), 4)} |
| pass@5 (before repair) | {self._fmt(gen.get('pass_at_k_before_repair', {}).get('pass@5'), 4)} |
| pass@10 (before repair) | {self._fmt(gen.get('pass_at_k_before_repair', {}).get('pass@10'), 4)} |
| % Problems Solved | {gen.get('pct_problems_solved', 0):.1f}% |
| Problems Solved | {gen.get('problems_solved', 0)}/{gen.get('total_problems', 0)} |
| Avg Test Pass Rate | {gen.get('avg_test_pass_rate', 0):.4f} |"""
        
        exec_section = f"""### 2.2 Execution Statistics

| Metric | Value |
|--------|-------|
| Total Executions | {exec_m.get('total_executions', 0):,} |
| Successes | {exec_m.get('total_successes', 0):,} |
| Failures | {exec_m.get('total_failures', 0):,} |
| Success Rate | {exec_m.get('success_rate', 0):.1f}% |
| Avg Executions/Problem | {exec_m.get('avg_executions_per_problem', 0):.1f} |
| Avg Test Pass Rate | {exec_m.get('avg_test_pass_rate', 0):.4f} |"""
        
        repair_section = f"""### 2.3 Repair Effectiveness

| Metric | Value |
|--------|-------|
| pass@1 (after repair) | {self._fmt(repair.get('pass_at_k_after_repair', {}).get('pass@1'), 4)} |
| pass@5 (after repair) | {self._fmt(repair.get('pass_at_k_after_repair', {}).get('pass@5'), 4)} |
| pass@10 (after repair) | {self._fmt(repair.get('pass_at_k_after_repair', {}).get('pass@10'), 4)} |
| Repair Success Rate | {repair.get('repair_success_rate', 0):.1f}% |
| Candidates Fixed | {repair.get('candidates_fixed_ratio', 'N/A')} |
| Δ pass@1 | {self._fmt(repair.get('delta_improvement', {}).get('delta_pass@1'), 4)} |
| Δ pass@10 | {self._fmt(repair.get('delta_improvement', {}).get('delta_pass@10'), 4)} |"""
        
        # Per-round table
        rounds = repair.get("per_round_analysis", [])
        round_rows = []
        for r in rounds:
            round_rows.append(
                f"| Round {r['round']} | {r['candidates_fixed']} | "
                f"{r['candidates_attempted']} | {r['fix_rate']:.1f}% |"
            )
        round_table = (
            "| Round | Fixed | Attempted | Fix Rate |\n"
            "|-------|-------|-----------|----------|\n"
            + "\n".join(round_rows)
        ) if round_rows else "_No repair rounds executed._"
        
        repair_section += f"\n\n**Per-Round Breakdown:**\n\n{round_table}"
        
        cost_section = f"""### 2.4 Efficiency & Cost

| Metric | Value |
|--------|-------|
| Avg Executions/Solution | {cost.get('avg_executions_per_successful_solution', 0):.1f} |
| Avg Latency/Problem | {cost.get('avg_latency_per_problem', 0):.2f}s |
| Latency: Generation | {cost.get('latency_breakdown', {}).get('generation', 0):.2f}s |
| Latency: Execution | {cost.get('latency_breakdown', {}).get('execution', 0):.2f}s |
| Latency: Repair | {cost.get('latency_breakdown', {}).get('repair', 0):.2f}s |
| Cost/Problem | ${cost.get('cost_per_problem', 0):.4f} |
| Cost/Solved Problem | ${cost.get('cost_per_solved_problem', 0):.4f} |
| Marginal Cost of Repair | ${cost.get('marginal_cost_of_repair', 0):.4f} |
| Total Cost | ${cost.get('total_cost', 0):.2f} |"""
        
        return f"## 2. Detailed Metrics\n\n{gen_section}\n\n{exec_section}\n\n{repair_section}\n\n{cost_section}"
    
    def _tradeoff_analysis(self) -> str:
        """Section 3: Tradeoff Analysis."""
        tradeoff = self.agents.get("tradeoff_analysis", {})
        
        # Accuracy vs k table
        k_curve = tradeoff.get("accuracy_vs_k", [])
        k_rows = []
        for point in k_curve:
            accuracy = point.get("accuracy")
            efficiency = point.get("efficiency")
            if accuracy is None:
                continue
            k_rows.append(
                f"| {point['k']} | {self._fmt(accuracy, 4)} | "
                f"{point['avg_latency']:.2f}s | {self._fmt(efficiency, 4)} |"
            )
        k_table = (
            "| k | Accuracy (pass@k) | Avg Latency | Efficiency |\n"
            "|---|-------------------|-------------|------------|\n"
            + "\n".join(k_rows)
        ) if k_rows else "_No data._"
        
        # Repair rounds table
        r_curve = tradeoff.get("accuracy_vs_repair_rounds", [])
        r_rows = []
        for point in r_curve:
            accuracy = point.get("accuracy")
            if accuracy is None:
                continue
            r_rows.append(
                f"| {point['repair_rounds']} | {self._fmt(accuracy, 4)} | "
                f"{point['avg_latency']:.2f}s | {point['marginal_gain']:+.4f} |"
            )
        r_table = (
            "| Repair Rounds | Accuracy | Avg Latency | Marginal Gain |\n"
            "|---------------|----------|-------------|---------------|\n"
            + "\n".join(r_rows)
        ) if r_rows else "_No data._"
        
        optimal = tradeoff.get("optimal_operating_point", {})
        best_k = optimal.get("best_k", {})
        
        return f"""## 3. Tradeoff Analysis

### 3.1 Latency vs Accuracy (Varying k)

{k_table}

### 3.2 Cost vs Performance (Varying Repair Rounds)

{r_table}

### 3.3 Optimal Operating Point

| Parameter | Value |
|-----------|-------|
| Best k | {best_k.get('k', 'N/A')} |
| Best k Accuracy | {self._fmt(best_k.get('accuracy'), 4)} |
| Best k Latency | {best_k.get('avg_latency', 0):.2f}s |
| Optimal Repair Rounds | {optimal.get('optimal_repair_rounds', 'N/A')} |"""
    
    def _error_analysis(self) -> str:
        """Section 4: Error Analysis."""
        errors = self.agents.get("error_analysis", {})
        
        # ── Aggregated distribution table ──
        agg_dist = errors.get("aggregated_distribution", {})
        agg_rows = []
        for group_name in ["SYNTAX", "RUNTIME", "LOGICAL", "TIMEOUT"]:
            info = agg_dist.get(group_name, {"count": 0, "percentage": 0.0})
            agg_rows.append(
                f"| **{group_name}** | {info['count']} | {info['percentage']:.1f}% |"
            )
        agg_table = (
            "| Group | Count | Percentage |\n"
            "|-------|-------|------------|\n"
            + "\n".join(agg_rows)
        ) if agg_rows else "_No errors._"
        
        # ── Granular distribution table ──
        dist = errors.get("distribution", {})
        dist_rows = []
        for etype, info in dist.items():
            dist_rows.append(
                f"| {etype} | {info['count']} | {info['percentage']:.1f}% |"
            )
        dist_table = (
            "| Error Type | Count | Percentage |\n"
            "|------------|-------|------------|\n"
            + "\n".join(dist_rows)
        ) if dist_rows else "_No errors recorded._"
        
        # ── Repairability table ──
        repair = errors.get("repairability", {})
        repair_rows = []
        for etype, info in repair.items():
            repair_rows.append(
                f"| {etype} | {info['total']} | {info['repaired']} | "
                f"{info['repair_rate']:.1f}% |"
            )
        repair_table = (
            "| Error Type | Total | Repaired | Repair Rate |\n"
            "|------------|-------|----------|-------------|\n"
            + "\n".join(repair_rows)
        ) if repair_rows else "_No repair data._"
        
        # Most fixable
        fixable = errors.get("most_fixable_types", [])
        fixable_str = ""
        if fixable:
            fixable_str = "\n**Most Fixable Error Types (ranked):**\n"
            for i, f in enumerate(fixable, 1):
                fixable_str += f"{i}. `{f['type']}` — {f['repair_rate']:.1f}% repair rate\n"
        
        return f"""## 4. Error Analysis

### 4.1 Error Distribution (Aggregated)

{agg_table}

### 4.2 Error Distribution (Granular)

{dist_table}

### 4.3 Repairability Insights

{repair_table}
{fixable_str}"""
    
    def _insights(self) -> str:
        """Section 5: Insights."""
        gen = self.agents.get("generation_eval", {})
        repair = self.agents.get("repair_analysis", {})
        oracle = self.agents.get("oracle_analysis", {})
        cost = self.agents.get("efficiency_cost", {})
        tradeoff = self.agents.get("tradeoff_analysis", {})
        
        # ── Strengths ──
        strengths = []
        p1_before = gen.get("pass_at_k_before_repair", {}).get("pass@1", 0)
        p1_after = repair.get("pass_at_k_after_repair", {}).get("pass@1", 0)
        
        if p1_before > 0.3:
            strengths.append(f"Strong base generation: pass@1 = {p1_before:.3f}")
        if p1_after > p1_before:
            delta = p1_after - p1_before
            strengths.append(f"Repair loop adds +{delta:.3f} to pass@1")
        
        oracle_rate = oracle.get("oracle_solve_rate", 0)
        if oracle_rate > 50:
            strengths.append(f"High oracle ceiling: {oracle_rate:.1f}% solvable with best selection")
        
        repair_success = repair.get("repair_success_rate", 0)
        if repair_success > 20:
            strengths.append(f"Repair fixes {repair_success:.1f}% of failed candidates")
        
        if not strengths:
            strengths.append("Further analysis needed to identify strengths")
        
        # ── Weaknesses ──
        weaknesses = []
        if p1_before < 0.3:
            weaknesses.append(f"Low base generation quality: pass@1 = {p1_before:.3f}")
        
        gap = oracle.get("gap_to_oracle", {})
        gap_from_repair = gap.get("from_after_repair_pass1", 0)
        if gap_from_repair > 0.15:
            weaknesses.append(
                f"Large gap to oracle: {gap_from_repair:.3f} "
                f"(room for better selection/ranking)"
            )
        
        if repair_success < 15:
            weaknesses.append(f"Low repair success rate: {repair_success:.1f}%")
        
        # Diminishing returns
        dim_returns = repair.get("diminishing_returns", [])
        for dr in dim_returns:
            if dr["rate_change"] < -10:
                weaknesses.append(
                    f"Significant diminishing returns from round "
                    f"{dr['from_round']} to {dr['to_round']}"
                )
        
        if not weaknesses:
            weaknesses.append("No significant weaknesses identified")
        
        # ── Bottlenecks ──
        bottlenecks = []
        latency = cost.get("latency_breakdown", {})
        gen_lat = latency.get("generation", 0)
        exec_lat = latency.get("execution", 0)
        repair_lat = latency.get("repair", 0)
        total_lat = gen_lat + exec_lat + repair_lat
        
        if total_lat > 0:
            max_component = max(
                [("Generation", gen_lat), ("Execution", exec_lat), ("Repair", repair_lat)],
                key=lambda x: x[1]
            )
            pct = max_component[1] / total_lat * 100
            bottlenecks.append(
                f"{max_component[0]} is the primary bottleneck "
                f"({pct:.0f}% of total latency)"
            )
        
        if not bottlenecks:
            bottlenecks.append("Latency is well-distributed across pipeline stages")
        
        # ── Gap to Oracle ──
        gap_insights = []
        oracle_p1 = oracle.get("oracle_pass_at_1", 0)
        gap_insights.append(f"Oracle pass@1: {oracle_p1:.3f}")
        gap_insights.append(f"Best achieved pass@1: {p1_after:.3f}")
        gap_insights.append(f"Gap: {oracle_p1 - p1_after:.3f}")
        
        solved_only_repair = cost.get("solved_only_by_repair", 0)
        if solved_only_repair > 0:
            gap_insights.append(
                f"{solved_only_repair} problems were solved ONLY through repair"
            )
        
        return f"""## 5. Insights

### 5.1 Strengths
{chr(10).join(f'- {s}' for s in strengths)}

### 5.2 Weaknesses
{chr(10).join(f'- {w}' for w in weaknesses)}

### 5.3 Bottlenecks
{chr(10).join(f'- {b}' for b in bottlenecks)}

### 5.4 Gap to Oracle
{chr(10).join(f'- {g}' for g in gap_insights)}"""
    
    def _conclusions(self) -> str:
        """Section 6: Conclusions."""
        gen = self.agents.get("generation_eval", {})
        repair = self.agents.get("repair_analysis", {})
        oracle = self.agents.get("oracle_analysis", {})
        cost = self.agents.get("efficiency_cost", {})
        
        p1_before = gen.get("pass_at_k_before_repair", {}).get("pass@1", 0)
        p1_after = repair.get("pass_at_k_after_repair", {}).get("pass@1", 0)
        p10_after = repair.get("pass_at_k_after_repair", {}).get("pass@10", 0)
        oracle_rate = oracle.get("oracle_solve_rate", 0)
        total_cost = cost.get("total_cost", 0)
        avg_latency = cost.get("avg_latency_per_problem", 0)
        repair_cost = cost.get("marginal_cost_of_repair", 0)
        
        # Effectiveness assessment
        if p1_after > 0.5:
            effectiveness = "**High effectiveness**"
            eff_detail = f"The system achieves pass@1 of {p1_after:.3f} after repair."
        elif p1_after > 0.3:
            effectiveness = "**Moderate effectiveness**"
            eff_detail = f"The system achieves pass@1 of {p1_after:.3f} after repair, with room for improvement."
        else:
            effectiveness = "**Limited effectiveness**"
            eff_detail = f"The system achieves pass@1 of {p1_after:.3f} after repair, indicating significant room for improvement."
        
        # Scalability assessment
        if avg_latency < 10:
            scalability = "Good scalability characteristics with low per-problem latency."
        elif avg_latency < 30:
            scalability = "Moderate scalability — acceptable for batch processing."
        else:
            scalability = "Scalability is a concern — high per-problem latency."
        
        # Compute-efficiency
        delta_p1 = p1_after - p1_before
        if delta_p1 > 0 and repair_cost > 0:
            improvement_per_dollar = delta_p1 / repair_cost
            cost_eff = (
                f"Repair provides +{delta_p1:.3f} pass@1 improvement at "
                f"${repair_cost:.2f} marginal cost "
                f"(${improvement_per_dollar:.2f} per unit improvement)."
            )
        else:
            cost_eff = "Cost-efficiency analysis requires non-zero repair cost data."
        
        # Deployment suitability
        if p1_after > 0.5 and avg_latency < 15:
            deploy = "✅ **Suitable** for production deployment with acceptable accuracy and latency."
        elif p1_after > 0.3:
            deploy = "⚠️ **Conditionally suitable** — may work for non-critical applications or with human review."
        else:
            deploy = "❌ **Not recommended** for production without significant improvements."
        
        return f"""## 6. Conclusions

### 6.1 Effectiveness
{effectiveness}: {eff_detail}
The repair loop improves pass@1 by {delta_p1:+.3f} ({p1_before:.3f} → {p1_after:.3f}).
Oracle ceiling is {oracle_rate:.1f}%, indicating the potential with perfect selection.

### 6.2 Scalability
{scalability}
Average latency per problem: {avg_latency:.2f}s.
Total evaluation cost: ${total_cost:.2f}.

### 6.3 Compute-Efficiency Tradeoffs
{cost_eff}

### 6.4 Deployment Suitability
{deploy}

---

*Report generated by the Multi-Agent Evaluation System*
*All metrics are execution-based (unit test correctness). No assumptions or heuristics.*"""
    
    def save(self, filepath: str):
        """Save the report to a file."""
        report = self.generate()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Report] Saved to {filepath}")
        return report
