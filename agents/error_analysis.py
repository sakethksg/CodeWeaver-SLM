"""
Agent 7: Error Analysis Agent

Categorizes failures using the granular error taxonomy:
  - syntax_error, runtime_type, runtime_value, runtime_resource,
    logical_error, timeout

Reports both granular and aggregated (SYNTAX/RUNTIME/LOGICAL/TIMEOUT) distributions.
Validates that percentage distributions sum to ~100%.
"""

from typing import Any, Dict, List
import numpy as np
from collections import Counter

from agents.base_agent import BaseAgent
from agents.prompts import ERROR_ANALYSIS_PROMPT
from pipeline.executor import AGGREGATED_GROUPS


class ErrorAnalysisAgent(BaseAgent):
    """Categorizes and analyzes errors with granular + aggregated taxonomy."""
    
    SYSTEM_PROMPT = ERROR_ANALYSIS_PROMPT
    
    def __init__(self):
        super().__init__("Error Analysis Agent")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze errors per the prompt's output schema.
        
        Input: {"errors": [{"task_id", "dataset", "error_type", "was_repaired", "stage"}]}
        Output: matches ERROR_ANALYSIS_PROMPT output schema exactly.
        """
        self.anomalies = []  # reset
        errors = data.get("errors", [])
        
        if not errors:
            self.results = {
                "agent": self.name,
                "total_errors": 0,
                "distribution": {},
                "aggregated_distribution": {
                    "SYNTAX": {"count": 0, "percentage": 0.0},
                    "RUNTIME": {"count": 0, "percentage": 0.0},
                    "LOGICAL": {"count": 0, "percentage": 0.0},
                    "TIMEOUT": {"count": 0, "percentage": 0.0},
                },
                "repairability": {},
                "most_fixable_types": [],
                "per_dataset": {},
                "anomalies": [],
            }
            return self.results
        
        total_errors = len(errors)
        
        # ── Granular error distribution ──
        error_types = [e["error_type"] for e in errors]
        type_counts = Counter(error_types)
        
        distribution = {}
        for etype, count in type_counts.items():
            distribution[etype] = {
                "count": count,
                "percentage": self.safe_div(count, total_errors) * 100,
            }
        
        # ── Aggregated distribution ──
        aggregated_distribution = {}
        for group_name, member_types in AGGREGATED_GROUPS.items():
            group_count = sum(
                type_counts.get(mt, 0) for mt in member_types
            )
            aggregated_distribution[group_name] = {
                "count": group_count,
                "percentage": self.safe_div(group_count, total_errors) * 100,
            }
        
        # ── Validate percentages sum to ~100% ──
        pct_err = self.validate_percentage_sum(distribution, tolerance=0.5)
        if pct_err:
            self.flag_inconsistency(pct_err)
        
        agg_pct_err = self.validate_percentage_sum(aggregated_distribution, tolerance=0.5)
        if agg_pct_err:
            self.flag_inconsistency(f"Aggregated: {agg_pct_err}")
        
        # ── Repairability by type ──
        repairability = {}
        for etype in type_counts:
            type_errors = [e for e in errors if e["error_type"] == etype]
            repaired = sum(1 for e in type_errors if e.get("was_repaired", False))
            repairability[etype] = {
                "total": len(type_errors),
                "repaired": repaired,
                "repair_rate": self.safe_div(repaired, len(type_errors)) * 100,
            }
        
        # ── Sort by repairability (most fixable first) ──
        most_fixable = sorted(
            repairability.items(),
            key=lambda x: x[1]["repair_rate"],
            reverse=True,
        )
        
        # ── Per-dataset breakdown ──
        datasets = set(e.get("dataset", "") for e in errors)
        per_dataset = {}
        for ds in datasets:
            ds_errors = [e for e in errors if e.get("dataset", "") == ds]
            ds_counts = Counter(e["error_type"] for e in ds_errors)
            per_dataset[ds] = {
                etype: {
                    "count": count,
                    "percentage": self.safe_div(count, len(ds_errors)) * 100,
                }
                for etype, count in ds_counts.items()
            }
        
        self.results = {
            "agent": self.name,
            "total_errors": total_errors,
            "distribution": distribution,
            "aggregated_distribution": aggregated_distribution,
            "repairability": repairability,
            "most_fixable_types": [
                {"type": t, **r} for t, r in most_fixable
            ],
            "per_dataset": per_dataset,
        }
        self._attach_anomalies(self.results)
        
        return self.results
