"""
Abstract base agent interface.

All evaluation agents inherit from this class and implement the `analyze` method.
Provides common utilities for metric computation, schema validation, and
anomaly detection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.special import comb


class BaseAgent(ABC):
    """Base class for all evaluation agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}
        self.anomalies: List[str] = []
        
        # Load prompt from registry (lazy import to avoid circular)
        self._prompt: Optional[str] = None
    
    @property
    def prompt(self) -> str:
        """Get the system prompt for this agent."""
        if self._prompt is None:
            from agents.prompts import AGENT_PROMPTS
            self._prompt = AGENT_PROMPTS.get(self.name, "")
        return self._prompt
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze evaluation data and compute metrics.
        
        Args:
            data: Dictionary containing raw evaluation data
        
        Returns:
            Dictionary of computed metrics matching the output schema
            defined in this agent's prompt.
        """
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """Return the latest analysis results."""
        return self.results
    
    # ─────────────────────────────────────────────
    # Anomaly detection
    # ─────────────────────────────────────────────
    
    def flag_anomaly(self, message: str):
        """Flag an anomaly detected during analysis."""
        tag = f"[ANOMALY] {message}"
        self.anomalies.append(tag)
        print(f"  ⚠ {self.name}: {tag}")
    
    def flag_inconsistency(self, message: str):
        """Flag a cross-agent inconsistency."""
        tag = f"[INCONSISTENCY] {message}"
        self.anomalies.append(tag)
        print(f"  ⚠ {self.name}: {tag}")
    
    def _attach_anomalies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Attach anomalies list to results dict."""
        results["anomalies"] = self.anomalies.copy()
        return results
    
    # ─────────────────────────────────────────────
    # Schema validation
    # ─────────────────────────────────────────────
    
    @staticmethod
    def validate_percentage_sum(
        distribution: Dict[str, Dict], 
        tolerance: float = 0.5
    ) -> Optional[str]:
        """
        Validate that percentage values in a distribution sum to ~100%.
        Returns error message if invalid, None if valid.
        """
        total = sum(
            v.get("percentage", 0) for v in distribution.values()
        )
        if abs(total - 100.0) > tolerance:
            return (
                f"Distribution percentages sum to {total:.2f}%, "
                f"expected ~100% (tolerance ±{tolerance}%)"
            )
        return None
    
    @staticmethod
    def validate_monotonic_non_decreasing(
        values: List[float], labels: List[str]
    ) -> List[str]:
        """
        Check that a sequence is monotonically non-decreasing.
        Returns list of anomaly messages for violations.
        """
        anomalies = []
        for i in range(1, len(values)):
            if values[i] < values[i - 1] - 1e-9:  # tolerance for float
                anomalies.append(
                    f"Non-monotonic: {labels[i-1]}={values[i-1]:.4f} > "
                    f"{labels[i]}={values[i]:.4f}"
                )
        return anomalies
    
    @staticmethod
    def validate_upper_bound(
        upper: float, lower: float,
        upper_name: str, lower_name: str,
    ) -> Optional[str]:
        """Validate that upper >= lower. Returns error message if violated."""
        if upper < lower - 1e-9:
            return (
                f"{upper_name}={upper:.4f} < {lower_name}={lower:.4f} "
                f"(expected {upper_name} >= {lower_name})"
            )
        return None
    
    # ─────────────────────────────────────────────
    # Shared metric utilities
    # ─────────────────────────────────────────────
    
    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """
        Unbiased estimator of pass@k (Chen et al., 2021).
        
        Formula: pass@k = 1 - C(n-c, k) / C(n, k)
        
        Args:
            n: Total number of samples
            c: Number of correct samples
            k: k value for pass@k
        
        Returns:
            pass@k probability
        """
        if n - c < k:
            return 1.0
        return 1.0 - float(comb(n - c, k) / comb(n, k))
    
    @staticmethod
    def compute_pass_at_k_for_problems(
        problems_data: List[Dict],
        k_values: List[int],
        key_n: str = "num_samples",
        key_c: str = "num_correct",
    ) -> Dict[int, float]:
        """
        Compute average pass@k across all problems.
        
        Excludes problems where n < k (per prompt specification).
        
        Args:
            problems_data: List of dicts with num_samples and num_correct
            k_values: List of k values to compute
        
        Returns:
            Dict mapping k -> average pass@k
        """
        results = {}
        for k in k_values:
            scores = []
            for p in problems_data:
                n = p[key_n]
                c = p[key_c]
                if n >= k:
                    scores.append(BaseAgent.pass_at_k(n, c, k))
            results[k] = float(np.mean(scores)) if scores else 0.0
        return results
    
    @staticmethod
    def safe_div(a: float, b: float, default: float = 0.0) -> float:
        """Safe division with default for zero denominator."""
        return a / b if b != 0 else default
