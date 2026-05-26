"""
Experiment configuration for the multi-agent evaluation system.
All parameters are centralized here for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for the evaluation experiment."""
    
    # --- Model / Inference ---
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # vLLM doesn't require a real key
    
    # --- Generation ---
    k: int = 10                            # number of candidate samples per problem
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.8])
    max_tokens: int = 1024
    
    # --- Repair ---
    repair_rounds: int = 2                 # number of repair iterations
    fixes_per_failure: int = 2             # repair candidates per failed candidate
    repair_temperature: float = 0.3        # lower temp for more focused repairs
    
    # --- Execution ---
    execution_timeout: float = 5.0         # seconds per test execution
    
    # --- Cost Model (approximate) ---
    cost_per_1k_input_tokens: float = 0.0   # self-hosted, so 0 API cost
    cost_per_1k_output_tokens: float = 0.0
    gpu_cost_per_hour: float = 2.50         # $/hr for GPU compute (adjust to your setup)
    
    # --- Concurrency ---
    max_concurrency: int = 8               # max parallel problems in pipeline
    executor_workers: int = 4              # parallel subprocess workers for execution
    
    # --- Checkpointing (research reproducibility) ---
    enable_checkpointing: bool = False
    checkpoint_db: str = "checkpoints.db"
    experiment_id: str = ""                # thread_id for checkpoint isolation
    
    # --- Datasets ---
    datasets: List[str] = field(default_factory=lambda: ["humaneval", "mbpp"])
    
    # --- Output ---
    output_dir: str = "results"
    report_file: str = "evaluation_report.md"
    
    # --- Tradeoff sweep ---
    tradeoff_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10])
    tradeoff_repair_rounds: List[int] = field(default_factory=lambda: [0, 1, 2, 3])


# Global singleton
CONFIG = ExperimentConfig()
