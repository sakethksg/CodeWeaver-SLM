"""
Structured repair loop.

Takes failed candidates, extracts error feedback, and generates repair
candidates using the LLM. Supports multiple repair rounds with diminishing
returns tracking.
"""

import time
from typing import Dict, List, Tuple
from openai import OpenAI

from config import CONFIG
from pipeline.executor import ExecutionResult, ErrorType


def _build_repair_prompt(
    original_code: str,
    error_info: ExecutionResult,
    problem: Dict,
) -> Tuple[str, str]:
    """Build a repair prompt from failed code and error information."""
    system_msg = (
        "You are an expert Python debugger. Fix the following code based on "
        "the error message. Return ONLY the corrected complete function, no "
        "explanations or markdown."
    )
    
    error_desc = {
        ErrorType.SYNTAX: "Syntax Error",
        ErrorType.RUNTIME_TYPE: "Runtime Type Error",
        ErrorType.RUNTIME_VALUE: "Runtime Value Error",
        ErrorType.RUNTIME_RESOURCE: "Runtime Resource Error",
        ErrorType.LOGICAL: "Logical Error (test assertion failed)",
        ErrorType.TIMEOUT: "Timeout (execution took too long)",
    }.get(error_info.error_type, "Unknown Error")
    
    user_msg = (
        f"The following code has a {error_desc}.\n\n"
        f"## Original Problem:\n{problem.get('prompt', '')}\n\n"
        f"## Failed Code:\n```python\n{original_code}\n```\n\n"
        f"## Error Output:\n```\n{error_info.error_message}\n```\n\n"
        f"Fix the code and return the complete corrected function."
    )
    
    return system_msg, user_msg


def repair_candidates(
    failed_candidates: List[Tuple[int, str, ExecutionResult]],
    problem: Dict,
    repair_rounds: int = None,
    fixes_per_failure: int = None,
    client: OpenAI = None,
) -> Tuple[List[str], List[int], Dict]:
    """
    Repair failed candidates through structured error-guided repair.
    
    Args:
        failed_candidates: List of (slot_index, code, ExecutionResult) tuples for failures
        problem: The original problem dict
        repair_rounds: Number of repair iterations (default: CONFIG.repair_rounds)
        fixes_per_failure: Repair candidates per failure (default: CONFIG.fixes_per_failure)
        client: OpenAI client
    
    Returns:
        Tuple of (list of repaired code strings, list of slot indices, repair_stats dict)
    """
    if repair_rounds is None:
        repair_rounds = CONFIG.repair_rounds
    if fixes_per_failure is None:
        fixes_per_failure = CONFIG.fixes_per_failure
    if client is None:
        client = OpenAI(
            base_url=CONFIG.api_base,
            api_key=CONFIG.api_key,
        )
    
    all_repairs = []
    all_slot_indices = []
    round_stats = []
    total_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Current set of candidates to repair
    current_failures = list(failed_candidates)
    
    for round_idx in range(repair_rounds):
        round_start = time.perf_counter()
        round_repairs = []
        
        for slot_index, code, error_result in current_failures:
            system_msg, user_msg = _build_repair_prompt(code, error_result, problem)
            
            try:
                response = client.chat.completions.create(
                    model=CONFIG.model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=CONFIG.repair_temperature,
                    max_tokens=CONFIG.max_tokens,
                    n=fixes_per_failure,
                    stop=["\n\nclass ", "\n\ndef ", "\n\n#", "\n\nif __name__"],
                )
                
                for choice in response.choices:
                    repaired = choice.message.content
                    if repaired:
                        # For HumanEval, ensure prompt is prepended
                        if problem.get("dataset") == "humaneval":
                            if not repaired.strip().startswith("def "):
                                repaired = problem["prompt"] + repaired
                        round_repairs.append(repaired)
                        all_slot_indices.append(slot_index)
                
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens
            
            except Exception as e:
                print(f"[Repairer] Error in round {round_idx}: {e}")
        
        round_time = time.perf_counter() - round_start
        total_time += round_time
        
        round_stats.append({
            "round": round_idx + 1,
            "input_failures": len(current_failures),
            "repairs_generated": len(round_repairs),
            "round_time": round_time,
        })
        
        all_repairs.extend(round_repairs)
        
        # For next round, we'd need to re-execute and find new failures
        # But we return all repairs for the executor to evaluate
        # The orchestrator handles iterative repair-execute cycles
    
    stats = {
        "task_id": problem.get("task_id", ""),
        "total_repairs": len(all_repairs),
        "repair_time": total_time,
        "repair_rounds": round_stats,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }
    
    return all_repairs, all_slot_indices, stats
