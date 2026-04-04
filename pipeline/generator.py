"""
Code candidate generation via vLLM OpenAI-compatible API.

Generates k candidate solutions per problem using the configured model.
Supports multiple temperature settings for diversity.
"""

import time
from typing import Dict, List, Tuple
from openai import OpenAI

from config import CONFIG


def _build_generation_prompt(problem: Dict) -> str:
    """Build a prompt for code generation from a problem dict."""
    prompt = problem["prompt"]
    entry_point = problem.get("entry_point", "")
    dataset = problem.get("dataset", "")
    
    if dataset == "humaneval":
        # HumanEval prompts already include the function signature
        system_msg = (
            "You are an expert Python programmer. Complete the following function. "
            "Return ONLY the function body, no explanations or markdown."
        )
        user_msg = f"Complete this function:\n\n{prompt}"
    else:
        # MBPP: natural language description
        system_msg = (
            "You are an expert Python programmer. Write a Python function to solve "
            "the given problem. Return ONLY the complete function definition, no "
            "explanations or markdown."
        )
        user_msg = f"Write a Python function for:\n\n{prompt}"
    
    return system_msg, user_msg


def generate_candidates(
    problem: Dict,
    k: int = None,
    temperatures: List[float] = None,
    client: OpenAI = None,
) -> Tuple[List[str], Dict]:
    """
    Generate k candidate solutions for a problem.
    
    Args:
        problem: Problem dict with prompt, entry_point, dataset
        k: Number of candidates to generate (default: CONFIG.k)
        temperatures: Temperature settings to use (default: CONFIG.temperatures)
        client: OpenAI client (default: creates one from CONFIG)
    
    Returns:
        Tuple of (list of code strings, generation_stats dict)
    """
    if k is None:
        k = CONFIG.k
    if temperatures is None:
        temperatures = CONFIG.temperatures
    if client is None:
        client = OpenAI(
            base_url=CONFIG.api_base,
            api_key=CONFIG.api_key,
        )
    
    system_msg, user_msg = _build_generation_prompt(problem)
    
    candidates = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0
    
    # Distribute k across temperatures
    per_temp = k // len(temperatures)
    remainder = k % len(temperatures)
    
    for i, temp in enumerate(temperatures):
        n_samples = per_temp + (1 if i < remainder else 0)
        if n_samples == 0:
            continue
        
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temp,
                max_tokens=CONFIG.max_tokens,
                n=n_samples,
                stop=["\n\nclass ", "\n\ndef ", "\n\n#", "\n\nif __name__"],
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
            
            for choice in response.choices:
                code = choice.message.content
                if code:
                    # For HumanEval, prepend the prompt (function signature)
                    if problem.get("dataset") == "humaneval":
                        full_code = problem["prompt"] + code
                    else:
                        full_code = code
                    candidates.append(full_code)
            
            if response.usage:
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
        
        except Exception as e:
            print(f"[Generator] Error for {problem.get('task_id', '?')}: {e}")
            elapsed = time.perf_counter() - start
            total_time += elapsed
    
    stats = {
        "task_id": problem.get("task_id", ""),
        "num_candidates": len(candidates),
        "generation_time": total_time,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }
    
    return candidates, stats
