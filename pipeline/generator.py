"""
Code candidate generation via vLLM OpenAI-compatible API.

Generates k candidate solutions per problem using the configured model.
Supports multiple temperature settings for diversity.
Uses centralized LLM client from pipeline.llm for batched generation.
"""

import time
from typing import Dict, List, Tuple, Optional

from config import CONFIG


# Stop sequences for code generation
CODE_STOP_SEQUENCES = ["\n\nclass ", "\n\ndef ", "\n\n#", "\n\nif __name__"]


def _build_generation_prompt(problem: Dict) -> Tuple[str, str]:
    """Build a prompt for code generation from a problem dict."""
    prompt = problem["prompt"]
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
    client=None,
) -> Tuple[List[str], Dict]:
    """
    Generate k candidate solutions for a problem.

    Uses pipeline.llm.generate_batch() for efficient batched generation.
    Falls back to raw OpenAI client if provided for backward compat.

    Args:
        problem: Problem dict with prompt, entry_point, dataset
        k: Number of candidates to generate (default: CONFIG.k)
        temperatures: Temperature settings to use (default: CONFIG.temperatures)
        client: ChatOpenAI or OpenAI client (default: creates from CONFIG)

    Returns:
        Tuple of (list of code strings, generation_stats dict)
    """
    if k is None:
        k = CONFIG.k
    if temperatures is None:
        temperatures = CONFIG.temperatures

    system_msg, user_msg = _build_generation_prompt(problem)

    candidates = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0

    # Distribute k across temperatures
    per_temp = k // len(temperatures)
    remainder = k % len(temperatures)

    # Use pipeline.llm if no raw client provided
    if client is None:
        from pipeline.llm import create_llm, generate_batch

        for i, temp in enumerate(temperatures):
            n_samples = per_temp + (1 if i < remainder else 0)
            if n_samples == 0:
                continue

            llm = create_llm(temperature=temp)
            completions, stats = generate_batch(
                llm, system_msg, user_msg,
                n=n_samples, temperature=temp,
                stop=CODE_STOP_SEQUENCES,
            )
            total_time += stats["elapsed"]
            total_input_tokens += stats["input_tokens"]
            total_output_tokens += stats["output_tokens"]

            for code in completions:
                if problem.get("dataset") == "humaneval":
                    full_code = problem["prompt"] + code
                else:
                    full_code = code
                candidates.append(full_code)
    else:
        # Backward compat: use raw OpenAI client
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
                    stop=CODE_STOP_SEQUENCES,
                )
                elapsed = time.perf_counter() - start
                total_time += elapsed

                for choice in response.choices:
                    code = choice.message.content
                    if code:
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
