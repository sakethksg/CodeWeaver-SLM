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


def _finalize_humaneval_code(prompt: str, body_or_code: str) -> str:
    """Return full HumanEval code, using a full def if provided."""
    stripped = body_or_code.lstrip()
    if stripped.startswith("def "):
        return body_or_code.strip()

    if not body_or_code:
        return prompt

    lines = body_or_code.splitlines()
    needs_indent = any(line.strip() and not line.startswith((" ", "\t")) for line in lines)
    if needs_indent:
        lines = [("    " + line) if line.strip() else "" for line in lines]
    return prompt + "\n".join(lines)


def _finalize_mbpp_code(code: str) -> str:
    """Extract the first function definition for MBPP when possible."""
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            return "\n".join(lines[idx:]).strip()
    return code.strip()


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
        from pipeline.llm import create_llm, generate_batch, normalize_code_output

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
                normalized = normalize_code_output(code)
                if problem.get("dataset") == "humaneval":
                    full_code = _finalize_humaneval_code(problem["prompt"], normalized)
                else:
                    full_code = _finalize_mbpp_code(normalized)
                candidates.append(full_code)
    else:
        # Backward compat: use raw OpenAI client
        from pipeline.llm import chat_completions_create, normalize_code_output

        for i, temp in enumerate(temperatures):
            n_samples = per_temp + (1 if i < remainder else 0)
            if n_samples == 0:
                continue

            start = time.perf_counter()
            try:
                response = chat_completions_create(
                    client,
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
                        normalized = normalize_code_output(code)
                        if problem.get("dataset") == "humaneval":
                            full_code = _finalize_humaneval_code(problem["prompt"], normalized)
                        else:
                            full_code = _finalize_mbpp_code(normalized)
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
