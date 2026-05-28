"""
Structured repair loop with batched LLM calls.

Takes failed candidates, extracts error feedback, and generates repair
candidates. Uses pipeline.llm.repair_batch() for efficient batched inference.
Truncates error messages to 200 chars to minimize token usage.
"""

import time
from typing import Dict, List, Tuple, Optional

from config import CONFIG
from pipeline.executor import ExecutionResult, ErrorType


# Error type display names for repair prompts
_ERROR_DESCRIPTIONS = {
    ErrorType.SYNTAX: "Syntax Error",
    ErrorType.RUNTIME_TYPE: "Runtime Type Error",
    ErrorType.RUNTIME_VALUE: "Runtime Value Error",
    ErrorType.RUNTIME_RESOURCE: "Runtime Resource Error",
    ErrorType.LOGICAL: "Logical Error (test assertion failed)",
    ErrorType.TIMEOUT: "Timeout (execution took too long)",
}

# Max error message length in repair prompts (token optimization)
MAX_ERROR_LENGTH = 200

# Stop sequences for repair
REPAIR_STOP_SEQUENCES = ["\n\nclass ", "\n\ndef ", "\n\n#", "\n\nif __name__"]


def _finalize_humaneval_repair(prompt: str, repaired: str) -> str:
    """Return full HumanEval code, using a full def if provided."""
    stripped = repaired.lstrip()
    if stripped.startswith("def "):
        return repaired.strip()

    if not repaired:
        return prompt

    lines = repaired.splitlines()
    needs_indent = any(line.strip() and not line.startswith((" ", "\t")) for line in lines)
    if needs_indent:
        lines = [("    " + line) if line.strip() else "" for line in lines]
    return prompt + "\n".join(lines)


def _build_repair_prompt(
    original_code: str,
    error_info: ExecutionResult,
    problem: Dict,
) -> Tuple[str, str]:
    """
    Build a compact repair prompt from failed code and error information.

    Optimized for minimal token usage:
    - System message is short and direct
    - Error message truncated to MAX_ERROR_LENGTH chars
    - Problem description omitted (model has code context)
    """
    system_msg = (
        "You are an expert Python debugger. Fix the code based on the error. "
        "Return ONLY the corrected complete function, no explanations."
    )

    error_desc = _ERROR_DESCRIPTIONS.get(error_info.error_type, "Unknown Error")
    error_msg = error_info.error_message[:MAX_ERROR_LENGTH]

    user_msg = (
        f"Fix this {error_desc}:\n\n"
        f"```python\n{original_code}\n```\n\n"
        f"Error:\n```\n{error_msg}\n```"
    )

    return system_msg, user_msg


def repair_candidates(
    failed_candidates: List[Tuple[int, str, ExecutionResult]],
    problem: Dict,
    repair_rounds: int = None,
    fixes_per_failure: int = None,
    client=None,
) -> Tuple[List[str], List[int], Dict]:
    """
    Repair failed candidates through structured error-guided repair.

    Uses batched LLM calls: all failures in a round are sent as one batch.

    Args:
        failed_candidates: List of (slot_index, code, ExecutionResult) tuples
        problem: The original problem dict
        repair_rounds: Number of repair iterations (default: CONFIG.repair_rounds)
        fixes_per_failure: Repair candidates per failure (default: CONFIG.fixes_per_failure)
        client: ChatOpenAI or OpenAI client (default: creates from CONFIG)

    Returns:
        Tuple of (list of repaired code strings, list of slot indices, repair_stats dict)
    """
    if repair_rounds is None:
        repair_rounds = CONFIG.repair_rounds
    if fixes_per_failure is None:
        fixes_per_failure = CONFIG.fixes_per_failure

    all_repairs = []
    all_slot_indices = []
    round_stats = []
    total_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    current_failures = list(failed_candidates)

    for round_idx in range(repair_rounds):
        if not current_failures:
            break

        round_start = time.perf_counter()

        # Build all repair prompts for this round
        prompts = []
        slot_indices_for_prompts = []
        for slot_index, code, error_result in current_failures:
            sys_msg, usr_msg = _build_repair_prompt(code, error_result, problem)
            prompts.append((sys_msg, usr_msg))
            slot_indices_for_prompts.append(slot_index)

        round_repairs = []

        if client is None:
            # Use batched repair via pipeline.llm
            from pipeline.llm import create_llm, repair_batch, normalize_code_output

            llm = create_llm(temperature=CONFIG.repair_temperature)
            batch_results, stats = repair_batch(
                llm, prompts,
                n_per_prompt=fixes_per_failure,
                temperature=CONFIG.repair_temperature,
                stop=REPAIR_STOP_SEQUENCES,
            )

            total_input_tokens += stats["input_tokens"]
            total_output_tokens += stats["output_tokens"]

            for i, repairs_for_failure in enumerate(batch_results):
                slot_idx = slot_indices_for_prompts[i]
                for repaired in repairs_for_failure:
                    if repaired:
                        normalized = normalize_code_output(repaired)
                        if problem.get("dataset") == "humaneval":
                            normalized = _finalize_humaneval_repair(problem["prompt"], normalized)
                        round_repairs.append(normalized)
                        all_slot_indices.append(slot_idx)
        else:
            # Backward compat: use raw OpenAI client
            from pipeline.llm import chat_completions_create, normalize_code_output

            for i, (sys_msg, usr_msg) in enumerate(prompts):
                slot_idx = slot_indices_for_prompts[i]
                try:
                    response = chat_completions_create(
                        client,
                        model=CONFIG.model_name,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": usr_msg},
                        ],
                        temperature=CONFIG.repair_temperature,
                        max_tokens=CONFIG.max_tokens,
                        n=fixes_per_failure,
                        stop=REPAIR_STOP_SEQUENCES,
                    )

                    for choice in response.choices:
                        repaired = choice.message.content
                        if repaired:
                            normalized = normalize_code_output(repaired)
                            if problem.get("dataset") == "humaneval":
                                normalized = _finalize_humaneval_repair(problem["prompt"], normalized)
                            round_repairs.append(normalized)
                            all_slot_indices.append(slot_idx)

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

    stats = {
        "task_id": problem.get("task_id", ""),
        "total_repairs": len(all_repairs),
        "repair_time": total_time,
        "repair_rounds": round_stats,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }

    return all_repairs, all_slot_indices, stats
