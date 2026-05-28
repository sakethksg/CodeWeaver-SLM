"""
Centralized LLM client with batching and token tracking.

Provides:
  - create_llm(): Creates a ChatOpenAI instance pointing to vLLM
  - generate_batch(): Batched generation using n= parameter
  - repair_batch(): Batched repair using llm.batch()
  - TokenTracker: Accumulates token usage across calls
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import ExperimentConfig, CONFIG


@dataclass
class TokenTracker:
    """Tracks cumulative token usage across LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    total_time: float = 0.0

    def record(self, input_tok: int, output_tok: int, elapsed: float):
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.call_count += 1
        self.total_time += elapsed

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "call_count": self.call_count,
            "total_time": self.total_time,
        }


def create_llm(
    config: ExperimentConfig = None,
    temperature: float = None,
    max_tokens: int = None,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance pointing to vLLM or any OpenAI-compatible server.

    Args:
        config: ExperimentConfig (defaults to global CONFIG)
        temperature: Override temperature (defaults to config default)
        max_tokens: Override max_tokens (defaults to config)

    Returns:
        ChatOpenAI instance
    """
    if config is None:
        config = CONFIG

    return ChatOpenAI(
        model=config.model_name,
        base_url=config.api_base,
        api_key=config.api_key,
        temperature=temperature or config.temperatures[0],
        max_tokens=max_tokens or config.max_tokens,
    )


def chat_completions_create(client, **kwargs):
    """
    Call OpenAI-compatible chat completions across client variants.

    Supports:
      - openai.OpenAI (client.chat.completions.create)
      - openai.resources.chat.completions.Completions (client.create)
    """
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return client.chat.completions.create(**kwargs)
    if hasattr(client, "create"):
        return client.create(**kwargs)
    raise AttributeError("Unsupported OpenAI client: no chat.completions or create")


def generate_batch(
    llm: ChatOpenAI,
    system_msg: str,
    user_msg: str,
    n: int,
    temperature: float,
    stop: Optional[List[str]] = None,
) -> Tuple[List[str], Dict]:
    """
    Generate n completions in a single API call using the n= parameter.

    This is the most efficient way to generate multiple candidates:
    one API round-trip produces n completions.

    Args:
        llm: ChatOpenAI instance
        system_msg: System prompt
        user_msg: User prompt
        n: Number of completions to generate
        temperature: Sampling temperature
        stop: Stop sequences

    Returns:
        Tuple of (list of completion strings, stats dict)
    """
    start = time.perf_counter()

    try:
        # Use generate() with n= for multiple completions in one call
        response = chat_completions_create(
            llm.client,
            model=llm.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=llm.max_tokens,
            n=n,
            stop=stop,
        )
        elapsed = time.perf_counter() - start

        completions = []
        for choice in response.choices:
            text = choice.message.content
            if text:
                completions.append(text)

        stats = {
            "elapsed": elapsed,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return completions, stats

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[LLM] Generation error: {e}")
        return [], {"elapsed": elapsed, "input_tokens": 0, "output_tokens": 0}


def repair_batch(
    llm: ChatOpenAI,
    prompts: List[Tuple[str, str]],
    n_per_prompt: int = 1,
    temperature: float = 0.3,
    stop: Optional[List[str]] = None,
) -> Tuple[List[List[str]], Dict]:
    """
    Batch repair: send multiple repair prompts in parallel via llm.batch().

    Instead of making one API call per failure, this sends all repair
    prompts as a batch. vLLM can then batch them on the GPU side.

    Args:
        llm: ChatOpenAI instance
        prompts: List of (system_msg, user_msg) tuples
        n_per_prompt: Number of repair candidates per failure
        temperature: Repair temperature
        stop: Stop sequences

    Returns:
        Tuple of (list of list of repair strings per prompt, stats dict)
    """
    if not prompts:
        return [], {"elapsed": 0, "input_tokens": 0, "output_tokens": 0}

    start = time.perf_counter()

    # Build message lists for batch call
    all_results = []
    total_input = 0
    total_output = 0

    try:
        # Use direct API calls with n= for each prompt in sequence
        # (llm.batch sends each as a separate request, but vLLM batches on GPU)
        for sys_msg, usr_msg in prompts:
            response = chat_completions_create(
                llm.client,
                model=llm.model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
                ],
                temperature=temperature,
                max_tokens=llm.max_tokens,
                n=n_per_prompt,
                stop=stop,
            )

            repairs = []
            for choice in response.choices:
                text = choice.message.content
                if text:
                    repairs.append(text)
            all_results.append(repairs)

            if response.usage:
                total_input += response.usage.prompt_tokens
                total_output += response.usage.completion_tokens

    except Exception as e:
        print(f"[LLM] Repair batch error: {e}")

    elapsed = time.perf_counter() - start

    stats = {
        "elapsed": elapsed,
        "input_tokens": total_input,
        "output_tokens": total_output,
    }
    return all_results, stats
