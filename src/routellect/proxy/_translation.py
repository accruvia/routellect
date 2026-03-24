"""Thin wrapper around litellm for provider translation.

Isolates litellm from the rest of the proxy for testability.  All LLM
calls go through this module so tests can mock ``forward_completion``
without touching litellm internals.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from routellect.proxy._provider_registry import litellm_model_name


def _inject_credentials(credentials: dict[str, str]) -> None:
    """Set provider API keys as environment variables for litellm."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    for provider, key in credentials.items():
        env_var = env_map.get(provider)
        if env_var and env_var not in os.environ:
            os.environ[env_var] = key


async def forward_completion(
    *,
    provider: str,
    model_id: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
    credentials: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Forward a chat completion request to the real provider via litellm.

    Args:
        provider: Provider name (e.g. "openai", "anthropic").
        model_id: Model identifier (e.g. "gpt-4o").
        messages: OpenAI-format message list.
        stream: Whether to stream the response.
        credentials: Provider credentials to inject if not already in env.
        **kwargs: Additional litellm parameters (temperature, max_tokens, etc.).

    Returns:
        litellm response object (ModelResponse or generator for streaming).
    """
    import litellm

    if credentials:
        _inject_credentials(credentials)

    litellm_model = litellm_model_name(provider, model_id)

    response = await litellm.acompletion(
        model=litellm_model,
        messages=messages,
        stream=stream,
        **kwargs,
    )
    return response
