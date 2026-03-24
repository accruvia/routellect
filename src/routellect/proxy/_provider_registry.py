"""Build the model universe from stored credentials and litellm."""

from __future__ import annotations

from routellect.protocols import ModelCapability

# Static model registry per provider.  We enumerate a useful subset rather
# than querying litellm's full list (which can be noisy and includes
# deprecated models).
#
# `tier` orders models from best (1) to budget (5).  The graduated demotion
# selector starts at tier 1 and steps down only when grades confirm the
# current tier is adequate.  Lower tier = higher capability = higher cost.
_MODEL_CATALOG: dict[str, list[dict]] = {
    "openai": [
        {"model_id": "gpt-4o", "streaming": True, "tools": True, "ctx": 128_000, "tier": 3},
        {"model_id": "gpt-4o-mini", "streaming": True, "tools": True, "ctx": 128_000, "tier": 5},
        {"model_id": "gpt-4-turbo", "streaming": True, "tools": True, "ctx": 128_000, "tier": 3},
        {"model_id": "o3-mini", "streaming": True, "tools": False, "ctx": 128_000, "tier": 4},
    ],
    "anthropic": [
        {"model_id": "claude-opus-4-6", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 1},
        {"model_id": "claude-sonnet-4-6", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 1},
        {"model_id": "claude-opus-4-5-20251101", "streaming": True, "tools": True, "ctx": 200_000, "tier": 2},
        {"model_id": "claude-sonnet-4-5-20250929", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 2},
        {"model_id": "claude-opus-4-1-20250805", "streaming": True, "tools": True, "ctx": 200_000, "tier": 3},
        {"model_id": "claude-sonnet-4-20250514", "streaming": True, "tools": True, "ctx": 200_000, "tier": 3},
        {"model_id": "claude-opus-4-20250514", "streaming": True, "tools": True, "ctx": 200_000, "tier": 3},
        {"model_id": "claude-haiku-4-5-20251001", "streaming": True, "tools": True, "ctx": 200_000, "tier": 4},
    ],
    "google": [
        {"model_id": "gemini-3.1-pro-preview", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 1},
        {"model_id": "gemini-3-flash-preview", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 2},
        {"model_id": "gemini-2.5-pro", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 2},
        {"model_id": "gemini-2.5-flash", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 3},
        {"model_id": "gemini-2.5-flash-lite", "streaming": True, "tools": True, "ctx": 1_000_000, "tier": 5},
    ],
    "groq": [
        {"model_id": "llama-3.3-70b-versatile", "streaming": True, "tools": True, "ctx": 128_000, "tier": 4},
        {"model_id": "llama-3.1-8b-instant", "streaming": True, "tools": False, "ctx": 128_000, "tier": 5},
    ],
}

# Tier descriptions for logging/display.
TIER_LABELS: dict[int, str] = {
    1: "flagship",
    2: "premium",
    3: "standard",
    4: "efficient",
    5: "budget",
}

# litellm model prefixes per provider
LITELLM_PREFIXES: dict[str, str] = {
    "openai": "",
    "anthropic": "anthropic/",
    "google": "gemini/",
    "groq": "groq/",
}


def litellm_model_name(provider: str, model_id: str) -> str:
    """Return the litellm-prefixed model name for a provider/model pair."""
    prefix = LITELLM_PREFIXES.get(provider, "")
    return f"{prefix}{model_id}"


def get_model_tier(provider: str, model_id: str) -> int:
    """Return the tier for a model, or 99 if unknown."""
    catalog = _MODEL_CATALOG.get(provider, [])
    for entry in catalog:
        if entry["model_id"] == model_id:
            return entry.get("tier", 99)
    return 99


def build_model_universe(credentials: dict[str, str]) -> list[ModelCapability]:
    """Build a list of available ModelCapability from configured providers.

    Args:
        credentials: Mapping of provider name to API key.

    Returns:
        List of ModelCapability for all models from configured providers,
        sorted by tier (best first).
    """
    models: list[ModelCapability] = []
    for provider, _key in credentials.items():
        catalog = _MODEL_CATALOG.get(provider, [])
        for entry in catalog:
            models.append(
                ModelCapability(
                    backend=provider,
                    provider=provider,
                    model_id=entry["model_id"],
                    supports_streaming=entry.get("streaming", False),
                    supports_tools=entry.get("tools", False),
                    max_context_tokens=entry.get("ctx"),
                    available=True,
                )
            )
    # Sort by tier so the selector sees them in demotion order.
    models.sort(key=lambda m: get_model_tier(m.provider, m.model_id))
    return models
