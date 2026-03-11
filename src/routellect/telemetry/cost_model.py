"""API cost estimation helpers for A/B test token data."""

from __future__ import annotations

PRICE_TABLE: dict[str, dict[str, float]] = {
    "anthropic": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
    "openai": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "google": {"input": 1.25, "cached_input": 0.315, "output": 5.00},
    "groq": {"input": 0.59, "cached_input": 0.59, "output": 0.79},
}

_TOKENS_PER_MILLION = 1_000_000


def estimate_cost(
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    provider: str = "anthropic",
) -> dict[str, float]:
    """Estimate API-equivalent USD cost from token usage."""
    provider_rates = PRICE_TABLE.get(provider)
    if provider_rates is None:
        supported = ", ".join(sorted(PRICE_TABLE))
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}")

    input_cost = (input_tokens / _TOKENS_PER_MILLION) * provider_rates["input"]
    cached_cost = (cached_tokens / _TOKENS_PER_MILLION) * provider_rates["cached_input"]
    output_cost = (output_tokens / _TOKENS_PER_MILLION) * provider_rates["output"]
    total_cost = input_cost + cached_cost + output_cost
    savings_from_caching = ((cached_tokens / _TOKENS_PER_MILLION) * provider_rates["input"]) - cached_cost

    return {
        "input_cost": input_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "savings_from_caching": savings_from_caching,
    }


def format_cost_comparison(runs: list[dict]) -> str:
    """Format a readable cost comparison for multiple A/B runs."""
    if not runs:
        return "No runs to compare."

    cost_rows: list[tuple[str, str, dict[str, float]]] = []
    for index, run in enumerate(runs, start=1):
        provider = str(run.get("provider", "anthropic"))
        label = str(run.get("mode") or run.get("name") or f"run_{index}")
        costs = estimate_cost(
            input_tokens=int(run.get("input_tokens", 0)),
            cached_tokens=int(run.get("cached_tokens", 0)),
            output_tokens=int(run.get("output_tokens", 0)),
            provider=provider,
        )
        cost_rows.append((label, provider, costs))

    cheapest_total = min(costs["total_cost"] for _, _, costs in cost_rows)
    lines = ["API Cost Comparison"]
    for label, provider, costs in cost_rows:
        marker = " <-- cheapest" if costs["total_cost"] == cheapest_total else ""
        lines.append(
            f"- {label} ({provider}): "
            f"total=${costs['total_cost']:.6f} "
            f"(input=${costs['input_cost']:.6f}, "
            f"cached=${costs['cached_cost']:.6f}, "
            f"output=${costs['output_cost']:.6f}, "
            f"cache_savings=${costs['savings_from_caching']:.6f}){marker}"
        )

    return "\n".join(lines)


def get_model_cost(model_id: str) -> dict[str, float]:
    """Return per-token input/output rates for a model from MODEL_REGISTRY."""
    # Lazy import avoids circular dependency during module import.
    raise ValueError("get_model_cost requires an application-specific model registry")
