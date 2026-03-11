"""Tests for API cost estimation from A/B token usage."""

import pytest

from routellect.telemetry.cost_model import (
    PRICE_TABLE,
    estimate_cost,
    format_cost_comparison,
)


def test_price_table_has_required_providers():
    for provider in ("anthropic", "openai", "google", "groq"):
        assert provider in PRICE_TABLE
        rates = PRICE_TABLE[provider]
        assert "input" in rates
        assert "cached_input" in rates
        assert "output" in rates


def test_estimate_cost_breakdown():
    result = estimate_cost(
        input_tokens=2_000_000,
        cached_tokens=1_000_000,
        output_tokens=3_000_000,
        provider="openai",
    )
    assert result["input_cost"] == pytest.approx(5.0)
    assert result["cached_cost"] == pytest.approx(1.25)
    assert result["output_cost"] == pytest.approx(30.0)
    assert result["total_cost"] == pytest.approx(36.25)
    assert result["savings_from_caching"] == pytest.approx(1.25)


def test_estimate_cost_defaults_to_anthropic():
    result = estimate_cost(input_tokens=1_000_000, cached_tokens=0, output_tokens=0)
    assert result["input_cost"] == pytest.approx(3.0)


def test_estimate_cost_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        estimate_cost(1, 1, 1, provider="fakeprovider")


def test_savings_from_caching():
    # With anthropic: input=$3/M, cached=$0.30/M
    # 1M cached tokens: savings = 1M * ($3 - $0.30) / 1M = $2.70
    result = estimate_cost(
        input_tokens=1_000_000,
        cached_tokens=1_000_000,
        output_tokens=0,
        provider="anthropic",
    )
    assert result["savings_from_caching"] == pytest.approx(2.70)


def test_format_cost_comparison_marks_cheapest():
    runs = [
        {"mode": "direct", "input_tokens": 100_000, "cached_tokens": 50_000, "output_tokens": 10_000},
        {"mode": "schema", "input_tokens": 200_000, "cached_tokens": 100_000, "output_tokens": 20_000},
    ]
    output = format_cost_comparison(runs)
    assert "API Cost Comparison" in output
    assert "direct" in output
    assert "schema" in output
    assert output.count("<-- cheapest") == 1
    # Direct should be cheapest (fewer tokens)
    assert "direct" in output.split("<-- cheapest")[0].split("\n")[-1]


def test_format_cost_comparison_empty_runs():
    assert format_cost_comparison([]) == "No runs to compare."
