"""Tests for the routing event schema (Step 5)."""

from __future__ import annotations

from routellect.protocols import ModelCapability, RoutingDecision, RoutingOutcome
from routellect.routing_events import (
    ModelUniverseSnapshot,
    RoutingDecisionEvent,
    RoutingOutcomeEvent,
)


def _sample_models() -> list[ModelCapability]:
    return [
        ModelCapability(backend="claude", provider="anthropic", model_id="claude-sonnet-4-6"),
        ModelCapability(backend="codex", provider="openai", model_id="gpt-4o"),
    ]


class TestModelUniverseSnapshot:
    def test_hash_is_deterministic_for_same_models(self):
        a = ModelUniverseSnapshot(models=_sample_models())
        b = ModelUniverseSnapshot(models=_sample_models())
        assert a.snapshot_id == b.snapshot_id

    def test_hash_changes_when_models_differ(self):
        a = ModelUniverseSnapshot(models=_sample_models())
        b = ModelUniverseSnapshot(models=[_sample_models()[0]])
        assert a.snapshot_id != b.snapshot_id

    def test_hash_is_order_independent(self):
        a = ModelUniverseSnapshot(models=_sample_models())
        b = ModelUniverseSnapshot(models=list(reversed(_sample_models())))
        assert a.snapshot_id == b.snapshot_id

    def test_to_dict_includes_all_fields(self):
        snap = ModelUniverseSnapshot(models=_sample_models())
        d = snap.to_dict()
        assert "models" in d
        assert "snapshot_id" in d
        assert "created_at" in d
        assert len(d["models"]) == 2

    def test_snapshot_id_is_16_hex_chars(self):
        snap = ModelUniverseSnapshot(models=_sample_models())
        assert len(snap.snapshot_id) == 16
        int(snap.snapshot_id, 16)  # Should not raise


class TestRoutingDecisionEvent:
    def test_to_dict_captures_decision_and_fingerprint(self):
        decision = RoutingDecision(
            model_id="claude-sonnet-4-6",
            backend="claude",
            confidence=0.9,
            universe_hash="abc123",
        )
        event = RoutingDecisionEvent(
            task_fingerprint={"complexity": "medium"},
            decision=decision,
            universe_snapshot_id="abc123",
        )
        d = event.to_dict()
        assert d["task_fingerprint"]["complexity"] == "medium"
        assert d["decision"]["model_id"] == "claude-sonnet-4-6"
        assert d["universe_snapshot_id"] == "abc123"
        assert "timestamp" in d


class TestRoutingOutcomeEvent:
    def test_to_dict_captures_outcome_metrics(self):
        decision = RoutingDecision(
            model_id="gpt-4o",
            backend="codex",
            confidence=1.0,
            universe_hash="def456",
        )
        outcome = RoutingOutcome(
            success=True,
            latency_ms=1234,
            input_tokens=500,
            output_tokens=200,
            cost=0.05,
        )
        event = RoutingOutcomeEvent(
            decision=decision,
            outcome=outcome,
            universe_snapshot_id="def456",
        )
        d = event.to_dict()
        assert d["outcome"]["success"] is True
        assert d["outcome"]["latency_ms"] == 1234
        assert d["decision"]["backend"] == "codex"
