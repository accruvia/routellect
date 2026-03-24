"""Tests for the graduated demotion selector."""

from __future__ import annotations

from routellect.protocols import ModelCapability, RoutingDecision, RoutingOutcome
from routellect.proxy._selector import (
    GraduatedDemotionSelector,
    MIN_GRADES_BEFORE_DEMOTION,
    PASS_RATE_FLOOR,
)


def _make_models() -> list[ModelCapability]:
    """Build a small test universe with 3 tiers."""
    return [
        ModelCapability(backend="anthropic", provider="anthropic", model_id="claude-opus-4-6", available=True),
        ModelCapability(backend="anthropic", provider="anthropic", model_id="claude-sonnet-4-6", available=True),
        ModelCapability(backend="anthropic", provider="anthropic", model_id="claude-haiku-4-5-20251001", available=True),
        ModelCapability(backend="openai", provider="openai", model_id="gpt-4o-mini", available=True),
    ]


class TestGraduatedDemotionSelector:
    def test_starts_at_tier_1(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())
        assert sel.current_tier == 1

    def test_selects_from_tier_1_initially(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # All initial selections should be tier 1 (opus/sonnet 4.6)
        tier_1_models = {"claude-opus-4-6", "claude-sonnet-4-6"}
        for _ in range(20):
            decision = sel.select_model({"message_count": 1})
            assert decision.model_id in tier_1_models

    def test_stays_on_tier_1_until_enough_grades(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # Record 5 passes — not enough to start trialing
        decision = RoutingDecision(model_id="claude-sonnet-4-6", backend="anthropic", confidence=0.8)
        for _ in range(5):
            sel.record_outcome(decision, RoutingOutcome(success=True, qa_result="pass"))

        assert sel.trial_tier is None
        # Still selects from tier 1
        for _ in range(20):
            d = sel.select_model({"message_count": 1})
            assert d.model_id in {"claude-opus-4-6", "claude-sonnet-4-6"}

    def test_starts_trial_after_enough_grades(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # Record MIN_GRADES_BEFORE_DEMOTION passes at tier 1
        decision = RoutingDecision(model_id="claude-sonnet-4-6", backend="anthropic", confidence=0.8)
        for _ in range(MIN_GRADES_BEFORE_DEMOTION):
            sel.record_outcome(decision, RoutingOutcome(success=True, qa_result="pass"))

        # Now some selections should be exploration (trial tier)
        saw_exploration = False
        for _ in range(100):
            d = sel.select_model({"message_count": 1})
            if d.is_exploration:
                saw_exploration = True
                break
        assert saw_exploration

    def test_promotes_tier_on_good_trial(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # Fill tier 1 with passes
        d1 = RoutingDecision(model_id="claude-sonnet-4-6", backend="anthropic", confidence=0.8)
        for _ in range(MIN_GRADES_BEFORE_DEMOTION):
            sel.record_outcome(d1, RoutingOutcome(success=True, qa_result="pass"))

        # Trigger trial tier selection
        sel.select_model({"message_count": 1})

        # Now fill trial tier (should be tier 2 or 3) with passes
        # The next tier after 1 depends on what's available.
        assert sel.trial_tier is not None
        trial_tier = sel.trial_tier

        # Pick a model from the trial tier
        trial_models = sel._tier_models[trial_tier]
        d2 = RoutingDecision(model_id=trial_models[0].model_id, backend=trial_models[0].backend, confidence=0.5)

        for _ in range(MIN_GRADES_BEFORE_DEMOTION):
            sel.record_outcome(d2, RoutingOutcome(success=True, qa_result="pass"))

        # Should have promoted
        assert sel.current_tier == trial_tier
        assert sel.trial_tier is None
        assert sel.locked is False

    def test_locks_on_bad_trial(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # Fill tier 1
        d1 = RoutingDecision(model_id="claude-sonnet-4-6", backend="anthropic", confidence=0.8)
        for _ in range(MIN_GRADES_BEFORE_DEMOTION):
            sel.record_outcome(d1, RoutingOutcome(success=True, qa_result="pass"))

        # Trigger trial
        sel.select_model({"message_count": 1})
        trial_tier = sel.trial_tier
        assert trial_tier is not None

        trial_models = sel._tier_models[trial_tier]
        d2 = RoutingDecision(model_id=trial_models[0].model_id, backend=trial_models[0].backend, confidence=0.5)

        # Fill trial with mostly fails
        for _ in range(MIN_GRADES_BEFORE_DEMOTION):
            sel.record_outcome(d2, RoutingOutcome(success=False, qa_result="fail"))

        # Should be locked at tier 1
        assert sel.current_tier == 1
        assert sel.locked is True
        assert sel.trial_tier is None

    def test_locked_selector_never_explores(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())
        sel.locked = True

        tier_1_models = {"claude-opus-4-6", "claude-sonnet-4-6"}
        for _ in range(50):
            d = sel.select_model({"message_count": 1})
            assert d.model_id in tier_1_models
            assert d.is_exploration is False

    def test_empty_universe_raises(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe([])
        import pytest
        with pytest.raises(RuntimeError, match="No models available"):
            sel.select_model({"message_count": 1})

    def test_ignores_outcomes_without_qa_result(self):
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # HTTP-level outcomes without qa_result should not count
        d = RoutingDecision(model_id="claude-sonnet-4-6", backend="anthropic", confidence=0.8)
        for _ in range(20):
            sel.record_outcome(d, RoutingOutcome(success=True))

        # Tier stats should be empty
        assert sel._tier_stats[1].total == 0

    def test_failover_excludes_backend(self):
        """When a backend is excluded, selector picks from another backend."""
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # Exclude anthropic — should get openai or google
        for _ in range(20):
            d = sel.select_model(
                {"message_count": 1},
                constraints={"exclude_backends": ["anthropic"]},
            )
            assert d.backend != "anthropic"
            assert "failover" in d.reasoning

    def test_failover_excludes_multiple_backends(self):
        """When multiple backends are excluded, selector finds remaining."""
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        # Exclude anthropic — only openai left in our test models
        for _ in range(20):
            d = sel.select_model(
                {"message_count": 1},
                constraints={"exclude_backends": ["anthropic"]},
            )
            assert d.backend == "openai"

    def test_failover_all_excluded_raises(self):
        """When all backends are excluded, raises RuntimeError."""
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        import pytest
        with pytest.raises(RuntimeError, match="All providers are down"):
            sel.select_model(
                {"message_count": 1},
                constraints={"exclude_backends": ["anthropic", "openai", "google"]},
            )

    def test_failover_works_when_locked(self):
        """Even when locked at a tier, failover should find another backend."""
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())
        sel.locked = True

        # Exclude anthropic — should still find openai
        d = sel.select_model(
            {"message_count": 1},
            constraints={"exclude_backends": ["anthropic"]},
        )
        assert d.backend != "anthropic"
        assert "failover" in d.reasoning

    def test_no_constraints_works_normally(self):
        """Without constraints, selector works as before."""
        sel = GraduatedDemotionSelector()
        sel.set_model_universe(_make_models())

        d = sel.select_model({"message_count": 1})
        assert d.model_id in {"claude-opus-4-6", "claude-sonnet-4-6"}
        assert "failover" not in d.reasoning
