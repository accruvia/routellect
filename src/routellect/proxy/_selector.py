"""Graduated demotion model selector.

Starts with the best available models (tier 1) and cautiously steps down
to cheaper tiers only when grading data confirms the current tier is
adequate.  Stops demoting the moment quality degrades.

This is conservative by design — we are spending other people's money and
must not waste it on models that produce bad results.

Tier progression:
    1 (flagship)  →  2 (premium)  →  3 (standard)  →  4 (efficient)  →  5 (budget)

Each tier must accumulate enough "pass" grades before the selector will
trial the next tier down.  If a trial tier's pass rate drops below the
floor, demotion stops and the selector locks to the last known-good tier.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from routellect.protocols import ModelCapability, RoutingDecision, RoutingOutcome
from routellect.proxy._provider_registry import TIER_LABELS, get_model_tier

logger = logging.getLogger("routellect.proxy")

# Minimum grades at a tier before we consider trialing the next tier down.
MIN_GRADES_BEFORE_DEMOTION = 10

# Pass rate threshold.  If a trial tier falls below this, stop demoting.
PASS_RATE_FLOOR = 0.70

# Fraction of requests sent to the trial tier (the rest stay on current).
TRIAL_RATE = 0.15


@dataclass
class _TierStats:
    """Accumulated grades for one tier."""

    passes: int = 0
    fails: int = 0
    mixed: int = 0

    @property
    def total(self) -> int:
        return self.passes + self.fails + self.mixed

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 1.0  # Assume good until proven otherwise.
        return self.passes / self.total


@dataclass
class GraduatedDemotionSelector:
    """ModelSelectorProtocol implementation with stair-step demotion.

    Attributes:
        current_tier: The tier we're confidently serving from.
        trial_tier: The next tier down that we're cautiously testing.
                    None if we're not trialing.
        locked: If True, demotion has been halted because a trial failed.
    """

    current_tier: int = 1
    trial_tier: int | None = None
    locked: bool = False
    _models: list[ModelCapability] = field(default_factory=list)
    _tier_models: dict[int, list[ModelCapability]] = field(default_factory=dict)
    _tier_stats: dict[int, _TierStats] = field(default_factory=lambda: defaultdict(_TierStats))
    _available_tiers: list[int] = field(default_factory=list)

    def set_model_universe(self, models: list[ModelCapability]) -> None:
        """Organize models by tier and determine available tiers."""
        self._models = [m for m in models if m.available]
        self._tier_models = {}
        for m in self._models:
            tier = get_model_tier(m.provider, m.model_id)
            self._tier_models.setdefault(tier, []).append(m)

        self._available_tiers = sorted(self._tier_models.keys())

        if not self._available_tiers:
            return

        # Start at the best available tier.
        self.current_tier = self._available_tiers[0]

        tier_summary = []
        for t in self._available_tiers:
            label = TIER_LABELS.get(t, f"tier-{t}")
            model_names = [m.model_id for m in self._tier_models[t]]
            tier_summary.append(f"  Tier {t} ({label}): {', '.join(model_names)}")

        logger.info(
            "Model universe set. %d models across %d tiers:\n%s",
            len(self._models),
            len(self._available_tiers),
            "\n".join(tier_summary),
        )

    def _next_tier(self) -> int | None:
        """Return the next tier down from current, or None if at bottom."""
        try:
            idx = self._available_tiers.index(self.current_tier)
        except ValueError:
            return None
        if idx + 1 < len(self._available_tiers):
            return self._available_tiers[idx + 1]
        return None

    def _pick_from_tier(self, tier: int) -> ModelCapability:
        """Pick a model from the given tier (random within tier)."""
        candidates = self._tier_models.get(tier, [])
        if not candidates:
            # Fallback to current tier
            candidates = self._tier_models.get(self.current_tier, self._models)
        return random.choice(candidates)

    def select_model(
        self,
        task_fingerprint: dict,
        constraints: dict | None = None,
    ) -> RoutingDecision:
        if not self._models:
            raise RuntimeError(
                "No models available. Run setup first: python -m routellect.proxy --setup"
            )

        # If locked or no next tier, serve from current tier.
        if self.locked or self._next_tier() is None:
            chosen = self._pick_from_tier(self.current_tier)
            return RoutingDecision(
                model_id=chosen.model_id,
                backend=chosen.backend,
                confidence=0.9,
                reasoning=f"tier {self.current_tier} ({TIER_LABELS.get(self.current_tier, '?')})"
                + (" [locked]" if self.locked else ""),
            )

        # Check if current tier has enough data to start trialing.
        current_stats = self._tier_stats[self.current_tier]
        if current_stats.total < MIN_GRADES_BEFORE_DEMOTION:
            # Not enough data yet — stay on current tier to build confidence.
            chosen = self._pick_from_tier(self.current_tier)
            return RoutingDecision(
                model_id=chosen.model_id,
                backend=chosen.backend,
                confidence=0.8,
                reasoning=f"tier {self.current_tier} (building confidence: {current_stats.total}/{MIN_GRADES_BEFORE_DEMOTION})",
            )

        # Current tier has enough data and passes the floor — trial next tier.
        next_t = self._next_tier()
        if next_t is not None and self.trial_tier is None:
            self.trial_tier = next_t
            logger.info(
                "Starting trial of tier %d (%s). Current tier %d pass rate: %.0f%%",
                next_t,
                TIER_LABELS.get(next_t, "?"),
                self.current_tier,
                current_stats.pass_rate * 100,
            )

        # Send TRIAL_RATE of requests to the trial tier.
        if self.trial_tier is not None and random.random() < TRIAL_RATE:
            chosen = self._pick_from_tier(self.trial_tier)
            return RoutingDecision(
                model_id=chosen.model_id,
                backend=chosen.backend,
                confidence=0.5,
                reasoning=f"trial tier {self.trial_tier} ({TIER_LABELS.get(self.trial_tier, '?')})",
                is_exploration=True,
            )

        # Default: serve from current tier.
        chosen = self._pick_from_tier(self.current_tier)
        return RoutingDecision(
            model_id=chosen.model_id,
            backend=chosen.backend,
            confidence=0.85,
            reasoning=f"tier {self.current_tier} ({TIER_LABELS.get(self.current_tier, '?')})",
        )

    def record_outcome(
        self,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        """Update tier stats and evaluate whether to promote, hold, or lock."""
        tier = get_model_tier(decision.backend, decision.model_id)
        stats = self._tier_stats[tier]

        qa = outcome.qa_result
        if qa == "pass":
            stats.passes += 1
        elif qa == "fail":
            stats.fails += 1
        elif qa == "mixed":
            stats.mixed += 1
        else:
            # No grading data yet (just HTTP-level outcome).
            # Don't count toward tier stats — wait for the grader.
            return

        # Evaluate trial tier.
        if self.trial_tier is not None and tier == self.trial_tier:
            trial_stats = self._tier_stats[self.trial_tier]

            if trial_stats.total >= MIN_GRADES_BEFORE_DEMOTION:
                if trial_stats.pass_rate >= PASS_RATE_FLOOR:
                    # Trial passed — promote to this tier and look for next.
                    old_tier = self.current_tier
                    self.current_tier = self.trial_tier
                    self.trial_tier = None
                    logger.info(
                        "Tier %d (%s) promoted. Pass rate: %.0f%%. "
                        "Demoted from tier %d.",
                        self.current_tier,
                        TIER_LABELS.get(self.current_tier, "?"),
                        trial_stats.pass_rate * 100,
                        old_tier,
                    )
                else:
                    # Trial failed — lock at current tier.
                    logger.warning(
                        "Tier %d (%s) FAILED trial. Pass rate: %.0f%% (floor: %.0f%%). "
                        "Locking at tier %d (%s).",
                        self.trial_tier,
                        TIER_LABELS.get(self.trial_tier, "?"),
                        trial_stats.pass_rate * 100,
                        PASS_RATE_FLOOR * 100,
                        self.current_tier,
                        TIER_LABELS.get(self.current_tier, "?"),
                    )
                    self.locked = True
                    self.trial_tier = None


# Keep backward compat for tests that reference the old name.
CostRankedSelector = GraduatedDemotionSelector
