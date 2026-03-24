"""Default cost-ranked model selector.

Picks the cheapest available model.  Records outcomes for future learning
but the current implementation does not yet use historical data to adjust
selections.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from routellect.protocols import ModelCapability, RoutingDecision, RoutingOutcome
from routellect.telemetry.cost_model import PRICE_TABLE


@dataclass
class CostRankedSelector:
    """ModelSelectorProtocol implementation that ranks by cost."""

    _models: list[ModelCapability] = field(default_factory=list)
    _outcomes: list[tuple[RoutingDecision, RoutingOutcome]] = field(default_factory=list)
    exploration_rate: float = 0.1

    def set_model_universe(self, models: list[ModelCapability]) -> None:
        self._models = [m for m in models if m.available]

    def select_model(
        self,
        task_fingerprint: dict,
        constraints: dict | None = None,
    ) -> RoutingDecision:
        if not self._models:
            raise RuntimeError("No models available. Run setup first: python -m routellect.proxy --setup")

        # Exploration: pick a random model some percentage of the time
        # to build quality data in the routellect DB.
        if random.random() < self.exploration_rate:
            chosen = random.choice(self._models)
            return RoutingDecision(
                model_id=chosen.model_id,
                backend=chosen.backend,
                confidence=0.5,
                reasoning="exploration",
                is_exploration=True,
            )

        # Rank by cost (cheapest output token rate).
        def _cost_key(m: ModelCapability) -> float:
            rates = PRICE_TABLE.get(m.provider)
            if rates is None:
                return float("inf")
            return rates.get("output", float("inf"))

        ranked = sorted(self._models, key=_cost_key)
        chosen = ranked[0]

        return RoutingDecision(
            model_id=chosen.model_id,
            backend=chosen.backend,
            confidence=0.8,
            reasoning=f"cheapest available ({chosen.provider})",
        )

    def record_outcome(
        self,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        self._outcomes.append((decision, outcome))
