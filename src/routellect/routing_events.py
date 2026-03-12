"""Stable event schema shared by harness and routellect (Step 5).

These dataclasses define the shape of routing telemetry data.
The harness emits events using these types; routellect consumes them
for learning.  Both sides depend on this module so the schema stays
in sync.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .protocols import ModelCapability, RoutingDecision, RoutingOutcome


# ---------------------------------------------------------------------------
# model_universe_snapshot — emitted once per session / when universe changes
# ---------------------------------------------------------------------------


@dataclass
class ModelUniverseSnapshot:
    """Frozen snapshot of the model universe at decision time."""

    models: list[ModelCapability]
    snapshot_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            self.snapshot_id = self.compute_hash()

    def compute_hash(self) -> str:
        canonical = json.dumps(
            [
                {"backend": m.backend, "provider": m.provider, "model_id": m.model_id}
                for m in sorted(self.models, key=lambda m: (m.backend, m.model_id))
            ],
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# routing_decision_event — emitted before each invocation
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecisionEvent:
    """Full audit record for a single routing decision."""

    task_fingerprint: dict[str, Any]
    decision: RoutingDecision
    universe_snapshot_id: str
    constraints: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# routing_outcome_event — emitted after each invocation completes
# ---------------------------------------------------------------------------


@dataclass
class RoutingOutcomeEvent:
    """Full audit record for a routing outcome."""

    decision: RoutingDecision
    outcome: RoutingOutcome
    universe_snapshot_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
