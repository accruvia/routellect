"""Routellect public package."""

from routellect.protocols import (
    AccruviaServiceProtocol,
    FederatedEngineProtocol,
    ModelCapability,
    ModelSelectorProtocol,
    Recommendation,
    RecommendationSource,
    RoutellectServiceProtocol,
    RoutingDecision,
    RoutingOutcome,
    TelemetryProtocol,
    TrustRegistryProtocol,
)
from routellect.routing_events import (
    ModelUniverseSnapshot,
    RoutingDecisionEvent,
    RoutingOutcomeEvent,
)

__all__ = [
    # Canonical routing interface
    "ModelCapability",
    "ModelSelectorProtocol",
    "RoutingDecision",
    "RoutingOutcome",
    # Routing event schema
    "ModelUniverseSnapshot",
    "RoutingDecisionEvent",
    "RoutingOutcomeEvent",
    # Legacy types (still in use)
    "AccruviaServiceProtocol",
    "FederatedEngineProtocol",
    "Recommendation",
    "RecommendationSource",
    "RoutellectServiceProtocol",
    "TelemetryProtocol",
    "TrustRegistryProtocol",
]

__version__ = "0.1.0"
