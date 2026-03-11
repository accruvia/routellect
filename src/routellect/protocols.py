"""Service protocols (interfaces) for Accruvia.

This module defines the contracts that the proprietary server implements.
These are the PUBLIC interfaces - they go in the open source client.

The server (accruvia_server) DEPENDS ON these protocols and implements them.
Clients code against these protocols, not concrete implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class RecommendationSource(Enum):
    """Where a recommendation originated."""

    LOCAL = "local"  # From local telemetry only
    ACCRUVIA = "accruvia"  # From Accruvia service
    BLENDED = "blended"  # Weighted combination


@dataclass
class Recommendation:
    """A model recommendation with confidence."""

    model_id: str
    confidence: float
    source: RecommendationSource
    reasoning: str = ""


@runtime_checkable
class AccruviaServiceProtocol(Protocol):
    """Protocol for Accruvia recommendation service.

    This is the contract that the server implements.
    Clients can use this protocol for type hints without
    depending on the server implementation.

    Usage:
        # In client code
        def get_model(service: AccruviaServiceProtocol, task: dict) -> str:
            rec = service.get_recommendation(task)
            return rec.model_id
    """

    def get_recommendation(
        self,
        task_fingerprint: dict,
        local_recommendation: Recommendation | None = None,
    ) -> Recommendation:
        """Get a model recommendation for a task.

        Args:
            task_fingerprint: Task characteristics (complexity, domain, etc.)
            local_recommendation: Optional local recommendation to blend with

        Returns:
            Recommendation with model_id and confidence
        """
        ...

    def report_outcome(
        self,
        task_fingerprint: dict,
        model_id: str,
        success: bool,
        latency_ms: int,
        cost: float,
    ) -> None:
        """Report task outcome for learning.

        Args:
            task_fingerprint: Task characteristics
            model_id: Model that was used
            success: Whether the task succeeded
            latency_ms: Execution latency
            cost: Cost incurred
        """
        ...


@runtime_checkable
class TelemetryProtocol(Protocol):
    """Protocol for telemetry collection.

    Both client and server implement this - client stores locally,
    server aggregates across customers.
    """

    def record(
        self,
        event_type: str,
        data: dict,
        is_exploration: bool = False,
    ) -> None:
        """Record a telemetry event.

        Args:
            event_type: Type of event (e.g., "model_selection", "outcome")
            data: Event data
            is_exploration: Whether this is exploration data (vs production)
        """
        ...

    def get_statistics(self, event_type: str) -> dict:
        """Get aggregated statistics for an event type."""
        ...


@runtime_checkable
class TrustRegistryProtocol(Protocol):
    """Protocol for model trust tracking.

    Tracks which models are trusted for which task types.
    Client has local registry, server has aggregated registry.
    """

    def get_trust_score(self, model_id: str, task_type: str) -> float:
        """Get trust score for a model on a task type.

        Args:
            model_id: The model identifier
            task_type: Type of task (e.g., "schema_generation", "test_writing")

        Returns:
            Trust score between 0.0 and 1.0
        """
        ...

    def update_trust(
        self,
        model_id: str,
        task_type: str,
        success: bool,
        weight: float = 1.0,
    ) -> None:
        """Update trust based on outcome.

        Args:
            model_id: The model identifier
            task_type: Type of task
            success: Whether the task succeeded
            weight: How much to weight this observation
        """
        ...

    def get_recommended_model(self, task_type: str) -> str:
        """Get the most trusted model for a task type."""
        ...


class FederatedEngineProtocol(ABC):
    """Abstract base for federated decision engines.

    Blends local and remote recommendations with learned weights.
    This is an ABC (not Protocol) because it has shared implementation.
    """

    @abstractmethod
    def blend_recommendations(
        self,
        local: Recommendation,
        remote: Recommendation,
        population_type: str,
    ) -> Recommendation:
        """Blend local and remote recommendations.

        Args:
            local: Recommendation from local telemetry
            remote: Recommendation from Accruvia service
            population_type: Type of population (for per-population weights)

        Returns:
            Blended recommendation
        """
        pass

    @abstractmethod
    def update_weights(
        self,
        population_type: str,
        local_was_correct: bool,
        remote_was_correct: bool,
    ) -> None:
        """Update blending weights based on outcomes.

        Args:
            population_type: Type of population
            local_was_correct: Whether local recommendation was correct
            remote_was_correct: Whether remote recommendation was correct
        """
        pass

    @abstractmethod
    def get_accruvia_weight(self, population_type: str) -> float:
        """Get current weight for Accruvia recommendations.

        Returns:
            Weight between 0.0 and 1.0
        """
        pass


# Re-export for convenience
__all__ = [
    "AccruviaServiceProtocol",
    "FederatedEngineProtocol",
    "Recommendation",
    "RecommendationSource",
    "TelemetryProtocol",
    "TrustRegistryProtocol",
]
