"""Public routing and telemetry protocols for Routellect.

These contracts are intended to be stable public interfaces for the OSS module.
Hosted services or downstream applications may implement them, but the core
package should not depend on any proprietary deployment shape.
"""

from abc import ABC, abstractmethod, update_abstractmethods
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class RecommendationSource(Enum):
    """Where a recommendation originated."""

    LOCAL = "local"  # From local telemetry only
    ROUTELLECT = "routellect"  # From a hosted Routellect service
    ACCRUVIA = "routellect"  # Backward-compatible alias
    BLENDED = "blended"  # Weighted combination


@dataclass
class Recommendation:
    """A model recommendation with confidence."""

    model_id: str
    confidence: float
    source: RecommendationSource
    reasoning: str = ""


@runtime_checkable
class RoutellectServiceProtocol(Protocol):
    """Protocol for a hosted routing recommendation service.

    This is the public contract that optional hosted backends can implement.
    Clients can use this protocol for type hints without depending on any
    specific deployment or private server implementation.

    Usage:
        # In client code
        def get_model(service: RoutellectServiceProtocol, task: dict) -> str:
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

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Preserve compatibility for subclasses that still implement the
        # legacy Accruvia-named method, while making the Routellect-named
        # method the canonical abstract interface.
        subclass_dict = cls.__dict__
        implements_new = "get_routellect_weight" in subclass_dict
        implements_legacy = "get_accruvia_weight" in subclass_dict

        if implements_legacy and not implements_new:
            cls.get_routellect_weight = cls.get_accruvia_weight
        elif implements_new and not implements_legacy:
            cls.get_accruvia_weight = cls.get_routellect_weight

        update_abstractmethods(cls)

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
            remote: Recommendation from a hosted Routellect service
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
    def get_routellect_weight(self, population_type: str) -> float:
        """Get current weight for hosted Routellect recommendations.

        Returns:
            Weight between 0.0 and 1.0
        """
        raise NotImplementedError

    def get_accruvia_weight(self, population_type: str) -> float:
        """Backward-compatible alias for old extracted implementations."""
        return self.get_routellect_weight(population_type)


# Backward-compatible alias for extracted code that still uses the old name.
AccruviaServiceProtocol = RoutellectServiceProtocol


# Re-export for convenience
__all__ = [
    "FederatedEngineProtocol",
    "Recommendation",
    "RecommendationSource",
    "RoutellectServiceProtocol",
    "TelemetryProtocol",
    "TrustRegistryProtocol",
    "AccruviaServiceProtocol",
]
