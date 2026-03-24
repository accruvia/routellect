"""EMA-based circuit breaker for provider failover.

When a provider fails (billing, auth, quota), the circuit breaker marks
it as down with an exponentially increasing backoff.  Requests skip downed
providers until the backoff expires, then a single probe is sent.  If the
probe succeeds, the provider is restored.  If it fails, the backoff doubles.

A manual re-enable resets the circuit immediately (e.g., after topping up).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("routellect.proxy")

# Initial backoff after first failure (seconds).
INITIAL_BACKOFF_S = 60
# Maximum backoff cap (seconds).
MAX_BACKOFF_S = 3600  # 1 hour
# Multiplier per consecutive failure.
BACKOFF_MULTIPLIER = 2.0


@dataclass
class _ProviderState:
    failures: int = 0
    last_failure_at: float = 0.0
    current_backoff_s: float = INITIAL_BACKOFF_S
    manually_disabled: bool = False

    @property
    def is_down(self) -> bool:
        if self.manually_disabled:
            return True
        if self.failures == 0:
            return False
        return (time.monotonic() - self.last_failure_at) < self.current_backoff_s

    @property
    def seconds_until_probe(self) -> float:
        if not self.is_down or self.manually_disabled:
            return 0.0
        remaining = self.current_backoff_s - (time.monotonic() - self.last_failure_at)
        return max(0.0, remaining)


class CircuitBreaker:
    """Tracks provider health and manages backoff."""

    def __init__(self) -> None:
        self._states: dict[str, _ProviderState] = {}

    def _get(self, provider: str) -> _ProviderState:
        if provider not in self._states:
            self._states[provider] = _ProviderState()
        return self._states[provider]

    def is_available(self, provider: str) -> bool:
        """Check if a provider is available (not in backoff)."""
        return not self._get(provider).is_down

    def record_failure(self, provider: str) -> None:
        """Record a provider failure and increase backoff."""
        state = self._get(provider)
        state.failures += 1
        state.last_failure_at = time.monotonic()
        if state.failures == 1:
            state.current_backoff_s = INITIAL_BACKOFF_S
        else:
            state.current_backoff_s = min(
                state.current_backoff_s * BACKOFF_MULTIPLIER,
                MAX_BACKOFF_S,
            )
        logger.warning(
            "Circuit breaker: %s marked down (failure #%d, backoff %.0fs)",
            provider, state.failures, state.current_backoff_s,
        )

    def record_success(self, provider: str) -> None:
        """Record a successful call — resets the circuit."""
        state = self._get(provider)
        if state.failures > 0:
            logger.info("Circuit breaker: %s recovered after %d failures", provider, state.failures)
        state.failures = 0
        state.current_backoff_s = INITIAL_BACKOFF_S
        state.manually_disabled = False

    def re_enable(self, provider: str) -> None:
        """Manually re-enable a provider (e.g., after topping up credits)."""
        state = self._get(provider)
        state.failures = 0
        state.current_backoff_s = INITIAL_BACKOFF_S
        state.manually_disabled = False
        state.last_failure_at = 0.0
        logger.info("Circuit breaker: %s manually re-enabled", provider)

    def get_down_providers(self) -> list[str]:
        """Return list of providers currently in backoff."""
        return [p for p, s in self._states.items() if s.is_down]

    def get_status(self) -> dict[str, dict]:
        """Return status of all tracked providers."""
        result = {}
        for provider, state in self._states.items():
            result[provider] = {
                "available": not state.is_down,
                "failures": state.failures,
                "backoff_s": round(state.current_backoff_s),
                "seconds_until_probe": round(state.seconds_until_probe),
            }
        return result
