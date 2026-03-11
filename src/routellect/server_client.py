"""Optional hosted-service client scaffolding for Routellect.

The canonical surface in this module is organized around hosted-service
integration and local scaffold helpers. Compatibility aliases remain for
older extracted callers that still import server-centric names.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from routellect.identity import build_execution_proof
from routellect.protocols import (
    Recommendation,
    RecommendationSource,
)

logger = logging.getLogger(__name__)

# Default hosted-service URL used by the local scaffold.
DEFAULT_HOSTED_SERVICE_URL = "http://localhost:8000"

# Local scaffold URL for development and extracted integrations.
LOCAL_SCAFFOLD_URL = DEFAULT_HOSTED_SERVICE_URL

# Request timeout in seconds
DEFAULT_TIMEOUT = 30.0

# Number of retries for failed requests
DEFAULT_RETRIES = 3


@dataclass(init=False)
class HostedServiceClientConfig:
    """Configuration for the optional hosted-service client."""

    service_url: str
    timeout: float
    retries: int
    verify_ssl: bool
    api_key: str | None

    def __init__(
        self,
        service_url: str = DEFAULT_HOSTED_SERVICE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        verify_ssl: bool = True,
        api_key: str | None = None,
        *,
        server_url: str | None = None,
    ) -> None:
        if server_url is not None:
            service_url = server_url

        self.service_url = service_url
        self.timeout = timeout
        self.retries = retries
        self.verify_ssl = verify_ssl
        self.api_key = api_key

    @property
    def server_url(self) -> str:
        """Backward-compatible alias for older callers."""
        return self.service_url

    @server_url.setter
    def server_url(self, value: str) -> None:
        self.service_url = value


@dataclass
class RouteResult:
    """Result from /route endpoint."""

    task_id: str
    recommended_model: str
    confidence: float
    source: str

    def to_recommendation(self) -> Recommendation:
        """Convert to Recommendation."""
        source = RecommendationSource.ROUTELLECT
        if self.source == "local":
            source = RecommendationSource.LOCAL
        elif self.source == "blended":
            source = RecommendationSource.BLENDED
        elif self.source in {"accruvia", "routellect"}:
            source = RecommendationSource.ROUTELLECT

        return Recommendation(
            model_id=self.recommended_model,
            confidence=self.confidence,
            source=source,
        )


@dataclass
class SettleResult:
    """Result from /settle endpoint."""

    task_id: str
    settled: bool
    proof_valid: bool
    proof_confidence: float
    message: str


class HostedServiceClient:
    """HTTP client implementing the hosted routing service protocol.

    Connects to an optional hosted service for model recommendations and
    outcome reporting. Falls back gracefully when the service is unavailable.
    """

    def __init__(self, config: HostedServiceClientConfig | None = None) -> None:
        self.config = config or HostedServiceClientConfig()
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._pending_tasks: dict[str, dict] = {}

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "routellect/0.1.0",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.service_url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                headers=self._get_headers(),
            )
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.config.service_url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                headers=self._get_headers(),
            )
        return self._sync_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    # RoutellectServiceProtocol implementation
    def get_recommendation(
        self,
        task_fingerprint: dict,
        local_recommendation: Recommendation | None = None,
    ) -> Recommendation:
        """Get a model recommendation from the server.

        Synchronous version for compatibility with RoutellectServiceProtocol.
        """
        from routellect.identity import get_or_create_client_uuid

        client = self._get_sync_client()

        payload = {
            "client_uuid": get_or_create_client_uuid(),
            "task_fingerprint": task_fingerprint,
        }

        if local_recommendation:
            payload["local_recommendation"] = local_recommendation.model_id
            payload["local_confidence"] = local_recommendation.confidence

        try:
            response = client.post("/route", json=payload)
            response.raise_for_status()
            data = response.json()

            # Store task for later settlement
            self._pending_tasks[data["task_id"]] = {
                "task_fingerprint": task_fingerprint,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            return Recommendation(
                model_id=data["recommended_model"],
                confidence=data["confidence"],
                source=RecommendationSource.ROUTELLECT,
                reasoning="Hosted routing recommendation",
            )

        except httpx.HTTPError as e:
            logger.warning(f"Server request failed: {e}")
            # Fall back to local recommendation or default
            if local_recommendation:
                return Recommendation(
                    model_id=local_recommendation.model_id,
                    confidence=local_recommendation.confidence * 0.8,
                    source=RecommendationSource.LOCAL,
                reasoning="Hosted routing unavailable, using local recommendation",
            )
            return Recommendation(
                model_id="claude-opus-4-5-20251101",
                confidence=0.5,
                source=RecommendationSource.LOCAL,
                reasoning="Hosted routing unavailable, using default",
            )

    def report_outcome(
        self,
        task_fingerprint: dict,
        model_id: str,
        success: bool,
        latency_ms: int,
        cost: float,
    ) -> None:
        """Report task outcome to the server.

        Synchronous version for compatibility with RoutellectServiceProtocol.
        """
        # Find the task_id for this fingerprint
        task_id = None
        for tid, task_info in self._pending_tasks.items():
            if task_info["task_fingerprint"] == task_fingerprint:
                task_id = tid
                break

        if task_id is None:
            logger.warning("No pending task found for fingerprint, skipping settlement")
            return

        # Build execution proof (simplified for sync)
        proof = {
            "test_output_hash": "0" * 64,  # Placeholder
            "execution_time_ms": latency_ms,
            "python_version": "3.11.0",
            "dependency_versions": {},
        }

        from routellect.identity import get_or_create_client_uuid

        payload = {
            "task_id": task_id,
            "client_uuid": get_or_create_client_uuid(),
            "model_used": model_id,
            "success": success,
            "latency_ms": latency_ms,
            "cost": cost,
            "proof": proof,
        }

        try:
            client = self._get_sync_client()
            response = client.post("/settle", json=payload)
            response.raise_for_status()

            # Remove from pending
            del self._pending_tasks[task_id]

        except httpx.HTTPError as e:
            logger.warning(f"Settlement failed: {e}")

    # Async versions for more flexible usage
    async def route_async(
        self,
        task_fingerprint: dict,
        local_recommendation: Recommendation | None = None,
    ) -> RouteResult:
        """Get a model recommendation (async version)."""
        from routellect.identity import get_or_create_client_uuid

        client = await self._get_async_client()

        payload = {
            "client_uuid": get_or_create_client_uuid(),
            "task_fingerprint": task_fingerprint,
        }

        if local_recommendation:
            payload["local_recommendation"] = local_recommendation.model_id
            payload["local_confidence"] = local_recommendation.confidence

        response = await client.post("/route", json=payload)
        response.raise_for_status()
        data = response.json()

        # Store task for later settlement
        self._pending_tasks[data["task_id"]] = {
            "task_fingerprint": task_fingerprint,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return RouteResult(
            task_id=data["task_id"],
            recommended_model=data["recommended_model"],
            confidence=data["confidence"],
            source=data["source"],
        )

    async def settle_async(
        self,
        task_id: str,
        model_used: str,
        success: bool,
        latency_ms: int,
        cost: float,
        test_output: str = "",
    ) -> SettleResult:
        """Settle a task with outcome data (async version)."""
        from routellect.identity import get_or_create_client_uuid

        client = await self._get_async_client()

        # Build execution proof
        proof = build_execution_proof(
            test_output=test_output,
            execution_time_ms=latency_ms,
        )

        payload = {
            "task_id": task_id,
            "client_uuid": get_or_create_client_uuid(),
            "model_used": model_used,
            "success": success,
            "latency_ms": latency_ms,
            "cost": cost,
            "proof": proof.to_dict(),
        }

        response = await client.post("/settle", json=payload)
        response.raise_for_status()
        data = response.json()

        # Remove from pending
        if task_id in self._pending_tasks:
            del self._pending_tasks[task_id]

        return SettleResult(
            task_id=data["task_id"],
            settled=data["settled"],
            proof_valid=data["proof_valid"],
            proof_confidence=data["proof_confidence"],
            message=data["message"],
        )

    async def get_metrics_async(self) -> dict:
        """Get server metrics (async)."""
        client = await self._get_async_client()
        response = await client.get("/metrics")
        response.raise_for_status()
        return response.json()

    async def health_check_async(self) -> dict:
        """Check server health (async)."""
        client = await self._get_async_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    def is_available(self) -> bool:
        """Check if server is available (sync)."""
        try:
            client = self._get_sync_client()
            response = client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False


def create_hosted_service_client(
    service_url: str = DEFAULT_HOSTED_SERVICE_URL,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> HostedServiceClient:
    """Create an optional hosted routing client.

    Args:
        service_url: Hosted service URL.
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.

    Returns:
        Configured HostedServiceClient.
    """
    config = HostedServiceClientConfig(
        service_url=service_url,
        api_key=api_key,
        timeout=timeout,
    )
    return HostedServiceClient(config)


def create_local_scaffold_client() -> HostedServiceClient:
    """Create a client for the local hosted-service scaffold."""
    return create_hosted_service_client(service_url=LOCAL_SCAFFOLD_URL, timeout=10.0)


def create_client(
    server_url: str = DEFAULT_HOSTED_SERVICE_URL,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> HostedServiceClient:
    """Backward-compatible alias for older server-centric helpers."""
    return create_hosted_service_client(
        service_url=server_url,
        api_key=api_key,
        timeout=timeout,
    )


def create_scaffold_client() -> HostedServiceClient:
    """Backward-compatible alias for the local scaffold helper."""
    return create_local_scaffold_client()


def create_dev_client() -> HostedServiceClient:
    """Backward-compatible alias for the old helper name."""
    return create_scaffold_client()


# Canonical names exported by this module.
HostedRouterClient = HostedServiceClient
ServerClientConfig = HostedServiceClientConfig
DEFAULT_SERVER_URL = DEFAULT_HOSTED_SERVICE_URL
DEV_SERVER_URL = LOCAL_SCAFFOLD_URL

# Backward-compatible aliases for the extracted codebase.
AccruviaServerClient = HostedServiceClient

__all__ = [
    "DEFAULT_HOSTED_SERVICE_URL",
    "LOCAL_SCAFFOLD_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "HostedServiceClientConfig",
    "HostedServiceClient",
    "RouteResult",
    "SettleResult",
    "create_hosted_service_client",
    "create_local_scaffold_client",
    "DEFAULT_SERVER_URL",
    "DEV_SERVER_URL",
    "ServerClientConfig",
    "HostedRouterClient",
    "AccruviaServerClient",
    "create_client",
    "create_scaffold_client",
    "create_dev_client",
]
