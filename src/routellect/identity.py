"""Client identity management for anonymous telemetry.

This module implements the deferred identity model:
- Phase 1-2: Random client UUID (no PII, no GDPR)
- Phase 3+: Optional link UUID to account for credit claiming

UUID is generated on first install and stored in ~/.accruvia/client_id.
Telemetry includes proof-of-execution for Sybil resistance.
"""

import hashlib
import platform
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class ExecutionProof:
    """Proof that tests were actually executed (anti-Sybil).

    This data is hard to fake without actually running tests:
    - test_output_hash: SHA256 of full pytest output
    - execution_time_ms: Realistic timing (too fast = suspicious)
    - python_version: Environment fingerprint
    - dependency_versions: Package versions used
    """

    test_output_hash: str
    execution_time_ms: int
    python_version: str
    dependency_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_output_hash": self.test_output_hash,
            "execution_time_ms": self.execution_time_ms,
            "python_version": self.python_version,
            "dependency_versions": self.dependency_versions,
        }


@dataclass
class TelemetryPayload:
    """Anonymous telemetry payload with execution proof.

    Contains:
    - client_uuid: Random UUID (not PII)
    - outcome: Test success/failure
    - proof: Evidence of actual execution
    - task_fingerprint: Task characteristics (for learning)
    """

    client_uuid: str
    outcome: bool
    proof: ExecutionProof
    task_fingerprint: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_used: str = ""
    latency_ms: int = 0
    cost: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "client_uuid": self.client_uuid,
            "outcome": self.outcome,
            "proof": self.proof.to_dict(),
            "task_fingerprint": self.task_fingerprint,
            "timestamp": self.timestamp,
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
        }


def get_accruvia_dir() -> Path:
    """Get the Accruvia config directory (~/.accruvia)."""
    accruvia_dir = Path.home() / ".accruvia"
    accruvia_dir.mkdir(parents=True, exist_ok=True)
    return accruvia_dir


def get_client_id_path() -> Path:
    """Get path to client ID file."""
    return get_accruvia_dir() / "client_id"


def get_or_create_client_uuid() -> str:
    """Generate or load the client UUID.

    UUID is stored in ~/.accruvia/client_id and persists across sessions.
    This is NOT personal data - it's a random identifier that cannot
    be traced back to any individual without additional linkage.

    Returns:
        The client UUID as a string.
    """
    client_id_path = get_client_id_path()

    if client_id_path.exists():
        try:
            stored_uuid = client_id_path.read_text().strip()
            # Validate it's a valid UUID
            uuid.UUID(stored_uuid)
            return stored_uuid
        except (ValueError, OSError):
            # Invalid or corrupted, regenerate
            pass

    # Generate new UUID
    new_uuid = str(uuid.uuid4())
    try:
        client_id_path.write_text(new_uuid)
    except OSError:
        # Can't persist, but still return the UUID for this session
        pass

    return new_uuid


def hash_test_output(output: str) -> str:
    """Hash pytest output for proof-of-execution.

    Args:
        output: Full pytest stdout/stderr output.

    Returns:
        SHA256 hash of the output.
    """
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def get_python_version() -> str:
    """Get current Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_platform_info() -> str:
    """Get platform identifier (OS + architecture)."""
    return f"{platform.system()}-{platform.machine()}"


def build_execution_proof(
    test_output: str,
    execution_time_ms: int,
    dependency_versions: dict[str, str] | None = None,
) -> ExecutionProof:
    """Build proof-of-execution for anti-Sybil protection.

    Args:
        test_output: Full pytest stdout/stderr output.
        execution_time_ms: How long the tests took to run.
        dependency_versions: Optional dict of package versions.

    Returns:
        ExecutionProof containing evidence of actual execution.
    """
    return ExecutionProof(
        test_output_hash=hash_test_output(test_output),
        execution_time_ms=execution_time_ms,
        python_version=get_python_version(),
        dependency_versions=dependency_versions or {},
    )


def build_telemetry_payload(
    outcome: bool,
    proof: ExecutionProof,
    task_fingerprint: dict | None = None,
    model_used: str = "",
    latency_ms: int = 0,
    cost: float = 0.0,
) -> TelemetryPayload:
    """Build anonymous telemetry payload for server.

    Args:
        outcome: Whether the task succeeded.
        proof: Proof of actual execution.
        task_fingerprint: Task characteristics for learning.
        model_used: Which model was used.
        latency_ms: API call latency.
        cost: Cost incurred.

    Returns:
        TelemetryPayload ready to send to server.
    """
    return TelemetryPayload(
        client_uuid=get_or_create_client_uuid(),
        outcome=outcome,
        proof=proof,
        task_fingerprint=task_fingerprint or {},
        model_used=model_used,
        latency_ms=latency_ms,
        cost=cost,
    )


class CreditClaimManager:
    """Manages credit claiming for Phase 3+ monetization.

    In Phase 1-2, contributors earn credits against their UUID.
    In Phase 3, they can optionally claim credits by linking
    their UUID to a GitHub account via OAuth.
    """

    def __init__(self) -> None:
        self.claim_file = get_accruvia_dir() / "claimed_account"

    def is_claimed(self) -> bool:
        """Check if credits have been claimed to an account."""
        return self.claim_file.exists()

    def get_linked_account(self) -> str | None:
        """Get the linked account if claimed."""
        if not self.is_claimed():
            return None
        try:
            return self.claim_file.read_text().strip()
        except OSError:
            return None

    def claim_credits(self, github_username: str) -> dict:
        """Link UUID to GitHub account for credit claiming.

        This triggers the OAuth flow in Phase 3.
        For now, we just store the intended linkage locally.

        Args:
            github_username: GitHub username to link to.

        Returns:
            Status dict with claim info.
        """
        client_uuid = get_or_create_client_uuid()

        # Store the linkage request locally
        # In Phase 3, this will trigger actual OAuth
        try:
            self.claim_file.write_text(github_username)
        except OSError as e:
            return {
                "success": False,
                "error": f"Failed to save claim: {e}",
                "client_uuid": client_uuid,
            }

        return {
            "success": True,
            "client_uuid": client_uuid,
            "github_username": github_username,
            "message": "Credit claim registered. Will be processed in Phase 3.",
        }


# Module-level convenience functions
_claim_manager: CreditClaimManager | None = None


def get_claim_manager() -> CreditClaimManager:
    """Get the singleton CreditClaimManager."""
    global _claim_manager
    if _claim_manager is None:
        _claim_manager = CreditClaimManager()
    return _claim_manager


def claim_credits(github_username: str) -> dict:
    """Convenience function to claim credits."""
    return get_claim_manager().claim_credits(github_username)


def is_credits_claimed() -> bool:
    """Check if credits have been claimed."""
    return get_claim_manager().is_claimed()
