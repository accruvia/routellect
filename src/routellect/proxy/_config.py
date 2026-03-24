"""Proxy configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from routellect.protocols import ModelSelectorProtocol

_TRUTHY = {"1", "true", "yes"}


@dataclass
class ProxyConfig:
    """Configuration for the routellect LLM proxy."""

    host: str = "127.0.0.1"
    port: int = 11411
    auth_token: str | None = None
    log_bodies: bool = False
    selector: ModelSelectorProtocol | None = None

    @classmethod
    def from_env(cls) -> ProxyConfig:
        """Build config from ``ROUTELLECT_PROXY_*`` environment variables."""
        return cls(
            host=os.environ.get("ROUTELLECT_PROXY_HOST", "127.0.0.1"),
            port=int(os.environ.get("ROUTELLECT_PROXY_PORT", "11411")),
            auth_token=os.environ.get("ROUTELLECT_PROXY_TOKEN"),
            log_bodies=os.environ.get("ROUTELLECT_LOG_BODIES", "").lower() in _TRUTHY,
        )
