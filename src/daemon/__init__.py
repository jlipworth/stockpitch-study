"""Embedding daemon for persistent model caching.

Keeps BGE-M3 embeddings and reranker models loaded in GPU memory
to eliminate cold-start latency on searches.
"""

from .client import (
    DaemonClient,
    DaemonConnectionError,
    force_stop_daemon,
    get_daemon_client,
    spawn_daemon,
)
from .protocol import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    SOCKET_FILENAME,
    DaemonRequest,
    DaemonResponse,
    HealthInfo,
    RequestType,
    ResponseStatus,
)
from .server import EmbeddingDaemon, run_daemon

__all__ = [
    # Client
    "DaemonClient",
    "DaemonConnectionError",
    "get_daemon_client",
    "spawn_daemon",
    "force_stop_daemon",
    # Server
    "EmbeddingDaemon",
    "run_daemon",
    # Protocol
    "DaemonRequest",
    "DaemonResponse",
    "HealthInfo",
    "RequestType",
    "ResponseStatus",
    "SOCKET_FILENAME",
    "DEFAULT_IDLE_TIMEOUT_SECONDS",
]
