"""Protocol definitions for embedding daemon communication.

Defines request/response types for Unix socket IPC between CLI and daemon.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestType(str, Enum):
    """Types of requests the daemon can handle."""

    EMBED_QUERY = "embed_query"  # Single query embedding
    EMBED_BATCH = "embed_batch"  # Multiple texts embedding
    RERANK = "rerank"  # Cross-encoder reranking
    SEARCH = "search"  # Full search pipeline
    HEALTH = "health"  # Health check
    SHUTDOWN = "shutdown"  # Graceful shutdown


class ResponseStatus(str, Enum):
    """Response status codes."""

    OK = "ok"
    ERROR = "error"
    QUEUED = "queued"  # For non-blocking requests


@dataclass
class DaemonRequest:
    """Request sent from CLI to daemon."""

    type: RequestType
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    blocking: bool = True  # If False, daemon writes result to file

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = f"{time.time_ns()}"

    def to_json(self) -> str:
        """Serialize to JSON for socket transmission."""
        return json.dumps(
            {
                "type": self.type.value,
                "payload": self.payload,
                "request_id": self.request_id,
                "blocking": self.blocking,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "DaemonRequest":
        """Deserialize from JSON."""
        obj = json.loads(data)
        return cls(
            type=RequestType(obj["type"]),
            payload=obj.get("payload", {}),
            request_id=obj.get("request_id", ""),
            blocking=obj.get("blocking", True),
        )


@dataclass
class DaemonResponse:
    """Response sent from daemon to CLI."""

    status: ResponseStatus
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    request_id: str = ""

    def to_json(self) -> str:
        """Serialize to JSON for socket transmission."""
        return json.dumps(
            {
                "status": self.status.value,
                "data": self.data,
                "error": self.error,
                "request_id": self.request_id,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "DaemonResponse":
        """Deserialize from JSON."""
        obj = json.loads(data)
        return cls(
            status=ResponseStatus(obj["status"]),
            data=obj.get("data", {}),
            error=obj.get("error"),
            request_id=obj.get("request_id", ""),
        )

    @classmethod
    def ok(cls, data: dict[str, Any], request_id: str = "") -> "DaemonResponse":
        """Create a successful response."""
        return cls(status=ResponseStatus.OK, data=data, request_id=request_id)

    @classmethod
    def create_error(cls, message: str, request_id: str = "") -> "DaemonResponse":
        """Create an error response."""
        return cls(status=ResponseStatus.ERROR, error=message, request_id=request_id)

    @classmethod
    def queued(cls, output_path: str, request_id: str = "") -> "DaemonResponse":
        """Create a queued response for non-blocking requests."""
        return cls(
            status=ResponseStatus.QUEUED,
            data={"output_path": output_path},
            request_id=request_id,
        )


@dataclass
class HealthInfo:
    """Health information returned by health check."""

    uptime_seconds: float
    queue_depth: int
    current_job: str | None
    gpu_memory_free_gb: float | None
    gpu_memory_total_gb: float | None
    gpu_device_name: str | None
    embed_batch_size: int
    rerank_batch_size: int
    requests_processed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "queue_depth": self.queue_depth,
            "current_job": self.current_job,
            "gpu_memory_free_gb": self.gpu_memory_free_gb,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "gpu_device_name": self.gpu_device_name,
            "embed_batch_size": self.embed_batch_size,
            "rerank_batch_size": self.rerank_batch_size,
            "requests_processed": self.requests_processed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthInfo":
        """Create from dictionary."""
        return cls(
            uptime_seconds=data["uptime_seconds"],
            queue_depth=data["queue_depth"],
            current_job=data.get("current_job"),
            gpu_memory_free_gb=data.get("gpu_memory_free_gb"),
            gpu_memory_total_gb=data.get("gpu_memory_total_gb"),
            gpu_device_name=data.get("gpu_device_name"),
            embed_batch_size=data["embed_batch_size"],
            rerank_batch_size=data["rerank_batch_size"],
            requests_processed=data["requests_processed"],
        )


# Socket protocol constants
SOCKET_FILENAME = ".pitch-daemon.sock"
MESSAGE_DELIMITER = b"\n\x00\n"  # Unlikely to appear in JSON
MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB max (for large batch embeddings)

# Timeouts
DEFAULT_IDLE_TIMEOUT_SECONDS = 900  # 15 minutes
SOCKET_CONNECT_TIMEOUT_SECONDS = 5
MODEL_LOAD_TIMEOUT_SECONDS = 60  # Wait up to 60s for models to load on spawn
