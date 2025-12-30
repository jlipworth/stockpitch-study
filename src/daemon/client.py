"""Client for communicating with the embedding daemon.

Handles connection management, auto-spawn, and fallback to local processing.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import (
    MESSAGE_DELIMITER,
    MODEL_LOAD_TIMEOUT_SECONDS,
    SOCKET_CONNECT_TIMEOUT_SECONDS,
    SOCKET_FILENAME,
    DaemonRequest,
    DaemonResponse,
    HealthInfo,
    RequestType,
    ResponseStatus,
)

logger = logging.getLogger(__name__)


class DaemonConnectionError(Exception):
    """Failed to connect to daemon."""

    pass


class DaemonClient:
    """Client for communicating with the embedding daemon."""

    def __init__(self, socket_path: Path | None = None) -> None:
        self.socket_path = socket_path or Path.cwd() / SOCKET_FILENAME
        self._connected = False

    def is_daemon_running(self) -> bool:
        """Check if daemon socket exists and is responsive."""
        if not self.socket_path.exists():
            return False

        # Try to connect and send health check
        try:
            response = self._send_request_sync(DaemonRequest(type=RequestType.HEALTH), timeout=2.0)
            return response.status == ResponseStatus.OK
        except Exception:
            return False

    def _send_request_sync(self, request: DaemonRequest, timeout: float = 30.0) -> DaemonResponse:
        """Send request synchronously (blocking)."""
        return asyncio.run(self._send_request_async(request, timeout))

    async def _send_request_async(self, request: DaemonRequest, timeout: float = 30.0) -> DaemonResponse:
        """Send request asynchronously."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.socket_path)),
                timeout=SOCKET_CONNECT_TIMEOUT_SECONDS,
            )
        except (FileNotFoundError, ConnectionRefusedError) as e:
            raise DaemonConnectionError(f"Cannot connect to daemon: {e}") from e
        except TimeoutError as e:
            raise DaemonConnectionError("Connection timeout") from e

        try:
            # Send request
            message = request.to_json().encode("utf-8") + MESSAGE_DELIMITER
            writer.write(message)
            await writer.drain()

            # Read response
            data = await asyncio.wait_for(reader.readuntil(MESSAGE_DELIMITER), timeout=timeout)
            response_json = data[: -len(MESSAGE_DELIMITER)].decode("utf-8")
            return DaemonResponse.from_json(response_json)

        finally:
            writer.close()
            await writer.wait_closed()

    def send_request(self, request: DaemonRequest, timeout: float = 30.0) -> DaemonResponse:
        """Send request to daemon (blocking)."""
        return self._send_request_sync(request, timeout)

    def health(self) -> HealthInfo:
        """Get daemon health status."""
        response = self.send_request(DaemonRequest(type=RequestType.HEALTH))
        if response.status != ResponseStatus.OK:
            raise DaemonConnectionError(response.error or "Health check failed")
        return HealthInfo.from_dict(response.data)

    def shutdown(self) -> bool:
        """Request daemon shutdown."""
        try:
            response = self.send_request(DaemonRequest(type=RequestType.SHUTDOWN))
            return response.status == ResponseStatus.OK
        except Exception:
            return False

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        response = self.send_request(
            DaemonRequest(
                type=RequestType.EMBED_QUERY,
                payload={"query": query},
            )
        )
        if response.status != ResponseStatus.OK:
            raise DaemonConnectionError(response.error or "Embed query failed")
        return np.array(response.data["embedding"])

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        response = self.send_request(
            DaemonRequest(
                type=RequestType.EMBED_BATCH,
                payload={"texts": texts},
            ),
            timeout=300.0,  # 5 min for large batches
        )
        if response.status != ResponseStatus.OK:
            raise DaemonConnectionError(response.error or "Embed batch failed")
        return np.array(response.data["embeddings"])

    def rerank(
        self, query: str, texts: list[str], top_k: int | None = None
    ) -> tuple[list[str], list[float], list[int]]:
        """Rerank texts by relevance to query.

        Returns:
            Tuple of (reranked_texts, scores, original_indices)
        """
        response = self.send_request(
            DaemonRequest(
                type=RequestType.RERANK,
                payload={"query": query, "texts": texts, "top_k": top_k},
            ),
            timeout=60.0,
        )
        if response.status != ResponseStatus.OK:
            raise DaemonConnectionError(response.error or "Rerank failed")
        return (
            response.data["texts"],
            response.data["scores"],
            response.data["indices"],
        )


def spawn_daemon(
    socket_path: Path | None = None,
    background: bool = True,
    idle_timeout: int = 900,
    wait_for_ready: bool = True,
) -> subprocess.Popen[bytes] | None:
    """Spawn a new daemon process.

    Args:
        socket_path: Path to socket file (default: .pitch-daemon.sock in cwd)
        background: If True, daemon runs in background (detached)
        idle_timeout: Idle timeout in seconds (0 to disable)
        wait_for_ready: If True, wait for daemon to be ready before returning

    Returns:
        Popen object if not waiting for ready, None otherwise
    """
    if socket_path is None:
        socket_path = Path.cwd() / SOCKET_FILENAME

    # Build command
    cmd = [sys.executable, "-m", "src.daemon.server"]

    # Add arguments
    env = os.environ.copy()
    env["PITCH_DAEMON_SOCKET"] = str(socket_path)
    env["PITCH_DAEMON_IDLE_TIMEOUT"] = str(idle_timeout)

    if background:
        # Detach from parent process
        kwargs: dict[str, Any] = {
            "start_new_session": True,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
        }
    else:
        kwargs = {"env": env}

    logger.info(f"Spawning daemon (background={background})...")
    proc = subprocess.Popen(cmd, **kwargs)

    if wait_for_ready and background:
        # Wait for socket to appear
        logger.info("Waiting for daemon to load models...")
        start = time.time()
        while time.time() - start < MODEL_LOAD_TIMEOUT_SECONDS:
            if socket_path.exists():
                # Try to connect
                try:
                    client = DaemonClient(socket_path)
                    if client.is_daemon_running():
                        logger.info("Daemon ready")
                        return None
                except Exception:
                    pass
            time.sleep(0.5)

        logger.warning("Daemon did not become ready in time")
        return proc

    return proc


def get_daemon_client(auto_spawn: bool = True, socket_path: Path | None = None) -> DaemonClient | None:
    """Get a client connected to the daemon.

    Args:
        auto_spawn: If True, spawn daemon if not running
        socket_path: Path to socket file

    Returns:
        DaemonClient if connected, None if unavailable
    """
    if socket_path is None:
        socket_path = Path.cwd() / SOCKET_FILENAME

    client = DaemonClient(socket_path)

    # Check if daemon is running
    if client.is_daemon_running():
        return client

    # Remove stale socket
    if socket_path.exists():
        logger.debug("Removing stale socket")
        socket_path.unlink()

    if not auto_spawn:
        return None

    # Spawn daemon
    spawn_daemon(socket_path, background=True, wait_for_ready=True)

    # Check again
    if client.is_daemon_running():
        return client

    logger.warning("Failed to spawn daemon, will use local models")
    return None


def force_stop_daemon(socket_path: Path | None = None) -> bool:
    """Force stop daemon by removing socket and killing process.

    Returns:
        True if daemon was stopped, False if not running
    """
    if socket_path is None:
        socket_path = Path.cwd() / SOCKET_FILENAME

    if not socket_path.exists():
        return False

    # Try graceful shutdown first
    try:
        client = DaemonClient(socket_path)
        client.shutdown()
        time.sleep(1)
    except Exception:
        pass

    # Force remove socket
    if socket_path.exists():
        socket_path.unlink()

    return True
