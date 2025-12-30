"""Embedding daemon server.

Persistent server that keeps BGE-M3 and reranker models loaded in GPU memory.
Communicates via Unix socket for fast local IPC.
"""

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    MESSAGE_DELIMITER,
    SOCKET_FILENAME,
    DaemonRequest,
    DaemonResponse,
    HealthInfo,
    RequestType,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchSizes:
    """Calculated batch sizes based on GPU memory."""

    embed: int
    rerank: int


def calculate_batch_sizes() -> BatchSizes:
    """Dynamically calculate batch sizes based on available VRAM."""
    try:
        from ..rag.embeddings import get_gpu_memory
    except ImportError:
        return BatchSizes(embed=4, rerank=2)

    gpu_mem = get_gpu_memory()
    if gpu_mem is None:
        return BatchSizes(embed=4, rerank=2)  # CPU fallback

    MODEL_OVERHEAD_GB = 3.5  # BGE-M3 (~2.2GB) + Reranker (~1GB) + buffers
    SAFETY_MARGIN_GB = 1.0  # Leave headroom for other processes
    PER_EMBED_BATCH_GB = 0.4  # ~400MB per batch item for long texts
    PER_RERANK_BATCH_GB = 0.6  # Reranker needs more memory per pair

    available = gpu_mem.free - MODEL_OVERHEAD_GB - SAFETY_MARGIN_GB

    return BatchSizes(
        embed=max(2, min(32, int(available / PER_EMBED_BATCH_GB))),
        rerank=max(2, min(16, int(available / PER_RERANK_BATCH_GB))),
    )


class IdleTimer:
    """Tracks idle time for auto-shutdown."""

    def __init__(self, timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS) -> None:
        self.last_activity = time.time()
        self.timeout = timeout_seconds
        self.enabled = timeout_seconds > 0

    def touch(self) -> None:
        """Reset the idle timer."""
        self.last_activity = time.time()

    def is_expired(self) -> bool:
        """Check if idle timeout has elapsed."""
        if not self.enabled:
            return False
        return time.time() - self.last_activity > self.timeout

    def seconds_remaining(self) -> float:
        """Get seconds until timeout."""
        if not self.enabled:
            return float("inf")
        return max(0, self.timeout - (time.time() - self.last_activity))


class EmbeddingDaemon:
    """Persistent embedding server with model caching and request queue."""

    def __init__(
        self,
        socket_path: Path,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
        output_dir: Path | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.output_dir = output_dir or Path.cwd() / "output" / "queries"
        self.idle_timer = IdleTimer(idle_timeout)

        # Models (loaded lazily)
        self._embedding_model: Any = None
        self._reranker: Any = None
        self._batch_sizes: BatchSizes | None = None

        # State
        self.start_time = time.time()
        self.requests_processed = 0
        self.current_job: str | None = None
        self._shutdown_requested = False
        self._server: asyncio.Server | None = None

        # Request queue for sequential processing
        self._request_queue: asyncio.Queue[tuple[DaemonRequest, Callable[[DaemonResponse], None]]] = asyncio.Queue()

    @property
    def embedding_model(self) -> Any:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            logger.info("Loading BGE-M3 embedding model...")
            from ..rag.embeddings import EmbeddingModel

            # use_daemon=False: Server IS the daemon, don't recursively connect
            self._embedding_model = EmbeddingModel(use_daemon=False)
            logger.info("Embedding model loaded")
        return self._embedding_model

    @property
    def reranker(self) -> Any:
        """Lazy-load reranker model."""
        if self._reranker is None:
            logger.info("Loading BGE reranker model...")
            from ..rag.embeddings import Reranker

            # use_daemon=False: Server IS the daemon, don't recursively connect
            self._reranker = Reranker(use_daemon=False)
            logger.info("Reranker model loaded")
        return self._reranker

    @property
    def batch_sizes(self) -> BatchSizes:
        """Get calculated batch sizes."""
        if self._batch_sizes is None:
            self._batch_sizes = calculate_batch_sizes()
            logger.info(
                f"Batch sizes calculated: embed={self._batch_sizes.embed}, " f"rerank={self._batch_sizes.rerank}"
            )
        return self._batch_sizes

    def get_health_info(self) -> HealthInfo:
        """Get current daemon health status."""
        try:
            from ..rag.embeddings import get_gpu_memory
        except ImportError:
            gpu_mem = None
        else:
            gpu_mem = get_gpu_memory()

        return HealthInfo(
            uptime_seconds=time.time() - self.start_time,
            queue_depth=self._request_queue.qsize(),
            current_job=self.current_job,
            gpu_memory_free_gb=gpu_mem.free if gpu_mem else None,
            gpu_memory_total_gb=gpu_mem.total if gpu_mem else None,
            gpu_device_name=gpu_mem.device_name if gpu_mem else None,
            embed_batch_size=self.batch_sizes.embed,
            rerank_batch_size=self.batch_sizes.rerank,
            requests_processed=self.requests_processed,
        )

    async def handle_request(self, request: DaemonRequest) -> DaemonResponse:
        """Process a single request."""
        self.idle_timer.touch()

        try:
            if request.type == RequestType.HEALTH:
                return DaemonResponse.ok(self.get_health_info().to_dict(), request.request_id)

            elif request.type == RequestType.SHUTDOWN:
                self._shutdown_requested = True
                return DaemonResponse.ok({"message": "Shutdown initiated"}, request.request_id)

            elif request.type == RequestType.EMBED_QUERY:
                query = request.payload.get("query", "")
                if not query:
                    return DaemonResponse.create_error("Missing 'query' in payload", request.request_id)

                self.current_job = f"embed_query: {query[:50]}..."
                embedding = self.embedding_model.encode_query(query)
                self.current_job = None

                return DaemonResponse.ok({"embedding": embedding.tolist()}, request.request_id)

            elif request.type == RequestType.EMBED_BATCH:
                texts = request.payload.get("texts", [])
                if not texts:
                    return DaemonResponse.create_error("Missing 'texts' in payload", request.request_id)

                self.current_job = f"embed_batch: {len(texts)} texts"
                embeddings = self.embedding_model.encode(texts)
                self.current_job = None

                return DaemonResponse.ok({"embeddings": embeddings.tolist()}, request.request_id)

            elif request.type == RequestType.RERANK:
                query = request.payload.get("query", "")
                texts = request.payload.get("texts", [])
                top_k = request.payload.get("top_k")

                if not query or not texts:
                    return DaemonResponse.create_error("Missing 'query' or 'texts' in payload", request.request_id)

                self.current_job = f"rerank: {len(texts)} candidates"
                results = self.reranker.rerank(query, texts, top_k=top_k)
                self.current_job = None

                return DaemonResponse.ok(
                    {
                        "texts": [r.text for r in results],
                        "scores": [r.score for r in results],
                        "indices": [r.original_rank for r in results],
                    },
                    request.request_id,
                )

            elif request.type == RequestType.SEARCH:
                # Full search pipeline - requires more context
                # This will be implemented when integrating with Searcher
                return DaemonResponse.create_error("SEARCH not yet implemented in daemon", request.request_id)

            else:
                return DaemonResponse.create_error(f"Unknown request type: {request.type}", request.request_id)

        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            self.current_job = None
            return DaemonResponse.create_error(str(e), request.request_id)
        finally:
            self.requests_processed += 1

    async def _process_queue(self) -> None:
        """Process requests from the queue sequentially."""
        while not self._shutdown_requested:
            try:
                # Wait for next request with timeout for idle check
                try:
                    request, respond = await asyncio.wait_for(self._request_queue.get(), timeout=60.0)
                except TimeoutError:
                    # Check idle timeout
                    if self.idle_timer.is_expired():
                        logger.info("Idle timeout reached, shutting down")
                        self._shutdown_requested = True
                    continue

                # Process the request
                response = await self.handle_request(request)

                # Handle non-blocking requests (save to file)
                if not request.blocking and response.status.value == "ok":
                    await self._save_result(request, response)
                    respond(DaemonResponse.queued(str(self._get_output_path(request)), request.request_id))
                else:
                    respond(response)

                self._request_queue.task_done()

            except Exception as e:
                logger.exception(f"Queue processor error: {e}")

    def _get_output_path(self, request: DaemonRequest) -> Path:
        """Get output path for non-blocking request result."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H%M%S")
        query_slug = request.payload.get("query", "unknown")[:30]
        query_slug = "".join(c if c.isalnum() else "_" for c in query_slug)
        return self.output_dir / f"{timestamp}_{query_slug}.json"

    async def _save_result(self, request: DaemonRequest, response: DaemonResponse) -> None:
        """Save result to file for non-blocking request."""
        output_path = self._get_output_path(request)
        result = {
            "request_id": request.request_id,
            "request_type": request.type.value,
            "request_payload": request.payload,
            "response_status": response.status.value,
            "response_data": response.data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        output_path.write_text(json.dumps(result, indent=2))
        logger.info(f"Result saved to {output_path}")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single client connection."""
        try:
            # Read message until delimiter
            data = await reader.readuntil(MESSAGE_DELIMITER)
            message = data[: -len(MESSAGE_DELIMITER)].decode("utf-8")

            request = DaemonRequest.from_json(message)
            logger.debug(f"Received request: {request.type.value}")

            # Create a future to receive the response
            response_future: asyncio.Future[DaemonResponse] = asyncio.Future()

            def respond(resp: DaemonResponse) -> None:
                if not response_future.done():
                    response_future.set_result(resp)

            # Queue the request
            await self._request_queue.put((request, respond))

            # Wait for response
            response = await response_future

            # Send response
            response_data = response.to_json().encode("utf-8") + MESSAGE_DELIMITER
            writer.write(response_data)
            await writer.drain()

        except asyncio.IncompleteReadError:
            logger.debug("Client disconnected before sending complete message")
        except Exception as e:
            logger.exception(f"Error handling client: {e}")
            try:
                error_response = DaemonResponse.create_error(str(e))
                writer.write(error_response.to_json().encode("utf-8") + MESSAGE_DELIMITER)
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self) -> None:
        """Start the daemon server."""
        # Remove stale socket if exists
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Pre-load models
        logger.info("Pre-loading models...")
        _ = self.embedding_model  # Trigger lazy load
        _ = self.reranker  # Trigger lazy load
        _ = self.batch_sizes  # Calculate batch sizes
        logger.info("Models loaded, starting server...")

        # Start queue processor
        queue_task = asyncio.create_task(self._process_queue())

        # Start Unix socket server
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))

        # Set socket permissions (owner read/write only)
        os.chmod(self.socket_path, 0o600)

        logger.info(f"Daemon listening on {self.socket_path}")

        try:
            async with self._server:
                # Run until shutdown requested
                while not self._shutdown_requested:
                    await asyncio.sleep(1)
        finally:
            queue_task.cancel()
            try:
                await queue_task
            except asyncio.CancelledError:
                pass

            # Cleanup
            if self.socket_path.exists():
                self.socket_path.unlink()
            logger.info("Daemon stopped")

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_requested = True


def run_daemon(
    socket_path: Path | None = None,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
    output_dir: Path | None = None,
) -> None:
    """Run the embedding daemon (blocking)."""
    if socket_path is None:
        socket_path = Path.cwd() / SOCKET_FILENAME

    daemon = EmbeddingDaemon(
        socket_path=socket_path,
        idle_timeout=idle_timeout,
        output_dir=output_dir,
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        daemon.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the daemon
    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    finally:
        # Ensure socket is cleaned up
        if socket_path.exists():
            socket_path.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Read environment variables set by spawn_daemon()
    socket_path_str = os.environ.get("PITCH_DAEMON_SOCKET")
    socket_path = Path(socket_path_str) if socket_path_str else None

    idle_timeout_str = os.environ.get("PITCH_DAEMON_IDLE_TIMEOUT")
    idle_timeout = int(idle_timeout_str) if idle_timeout_str else DEFAULT_IDLE_TIMEOUT_SECONDS

    run_daemon(socket_path=socket_path, idle_timeout=idle_timeout)
