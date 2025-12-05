"""Embedding pipeline for document chunks using BGE-M3.

BGE-M3 supports:
- Multi-linguality (100+ languages)
- Multi-granularity (up to 8192 tokens vs 512 for BGE-large)
- Multi-functionality (dense, sparse, and ColBERT retrieval)

We use dense embeddings via sentence-transformers for simplicity.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .parser import Chunk

# Default model - BGE-M3 (1024 dimensions, 8192 token context)
# Larger context is critical for long SEC filing sections
DEFAULT_MODEL = "BAAI/bge-m3"

# Default reranker - BGE Reranker v2 M3 (same family, ~1GB)
# Cross-encoder that scores (query, passage) pairs for better precision
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

# Device-specific batch sizes (BGE-M3 is ~2.2GB)
# RTX 4090: 24GB VRAM, but need headroom for multiple files
# M3 Pro: 18-36GB unified memory shared with system, be conservative
# CPU: Memory bound, keep small
BATCH_SIZE_BY_DEVICE = {
    "cuda": 8,  # Reduced from 16 - prevents OOM when indexing many files
    "mps": 8,  # M3 Pro conservative (shared memory)
    "cpu": 4,  # Memory bound
}


def clear_gpu_cache() -> None:
    """Clear GPU memory cache to prevent OOM during long indexing runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure cache is fully cleared


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""

    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return len(self.embedding)


def detect_device() -> str:
    """Auto-detect best available device for embeddings."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_batch_size(device: str) -> int:
    """Get device-appropriate batch size."""
    return BATCH_SIZE_BY_DEVICE.get(device, 32)


@dataclass
class GPUMemoryInfo:
    """GPU memory information in GB."""

    total: float
    used: float
    free: float
    device_name: str = ""

    def __str__(self) -> str:
        return f"{self.device_name}: {self.free:.1f}GB free / {self.total:.1f}GB total"


def get_gpu_memory() -> GPUMemoryInfo | None:
    """Get current GPU memory usage. Returns None if no GPU available."""
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / (1024**3)  # Convert to GB

    # Get allocated memory (more accurate than nvidia-smi for PyTorch)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)

    # Use reserved as "used" since that's what PyTorch has claimed
    used = reserved if reserved > 0 else allocated
    free = total - used

    return GPUMemoryInfo(
        total=total,
        used=used,
        free=free,
        device_name=props.name,
    )


def calculate_batch_size(
    free_memory_gb: float,
    model_size_gb: float = 6.0,  # BGE-M3 ~2.2GB + activations + buffer
    per_batch_gb: float = 2.0,  # ~2GB per batch item for long 10-K sections (conservative)
    min_batch: int = 4,
    max_batch: int = 8,  # Capped at 8 - higher causes OOM spikes with long SEC chunks
) -> int:
    """Calculate recommended batch size based on available GPU memory.

    Conservative estimates for long SEC filing sections (8K+ tokens each).
    For shorter texts, you can override with --batch-size.

    Note: Peak memory during forward pass can be much higher than steady-state.
    Batch size of 4-8 tested stable on RTX 4090 with BLK (340 files, 10K+ char chunks).

    Args:
        free_memory_gb: Available GPU memory in GB
        model_size_gb: Estimated model size + activations in GB
        per_batch_gb: Estimated memory per batch item in GB
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Recommended batch size
    """
    available = free_memory_gb - model_size_gb
    if available <= 0:
        return min_batch

    # Calculate how many batch items we can fit
    batch_size = int(available / per_batch_gb)

    # Clamp to reasonable range
    return max(min_batch, min(batch_size, max_batch))


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cuda/mps/cpu). Auto-detected if None.
            batch_size: Batch size for encoding. Auto-detected by device if None.
            normalize: Whether to L2-normalize embeddings (recommended for BGE)
        """
        self.model_name = model_name
        self.device = device or detect_device()
        self.batch_size = batch_size or get_default_batch_size(self.device)
        self.normalize = normalize

        # Lazy load model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the model."""
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"Model {self.model_name} does not report embedding dimension")
        return dim

    def encode(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            show_progress: Show progress bar

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.

        Note: BGE-M3 does not require instruction prefixes (unlike BGE-large).

        Args:
            query: Search query text

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embedding[0]

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = False,
    ) -> list[EmbeddedChunk]:
        """
        Embed a list of Chunk objects.

        Args:
            chunks: List of Chunk objects from parser
            show_progress: Show progress bar

        Returns:
            List of EmbeddedChunk objects with embeddings
        """
        if not chunks:
            return []

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Encode all texts
        embeddings = self.encode(texts, show_progress=show_progress)

        # Clear GPU cache after encoding to prevent OOM on long runs
        clear_gpu_cache()

        # Create EmbeddedChunk objects
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(
                EmbeddedChunk(
                    text=chunk.text,
                    embedding=embedding,
                    metadata=chunk.metadata.copy(),
                )
            )

        return embedded_chunks


@dataclass
class RerankResult:
    """Result from reranking a single document."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    original_rank: int = 0


class Reranker:
    """Cross-encoder reranker for improving search precision.

    Uses a cross-encoder model that scores (query, passage) pairs together,
    providing more accurate relevance scores than embedding similarity alone.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str | None = None,
        batch_size: int = 4,  # Very conservative - runs alongside embedding model with long texts
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            device: Device to use (cuda/mps/cpu). Auto-detected if None.
            batch_size: Batch size for scoring pairs
        """
        self.model_name = model_name
        self.device = device or detect_device()
        self.batch_size = batch_size

        # Lazy load model
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the model on first access."""
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        texts: list[str],
        top_k: int | None = None,
        metadata: list[dict] | None = None,
    ) -> list[RerankResult]:
        """
        Rerank texts by relevance to query.

        Args:
            query: The search query
            texts: List of texts to rerank
            top_k: Return only top K results (default: all)
            metadata: Optional metadata for each text

        Returns:
            List of RerankResult sorted by score (highest first)
        """
        if not texts:
            return []

        # Create (query, text) pairs for scoring
        pairs = [[query, text] for text in texts]

        # Score all pairs
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Handle metadata
        if metadata is None:
            metadata = [{} for _ in texts]

        # Create results with original rank tracking
        results = [
            RerankResult(
                text=text,
                score=float(score),
                metadata=meta,
                original_rank=i,
            )
            for i, (text, score, meta) in enumerate(zip(texts, scores, metadata))
        ]

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_candidates(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text",
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidate dictionaries, preserving all fields.

        Args:
            query: The search query
            candidates: List of dicts with text and other fields
            text_key: Key containing the text to rerank on
            top_k: Return only top K results

        Returns:
            List of candidate dicts with added 'rerank_score' and 'original_rank'
        """
        if not candidates:
            return []

        texts = [c[text_key] for c in candidates]
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Add scores and original rank to candidates
        scored = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            result["original_rank"] = i
            scored.append(result)

        # Sort by rerank score
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (-1 to 1, or 0 to 1 if normalized)
    """
    # If already normalized, dot product = cosine similarity
    return float(np.dot(a, b))


def batch_cosine_similarity(
    query: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple embeddings.

    Args:
        query: Query vector of shape (embedding_dim,)
        embeddings: Matrix of shape (n, embedding_dim)

    Returns:
        Array of similarities of shape (n,)
    """
    # If normalized, matrix-vector product gives similarities
    return embeddings @ query
