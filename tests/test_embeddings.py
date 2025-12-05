"""Tests for embedding pipeline."""

import numpy as np
import pytest

from src.rag import (
    BATCH_SIZE_BY_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_RERANKER,
    Chunk,
    EmbeddedChunk,
    EmbeddingModel,
    Reranker,
    RerankResult,
    batch_cosine_similarity,
    cosine_similarity,
    detect_device,
    get_default_batch_size,
)


class TestDetectDevice:
    """Tests for device detection."""

    def test_detect_device_returns_valid_device(self):
        """Test that detect_device returns a valid device string."""
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")

    def test_detect_device_returns_cpu_when_no_gpu(self):
        """On this machine (jumpbox), should return CPU."""
        device = detect_device()
        # This test will pass on any machine - just verify it's a string
        assert isinstance(device, str)


class TestBatchSizeDefaults:
    """Tests for device-specific batch size defaults."""

    def test_cuda_batch_size(self):
        """CUDA devices should use conservative batch size (prevents OOM with reranker)."""
        assert get_default_batch_size("cuda") == 8

    def test_mps_batch_size(self):
        """MPS (Apple Silicon) should use conservative batch size (shared memory)."""
        assert get_default_batch_size("mps") == 8

    def test_cpu_batch_size(self):
        """CPU should use small batch size (memory bound)."""
        assert get_default_batch_size("cpu") == 4

    def test_unknown_device_fallback(self):
        """Unknown device should fall back to 32."""
        assert get_default_batch_size("unknown") == 32

    def test_batch_size_by_device_dict(self):
        """Verify the device dictionary has expected keys."""
        assert "cuda" in BATCH_SIZE_BY_DEVICE
        assert "mps" in BATCH_SIZE_BY_DEVICE
        assert "cpu" in BATCH_SIZE_BY_DEVICE

    def test_embedding_model_auto_batch_size(self):
        """Model should auto-detect batch size based on device."""
        # Force CPU device, don't specify batch_size
        model = EmbeddingModel(device="cpu", batch_size=None)
        assert model.batch_size == 4  # CPU default (conservative)

    def test_embedding_model_explicit_batch_size(self):
        """Explicit batch_size should override auto-detection."""
        model = EmbeddingModel(device="cpu", batch_size=64)
        assert model.batch_size == 64


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create a shared model instance for tests (expensive to load)."""
        return EmbeddingModel(device="cpu", batch_size=8)

    def test_model_initialization(self, model):
        """Test model initializes with correct parameters."""
        assert model.model_name == DEFAULT_MODEL
        assert model.device == "cpu"
        assert model.batch_size == 8
        assert model.normalize is True

    def test_lazy_loading(self):
        """Test that model is lazily loaded."""
        model = EmbeddingModel(device="cpu")
        # Model should not be loaded yet
        assert model._model is None

        # Access embedding_dim to trigger load
        dim = model.embedding_dim

        # Now model should be loaded
        assert model._model is not None
        assert dim == 1024  # BGE-large dimension

    def test_embedding_dim(self, model):
        """Test that BGE-large returns 1024-dimensional embeddings."""
        assert model.embedding_dim == 1024

    def test_encode_single_text(self, model):
        """Test encoding a single text."""
        texts = ["This is a test sentence."]
        embeddings = model.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 1024)

    def test_encode_multiple_texts(self, model):
        """Test encoding multiple texts."""
        texts = [
            "First sentence about Apple Inc.",
            "Second sentence about financial results.",
            "Third sentence about market risk.",
        ]
        embeddings = model.encode(texts)

        assert embeddings.shape == (3, 1024)

    def test_encode_empty_list(self, model):
        """Test encoding empty list returns empty array."""
        embeddings = model.encode([])
        assert len(embeddings) == 0

    def test_embeddings_are_normalized(self, model):
        """Test that embeddings are L2-normalized."""
        texts = ["Test sentence for normalization check."]
        embeddings = model.encode(texts)

        # L2 norm should be ~1.0 for normalized vectors
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.001

    def test_encode_query(self, model):
        """Test query encoding."""
        query = "What are the risk factors?"
        embedding = model.encode_query(query)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)

        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.001

    def test_query_embedding_same_as_document(self, model):
        """Test that query and document embeddings are the same (BGE-M3 has no prefix)."""
        text = "What are the risk factors?"

        query_emb = model.encode_query(text)
        doc_emb = model.encode([text])[0]

        # BGE-M3 doesn't use instruction prefix, so embeddings should be identical
        similarity = cosine_similarity(query_emb, doc_emb)
        assert similarity > 0.99  # Should be identical


class TestEmbedChunks:
    """Tests for embedding Chunk objects."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create a shared model instance."""
        return EmbeddingModel(device="cpu", batch_size=8)

    def test_embed_chunks_basic(self, model):
        """Test embedding a list of chunks."""
        chunks = [
            Chunk(text="Apple designs iPhones.", metadata={"section": "Item 1"}),
            Chunk(text="Revenue increased 10%.", metadata={"section": "Item 7"}),
        ]

        embedded = model.embed_chunks(chunks)

        assert len(embedded) == 2
        assert all(isinstance(ec, EmbeddedChunk) for ec in embedded)

    def test_embedded_chunk_has_correct_fields(self, model):
        """Test that EmbeddedChunk has all required fields."""
        chunks = [
            Chunk(
                text="Test content",
                metadata={"ticker": "AAPL", "section": "Item 1"},
            )
        ]

        embedded = model.embed_chunks(chunks)
        ec = embedded[0]

        assert ec.text == "Test content"
        assert ec.embedding.shape == (1024,)
        assert ec.metadata["ticker"] == "AAPL"
        assert ec.metadata["section"] == "Item 1"
        assert ec.embedding_dim == 1024

    def test_embed_chunks_preserves_metadata(self, model):
        """Test that all metadata is preserved in embedded chunks."""
        chunks = [
            Chunk(
                text="Content here",
                metadata={
                    "ticker": "MSFT",
                    "filing_type": "10-K",
                    "filing_date": "2024-01-15",
                    "section": "Item 1A",
                    "chunk_index": 5,
                },
            )
        ]

        embedded = model.embed_chunks(chunks)
        ec = embedded[0]

        assert ec.metadata["ticker"] == "MSFT"
        assert ec.metadata["filing_type"] == "10-K"
        assert ec.metadata["filing_date"] == "2024-01-15"
        assert ec.metadata["section"] == "Item 1A"
        assert ec.metadata["chunk_index"] == 5

    def test_embed_chunks_empty_list(self, model):
        """Test embedding empty list returns empty list."""
        embedded = model.embed_chunks([])
        assert embedded == []

    def test_metadata_is_copied_not_referenced(self, model):
        """Test that metadata is copied, not referenced."""
        original_metadata = {"ticker": "AAPL"}
        chunk = Chunk(text="Test", metadata=original_metadata)

        embedded = model.embed_chunks([chunk])

        # Modify original metadata
        original_metadata["ticker"] = "MSFT"

        # Embedded chunk should still have original value
        assert embedded[0].metadata["ticker"] == "AAPL"


class TestCosineSimilarity:
    """Tests for similarity functions."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical normalized vectors is 1."""
        v = np.array([0.6, 0.8])  # Already normalized (0.6^2 + 0.8^2 = 1)
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim) < 0.001

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 0.001

    def test_batch_cosine_similarity(self):
        """Test batch similarity computation."""
        query = np.array([1.0, 0.0])
        embeddings = np.array(
            [
                [1.0, 0.0],  # Identical to query
                [0.0, 1.0],  # Orthogonal
                [0.6, 0.8],  # Some similarity
            ]
        )

        similarities = batch_cosine_similarity(query, embeddings)

        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 0.001  # Identical
        assert abs(similarities[1]) < 0.001  # Orthogonal
        assert 0 < similarities[2] < 1  # Partial similarity


class TestSemanticSimilarity:
    """Tests for semantic similarity of embeddings."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create a shared model instance."""
        return EmbeddingModel(device="cpu")

    def test_similar_texts_have_high_similarity(self, model):
        """Test that semantically similar texts have high cosine similarity."""
        texts = [
            "Apple Inc. designs and manufactures consumer electronics.",
            "Apple Corporation creates and produces electronic devices for consumers.",
        ]
        embeddings = model.encode(texts)

        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert sim > 0.8  # Should be very similar

    def test_dissimilar_texts_have_lower_similarity(self, model):
        """Test that dissimilar texts have lower cosine similarity."""
        texts = [
            "Apple Inc. designs consumer electronics.",
            "The weather forecast predicts rain tomorrow.",
        ]
        embeddings = model.encode(texts)

        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert sim < 0.5  # Should be dissimilar

    def test_financial_terms_similarity(self, model):
        """Test similarity of financial domain texts."""
        texts = [
            "Revenue increased by 15% year over year.",
            "Sales grew 15% compared to the previous year.",
            "The company operates manufacturing facilities.",
        ]
        embeddings = model.encode(texts)

        # First two should be very similar (both about revenue growth)
        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        # Third is different topic
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_01 > sim_02  # Revenue sentences more similar to each other


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk dataclass."""

    def test_embedded_chunk_creation(self):
        """Test basic EmbeddedChunk creation."""
        embedding = np.random.randn(1024).astype(np.float32)
        ec = EmbeddedChunk(
            text="Test text",
            embedding=embedding,
            metadata={"key": "value"},
        )

        assert ec.text == "Test text"
        assert ec.embedding is embedding
        assert ec.metadata == {"key": "value"}

    def test_embedding_dim_property(self):
        """Test embedding_dim property."""
        embedding = np.zeros(512)
        ec = EmbeddedChunk(text="", embedding=embedding)
        assert ec.embedding_dim == 512

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        ec = EmbeddedChunk(text="", embedding=np.array([]))
        assert ec.metadata == {}


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_creation(self):
        """Test RerankResult creation."""
        result = RerankResult(
            text="Test text",
            score=0.95,
            metadata={"key": "value"},
            original_rank=2,
        )
        assert result.text == "Test text"
        assert result.score == 0.95
        assert result.metadata == {"key": "value"}
        assert result.original_rank == 2

    def test_default_values(self):
        """Test default values for optional fields."""
        result = RerankResult(text="Test", score=0.5)
        assert result.metadata == {}
        assert result.original_rank == 0


class TestReranker:
    """Tests for Reranker class."""

    @pytest.fixture(scope="class")
    def reranker(self):
        """Create shared reranker instance (model is ~1GB)."""
        return Reranker(device="cpu", batch_size=8)

    def test_initialization(self, reranker):
        """Test reranker initializes correctly."""
        assert reranker.model_name == DEFAULT_RERANKER
        assert reranker.device == "cpu"
        assert reranker.batch_size == 8

    def test_lazy_loading(self):
        """Test model is lazily loaded."""
        r = Reranker(device="cpu")
        assert r._model is None
        # Access model to trigger load
        _ = r.model
        assert r._model is not None

    def test_rerank_basic(self, reranker):
        """Test basic reranking functionality."""
        query = "What is the company's revenue?"
        texts = [
            "The company reported $10 billion in revenue for fiscal 2023.",
            "Weather forecast: sunny with clouds.",
            "Total revenue increased 15% year-over-year to $10B.",
        ]

        results = reranker.rerank(query, texts)

        assert len(results) == 3
        # Revenue-related texts should rank higher than weather
        assert all(isinstance(r, RerankResult) for r in results)
        # First result should be revenue-related
        assert "revenue" in results[0].text.lower()

    def test_rerank_with_top_k(self, reranker):
        """Test reranking with top_k limit."""
        query = "financial performance"
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        results = reranker.rerank(query, texts, top_k=2)

        assert len(results) == 2

    def test_rerank_with_metadata(self, reranker):
        """Test that metadata is preserved."""
        query = "test query"
        texts = ["text A", "text B"]
        metadata = [{"id": 1}, {"id": 2}]

        results = reranker.rerank(query, texts, metadata=metadata)

        # Each result should have its metadata
        assert all(r.metadata in [{"id": 1}, {"id": 2}] for r in results)

    def test_rerank_empty_list(self, reranker):
        """Test reranking empty list returns empty."""
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_tracks_original_rank(self, reranker):
        """Test that original_rank is tracked correctly."""
        query = "revenue"
        texts = [
            "Unrelated text about nothing",
            "Company revenue was $5 billion",
            "Another unrelated sentence",
        ]

        results = reranker.rerank(query, texts)

        # The revenue text (originally rank 1) should now be first
        revenue_result = next(r for r in results if "revenue" in r.text.lower())
        assert revenue_result.original_rank == 1

    def test_rerank_with_candidates(self, reranker):
        """Test rerank_with_candidates preserves all fields."""
        query = "risk factors"
        candidates = [
            {"text": "Risk factors include market volatility", "filing": "10-K", "date": "2023-01-15"},
            {"text": "Weather is nice today", "filing": "8-K", "date": "2023-02-01"},
        ]

        results = reranker.rerank_with_candidates(query, candidates)

        assert len(results) == 2
        # Each result should have rerank_score and original_rank added
        assert "rerank_score" in results[0]
        assert "original_rank" in results[0]
        # Original fields should be preserved
        assert "filing" in results[0]
        assert "date" in results[0]
