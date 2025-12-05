"""Tests for the LanceDB vector store."""

import numpy as np
import pytest

from src.rag.embeddings import EmbeddedChunk
from src.rag.store import IndexManifest, SearchResult, VectorStore

from .conftest import TEST_TICKER

# temp_index_dir fixture is provided by conftest.py


@pytest.fixture
def sample_chunks():
    """Create sample embedded chunks for testing."""
    # Use small random vectors for testing (dimension 64 for speed)
    np.random.seed(42)
    return [
        EmbeddedChunk(
            text="Company reported strong revenue growth in Q4 2024.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": TEST_TICKER,
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "section": "Item 7",
                "section_title": "Management's Discussion and Analysis",
                "source": f"{TEST_TICKER}/10-K_2024.html",
                "chunk_index": 0,
            },
        ),
        EmbeddedChunk(
            text="Risk factors include supply chain disruptions and competition.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": TEST_TICKER,
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "section": "Item 1A",
                "section_title": "Risk Factors",
                "source": f"{TEST_TICKER}/10-K_2024.html",
                "chunk_index": 1,
            },
        ),
        EmbeddedChunk(
            text="The company operates in multiple geographic regions.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": TEST_TICKER,
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "section": "Item 1",
                "section_title": "Business",
                "source": f"{TEST_TICKER}/10-K_2024.html",
                "chunk_index": 2,
            },
        ),
    ]


@pytest.fixture
def vector_store(temp_index_dir):
    """Create a vector store instance for testing."""
    return VectorStore(
        index_dir=temp_index_dir,
        ticker=TEST_TICKER,
        embedding_dim=64,
        embedding_model="test-model",
    )


class TestIndexManifest:
    """Tests for IndexManifest."""

    def test_create_manifest(self):
        manifest = IndexManifest(ticker=TEST_TICKER)
        assert manifest.ticker == TEST_TICKER
        assert manifest.indexed_files == {}
        assert manifest.chunk_count == 0

    def test_save_and_load(self, temp_index_dir):
        manifest = IndexManifest(
            ticker=TEST_TICKER,
            indexed_files={"file1.html": "hash123"},
            chunk_count=10,
            embedding_model="test-model",
            embedding_dim=64,
        )
        path = temp_index_dir / "manifest.json"
        manifest.save(path)

        loaded = IndexManifest.load(path)
        assert loaded.ticker == TEST_TICKER
        assert loaded.indexed_files == {"file1.html": "hash123"}
        assert loaded.chunk_count == 10
        assert loaded.embedding_model == "test-model"
        assert loaded.embedding_dim == 64


class TestVectorStore:
    """Tests for VectorStore initialization and basic operations."""

    def test_init_creates_directory(self, temp_index_dir):
        store = VectorStore(index_dir=temp_index_dir, ticker=TEST_TICKER)
        assert store.company_dir.exists()
        assert store.ticker == TEST_TICKER

    def test_ticker_uppercase(self, temp_index_dir):
        # Test that lowercase ticker gets uppercased
        store = VectorStore(index_dir=temp_index_dir, ticker=TEST_TICKER.lower())
        assert store.ticker == TEST_TICKER

    def test_empty_store_count(self, vector_store):
        assert vector_store.count() == 0

    def test_add_chunks(self, vector_store, sample_chunks):
        added = vector_store.add(sample_chunks, source_file="test.html", file_hash="abc123")
        assert added == 3
        assert vector_store.count() == 3

    def test_add_updates_manifest(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks, source_file="test.html", file_hash="abc123")
        assert vector_store.manifest.indexed_files.get("test.html") == "abc123"
        assert vector_store.manifest.chunk_count == 3

    def test_add_empty_list(self, vector_store):
        added = vector_store.add([])
        assert added == 0

    def test_is_file_indexed(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks, source_file="test.html", file_hash="abc123")

        assert vector_store.is_file_indexed("test.html", "abc123") is True
        assert vector_store.is_file_indexed("test.html", "different") is False
        assert vector_store.is_file_indexed("other.html", "abc123") is False


class TestVectorSearch:
    """Tests for vector similarity search."""

    def test_search_empty_store(self, vector_store):
        query = np.random.randn(64).astype(np.float32)
        results = vector_store.search_vector(query, top_k=5)
        assert results == []

    def test_search_returns_results(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks)
        # Use the first chunk's embedding as query (should match itself)
        query = sample_chunks[0].embedding
        results = vector_store.search_vector(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be the closest match
        assert results[0].text == sample_chunks[0].text

    def test_search_with_filter(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks)
        query = np.random.randn(64).astype(np.float32)

        results = vector_store.search_vector(query, top_k=10, filter_dict={"section": "Item 1A"})

        assert len(results) == 1
        assert results[0].section == "Item 1A"

    def test_search_result_metadata(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks)
        query = sample_chunks[0].embedding
        results = vector_store.search_vector(query, top_k=1)

        result = results[0]
        assert result.ticker == TEST_TICKER
        assert result.filing_type == "10-K"
        assert result.filing_date == "2024-01-15"
        assert result.section == "Item 7"


class TestFTSSearch:
    """Tests for full-text search."""

    def test_fts_empty_store(self, vector_store):
        results = vector_store.search_fts("revenue", top_k=5)
        assert results == []

    def test_fts_finds_matching_text(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks)
        results = vector_store.search_fts("revenue growth", top_k=3)

        assert len(results) > 0
        assert "revenue" in results[0].text.lower()

    def test_fts_with_filter(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks)
        results = vector_store.search_fts("company", top_k=10, filter_dict={"section": "Item 1"})

        assert all(r.section == "Item 1" for r in results)


class TestHybridSearch:
    """Tests for hybrid search."""

    def test_hybrid_empty_store(self, vector_store):
        query_vec = np.random.randn(64).astype(np.float32)
        results = vector_store.search_hybrid(query_vec, "test query", top_k=5)
        assert results == []

    def test_hybrid_returns_results(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks)
        query_vec = sample_chunks[0].embedding
        results = vector_store.search_hybrid(query_vec, "revenue growth", top_k=3)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)


class TestStoreClear:
    """Tests for clearing and deleting from store."""

    def test_clear(self, vector_store, sample_chunks):
        vector_store.add(sample_chunks, source_file="test.html", file_hash="abc")
        assert vector_store.count() == 3

        vector_store.clear()
        assert vector_store.count() == 0
        assert vector_store.manifest.indexed_files == {}

    def test_delete_by_source(self, vector_store):
        np.random.seed(42)

        # Add chunks from different sources
        chunks1 = [
            EmbeddedChunk(
                text="Content from file 1",
                embedding=np.random.randn(64).astype(np.float32),
                metadata={"source": "file1.html"},
            )
        ]
        chunks2 = [
            EmbeddedChunk(
                text="Content from file 2",
                embedding=np.random.randn(64).astype(np.float32),
                metadata={"source": "file2.html"},
            )
        ]

        vector_store.add(chunks1, source_file="file1.html", file_hash="hash1")
        vector_store.add(chunks2, source_file="file2.html", file_hash="hash2")

        assert vector_store.count() == 2

        deleted = vector_store.delete_by_source("file1.html")
        assert deleted == 1
        assert vector_store.count() == 1
        assert "file1.html" not in vector_store.manifest.indexed_files


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_properties(self):
        result = SearchResult(
            text="Test content",
            score=0.95,
            metadata={
                "ticker": TEST_TICKER,
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "section": "Item 7",
            },
        )

        assert result.ticker == TEST_TICKER
        assert result.filing_type == "10-K"
        assert result.filing_date == "2024-01-15"
        assert result.section == "Item 7"
        assert result.score == 0.95

    def test_search_result_missing_metadata(self):
        result = SearchResult(text="Test", score=0.5, metadata={})
        assert result.ticker == ""
        assert result.filing_type == ""


class TestIncrementalIndexing:
    """Tests for incremental indexing functionality."""

    def test_skip_already_indexed(self, vector_store, sample_chunks):
        # Index first time
        vector_store.add(sample_chunks, source_file="test.html", file_hash="hash123")

        # Check file is indexed
        assert vector_store.is_file_indexed("test.html", "hash123")

        # File with same hash should not be re-indexed (caller checks this)
        assert vector_store.is_file_indexed("test.html", "hash123") is True

    def test_reindex_changed_file(self, vector_store, sample_chunks):
        # Index first time
        vector_store.add(sample_chunks, source_file="test.html", file_hash="hash123")

        # File with different hash should be flagged for re-indexing
        assert vector_store.is_file_indexed("test.html", "hash456") is False

    def test_manifest_persistence(self, temp_index_dir, sample_chunks):
        # Create store, add chunks, close
        store1 = VectorStore(index_dir=temp_index_dir, ticker=TEST_TICKER, embedding_dim=64)
        store1.add(sample_chunks, source_file="test.html", file_hash="hash123")

        # Create new store instance, should load existing manifest
        store2 = VectorStore(index_dir=temp_index_dir, ticker=TEST_TICKER, embedding_dim=64)
        assert store2.is_file_indexed("test.html", "hash123")
        assert store2.count() == 3
