"""Tests for the search module."""

import numpy as np
import pytest

from src.rag.embeddings import EmbeddedChunk
from src.rag.search import SearchConfig, Searcher, create_searcher
from src.rag.store import VectorStore

# temp_index_dir fixture is provided by conftest.py


@pytest.fixture
def populated_store(temp_index_dir):
    """Create a vector store with sample data."""
    store = VectorStore(
        index_dir=temp_index_dir,
        ticker="TEST",
        embedding_dim=64,
    )

    np.random.seed(42)
    chunks = [
        EmbeddedChunk(
            text="Apple reported strong revenue growth driven by iPhone sales.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": "TEST",
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "section": "Item 7",
                "section_title": "MD&A",
                "source": "test.html",
            },
        ),
        EmbeddedChunk(
            text="Risk factors include supply chain disruptions and competition from Samsung.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": "TEST",
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "section": "Item 1A",
                "section_title": "Risk Factors",
                "source": "test.html",
            },
        ),
        EmbeddedChunk(
            text="The company operates retail stores worldwide.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": "TEST",
                "filing_type": "10-Q",
                "filing_date": "2024-04-15",
                "section": "Item 1",
                "section_title": "Business",
                "source": "test2.html",
            },
        ),
    ]

    store.add(chunks)
    return store


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_config(self):
        config = SearchConfig()
        assert config.mode == "hybrid"
        assert config.top_k == 10
        assert config.vector_weight == 0.7

    def test_custom_config(self):
        config = SearchConfig(mode="vector", top_k=5)
        assert config.mode == "vector"
        assert config.top_k == 5


class TestSearcher:
    """Tests for Searcher class."""

    def test_init_requires_index(self, temp_index_dir):
        """Searcher should fail if no index exists."""
        with pytest.raises(ValueError, match="No index found"):
            Searcher(index_dir=temp_index_dir, ticker="NOEXIST")

    def test_init_with_existing_index(self, temp_index_dir, populated_store):
        """Searcher should work with existing index."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        assert searcher.ticker == "TEST"
        assert searcher.store.count() == 3

    def test_search_fts(self, temp_index_dir, populated_store):
        """FTS search should find matching text."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        results = searcher.search("revenue growth", mode="fts")

        assert len(results) > 0
        assert "revenue" in results[0].text.lower()

    def test_search_with_filing_type_filter(self, temp_index_dir, populated_store):
        """Search should filter by filing type."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        results = searcher.search("company", mode="fts", filing_type="10-Q")

        assert len(results) > 0
        assert all(r.filing_type == "10-Q" for r in results)

    def test_search_with_section_filter(self, temp_index_dir, populated_store):
        """Search should filter by section."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        results = searcher.search("risk", mode="fts", section="Item 1A")

        assert len(results) > 0
        assert all(r.section == "Item 1A" for r in results)

    def test_search_top_k(self, temp_index_dir, populated_store):
        """Search should respect top_k limit."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        results = searcher.search("company", mode="fts", top_k=1)

        assert len(results) <= 1


class TestSearcherContextWindow:
    """Tests for context window generation."""

    def test_get_context_window(self, temp_index_dir, populated_store):
        """Should return formatted context string."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        # Use FTS mode since test fixtures have small embeddings
        results, context = searcher.get_context_window("revenue", max_tokens=1000, mode="fts")

        assert len(results) > 0
        assert isinstance(context, str)
        assert len(context) > 0

    def test_context_window_respects_token_limit(self, temp_index_dir, populated_store):
        """Context should not exceed token limit."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        max_tokens = 100
        # Use FTS mode since test fixtures have small embeddings
        results, context = searcher.get_context_window("revenue", max_tokens=max_tokens, mode="fts")

        # Rough check: context chars should be around max_tokens * 4
        max_chars = max_tokens * 4
        assert len(context) <= max_chars * 1.5  # Allow some margin


class TestCreateSearcher:
    """Tests for factory function."""

    def test_create_searcher(self, temp_index_dir, populated_store):
        """Factory should create configured searcher."""
        searcher = create_searcher(
            index_dir=temp_index_dir,
            ticker="TEST",
            mode="vector",
            top_k=5,
        )

        assert searcher.config.mode == "vector"
        assert searcher.config.top_k == 5

    def test_create_searcher_case_insensitive(self, temp_index_dir, populated_store):
        """Factory should handle lowercase ticker."""
        searcher = create_searcher(
            index_dir=temp_index_dir,
            ticker="test",
        )

        assert searcher.ticker == "TEST"
