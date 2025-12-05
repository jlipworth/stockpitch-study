"""Tests for the RAG query module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.embeddings import EmbeddedChunk
from src.rag.query import DEFAULT_RAG_MODEL, RAGQueryEngine, RAGResponse, create_rag_engine
from src.rag.search import Searcher
from src.rag.store import SearchResult, VectorStore

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
            text="Apple reported strong revenue growth of 15% driven by iPhone sales.",
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
            text="Risk factors include supply chain disruptions and intense competition.",
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
    ]

    store.add(chunks)
    return store


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Based on the 10-K filing, revenue grew 15% driven by iPhone sales.")]
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 50
    return mock_response


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_response_creation(self):
        sources = [
            SearchResult(
                text="Test content",
                score=0.9,
                metadata={"filing_type": "10-K", "filing_date": "2024-01-15", "section": "Item 7"},
            )
        ]
        response = RAGResponse(
            answer="Test answer",
            sources=sources,
            model="claude-sonnet-4-5-20250929",
            input_tokens=100,
            output_tokens=50,
        )

        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_source_citations(self):
        sources = [
            SearchResult(
                text="Content 1",
                score=0.9,
                metadata={"filing_type": "10-K", "filing_date": "2024-01-15", "section": "Item 7"},
            ),
            SearchResult(
                text="Content 2",
                score=0.8,
                metadata={"filing_type": "10-Q", "filing_date": "2024-04-15", "section": "Item 1A"},
            ),
        ]
        response = RAGResponse(
            answer="Test",
            sources=sources,
            model="test",
            input_tokens=0,
            output_tokens=0,
        )

        citations = response.source_citations
        assert "[1] 10-K (2024-01-15) - Item 7" in citations
        assert "[2] 10-Q (2024-04-15) - Item 1A" in citations

    def test_empty_sources_citation(self):
        response = RAGResponse(
            answer="Test",
            sources=[],
            model="test",
            input_tokens=0,
            output_tokens=0,
        )

        assert response.source_citations == "No sources used."


class TestRAGQueryEngine:
    """Tests for RAGQueryEngine."""

    def test_init_requires_api_key(self, temp_index_dir, populated_store):
        """Engine should fail without API key."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                RAGQueryEngine(searcher=searcher)

    def test_init_with_api_key(self, temp_index_dir, populated_store):
        """Engine should initialize with API key."""
        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")

        with patch("src.rag.query.Anthropic"):
            engine = RAGQueryEngine(searcher=searcher, api_key="test-key")
            assert engine.model == DEFAULT_RAG_MODEL

    @patch("src.rag.query.Anthropic")
    def test_query_returns_response(
        self, mock_anthropic_class, temp_index_dir, populated_store, mock_anthropic_response
    ):
        """Query should return RAGResponse with answer and sources."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        engine = RAGQueryEngine(searcher=searcher, api_key="test-key")

        response = engine.query("What was the revenue growth?", search_mode="fts")

        assert isinstance(response, RAGResponse)
        assert "revenue" in response.answer.lower() or "15%" in response.answer
        assert response.input_tokens == 500
        assert response.output_tokens == 50

    @patch("src.rag.query.Anthropic")
    def test_query_with_filing_type_filter(
        self, mock_anthropic_class, temp_index_dir, populated_store, mock_anthropic_response
    ):
        """Query should filter by filing type."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        searcher = Searcher(index_dir=temp_index_dir, ticker="TEST")
        engine = RAGQueryEngine(searcher=searcher, api_key="test-key")

        response = engine.query(
            "What are the risks?",
            search_mode="fts",
            filing_type="10-K",
        )

        assert isinstance(response, RAGResponse)

    @patch("src.rag.query.Anthropic")
    def test_query_no_context_returns_empty_response(self, mock_anthropic_class, temp_index_dir):
        """Query with no matching context should return appropriate message."""
        # Create empty store
        store = VectorStore(index_dir=temp_index_dir, ticker="EMPTY", embedding_dim=64)
        np.random.seed(42)
        store.add(
            [
                EmbeddedChunk(
                    text="Unrelated content",
                    embedding=np.random.randn(64).astype(np.float32),
                    metadata={
                        "ticker": "EMPTY",
                        "filing_type": "10-K",
                        "filing_date": "2024-01-01",
                        "section": "Item 1",
                    },
                )
            ]
        )

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        searcher = Searcher(index_dir=temp_index_dir, ticker="EMPTY")
        engine = RAGQueryEngine(searcher=searcher, api_key="test-key")

        # Query for something not in the index with very specific filter
        response = engine.query(
            "What is the weather?",
            search_mode="fts",
            filing_type="8-K",  # No 8-Ks in the store
        )

        # Should return a message about no context
        assert "couldn't find" in response.answer.lower() or len(response.sources) == 0


class TestCreateRAGEngine:
    """Tests for factory function."""

    @patch("src.rag.query.Anthropic")
    def test_create_rag_engine(self, mock_anthropic_class, temp_index_dir, populated_store):
        """Factory should create configured engine."""
        engine = create_rag_engine(
            index_dir=temp_index_dir,
            ticker="TEST",
            api_key="test-key",
        )

        assert engine.searcher.ticker == "TEST"
        assert engine.model == DEFAULT_RAG_MODEL

    def test_create_rag_engine_no_index(self, temp_index_dir):
        """Factory should fail if no index exists."""
        with pytest.raises(ValueError, match="No index found"):
            create_rag_engine(
                index_dir=temp_index_dir,
                ticker="NOEXIST",
                api_key="test-key",
            )
