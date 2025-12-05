"""Tests for search feedback logging and analysis."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from src.rag.feedback import (
    DEFAULT_FEEDBACK_FILE,
    FeedbackLogger,
    SearchFeedbackEntry,
    analyze_feedback,
)


class MockSearchResult:
    """Mock SearchResult for testing."""

    def __init__(
        self,
        text: str,
        score: float,
        metadata: dict[str, Any] | None = None,
    ):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


class TestSearchFeedbackEntry:
    """Tests for SearchFeedbackEntry dataclass."""

    def test_required_fields(self):
        """Entry should have all required fields."""
        entry = SearchFeedbackEntry(
            timestamp="2024-01-15T10:00:00",
            ticker="AAPL",
            query="revenue growth",
            mode="hybrid",
            top_k=10,
            vector_weight=0.5,
            query_type="trend",
            query_confidence=0.8,
            num_results=5,
            reranked=False,
            filing_type_filter=None,
            section_filter=None,
            top_results=[],
        )
        assert entry.ticker == "AAPL"
        assert entry.query == "revenue growth"
        assert entry.mode == "hybrid"
        assert entry.query_type == "trend"

    def test_optional_fields_defaults(self):
        """Optional fields should have defaults."""
        entry = SearchFeedbackEntry(
            timestamp="2024-01-15T10:00:00",
            ticker="AAPL",
            query="test",
            mode="hybrid",
            top_k=10,
            vector_weight=0.5,
            query_type="default",
            query_confidence=0.5,
            num_results=0,
            reranked=False,
            filing_type_filter=None,
            section_filter=None,
            top_results=[],
        )
        assert entry.expanded_query is None
        assert entry.expansion_count == 0
        assert entry.user_rating is None
        assert entry.user_selected_indices is None

    def test_serialization_to_dict(self):
        """Entry should serialize to dict for JSON."""
        entry = SearchFeedbackEntry(
            timestamp="2024-01-15T10:00:00",
            ticker="AAPL",
            query="test query",
            mode="vector",
            top_k=5,
            vector_weight=0.7,
            query_type="conceptual",
            query_confidence=0.9,
            num_results=3,
            reranked=True,
            filing_type_filter="10-K",
            section_filter="Item 7",
            top_results=[{"score": 0.95, "preview": "sample"}],
        )
        data = asdict(entry)
        assert isinstance(data, dict)
        assert data["ticker"] == "AAPL"
        assert data["top_results"] == [{"score": 0.95, "preview": "sample"}]


class TestFeedbackLogger:
    """Tests for FeedbackLogger class."""

    def test_init_sets_paths(self, tmp_path: Path):
        """Logger should set correct log path."""
        logger = FeedbackLogger(tmp_path, "aapl")
        assert logger.ticker == "AAPL"  # Upper-cased
        assert logger.log_path == tmp_path / "AAPL" / DEFAULT_FEEDBACK_FILE

    def test_log_search_creates_file(self, tmp_path: Path):
        """log_search should create JSONL file."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        results = [
            MockSearchResult("Revenue was $100M", 0.95, {"filing_type": "10-K"}),
            MockSearchResult("Growth increased by 15%", 0.88, {"filing_type": "10-Q"}),
        ]

        logger.log_search(
            query="revenue growth",
            mode="hybrid",
            top_k=10,
            vector_weight=0.6,
            query_type="trend",
            query_confidence=0.8,
            results=results,
        )

        assert logger.log_path.exists()

    def test_log_search_writes_valid_json(self, tmp_path: Path):
        """log_search should write valid JSON lines."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        results = [
            MockSearchResult("Test content", 0.9, {"filing_type": "10-K"}),
        ]

        logger.log_search(
            query="test query",
            mode="hybrid",
            top_k=5,
            vector_weight=0.5,
            query_type="exact_metric",
            query_confidence=0.85,
            results=results,
        )

        with open(logger.log_path) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["ticker"] == "AAPL"
        assert entry["query"] == "test query"
        assert entry["mode"] == "hybrid"
        assert entry["query_type"] == "exact_metric"
        assert entry["num_results"] == 1

    def test_log_search_truncates_preview(self, tmp_path: Path):
        """Long text should be truncated in preview."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        long_text = "x" * 300
        results = [MockSearchResult(long_text, 0.9, {})]

        logger.log_search(
            query="test",
            mode="vector",
            top_k=5,
            vector_weight=0.7,
            query_type="default",
            query_confidence=0.5,
            results=results,
        )

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        preview = entry["top_results"][0]["preview"]
        assert len(preview) == 203  # 200 chars + "..."
        assert preview.endswith("...")

    def test_log_search_limits_to_10_results(self, tmp_path: Path):
        """Only top 10 results should be logged."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        results = [MockSearchResult(f"Result {i}", 0.9 - i * 0.01, {}) for i in range(15)]

        logger.log_search(
            query="test",
            mode="fts",
            top_k=15,
            vector_weight=0.3,
            query_type="entity",
            query_confidence=0.7,
            results=results,
        )

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert len(entry["top_results"]) == 10
        assert entry["num_results"] == 15  # But total count preserved

    def test_log_search_with_filters(self, tmp_path: Path):
        """Filters should be logged correctly."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        logger.log_search(
            query="test",
            mode="hybrid",
            top_k=5,
            vector_weight=0.5,
            query_type="default",
            query_confidence=0.5,
            results=[],
            filing_type_filter="10-K",
            section_filter="Item 7",
        )

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["filing_type_filter"] == "10-K"
        assert entry["section_filter"] == "Item 7"

    def test_log_search_with_expansion(self, tmp_path: Path):
        """Query expansion fields should be logged."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        logger.log_search(
            query="ARR",
            mode="fts",
            top_k=10,
            vector_weight=0.3,
            query_type="exact_metric",
            query_confidence=0.9,
            results=[],
            expanded_query="ARR OR 'annual recurring revenue'",
            expansion_count=1,
        )

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["expanded_query"] == "ARR OR 'annual recurring revenue'"
        assert entry["expansion_count"] == 1

    def test_log_search_with_reranking(self, tmp_path: Path):
        """Reranking flag should be logged."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        logger.log_search(
            query="test",
            mode="hybrid",
            top_k=5,
            vector_weight=0.5,
            query_type="default",
            query_confidence=0.5,
            results=[],
            reranked=True,
        )

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["reranked"] is True

    def test_log_search_appends_entries(self, tmp_path: Path):
        """Multiple searches should append to same file."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        for i in range(3):
            logger.log_search(
                query=f"query {i}",
                mode="hybrid",
                top_k=5,
                vector_weight=0.5,
                query_type="default",
                query_confidence=0.5,
                results=[],
            )

        with open(logger.log_path) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0])["query"] == "query 0"
        assert json.loads(lines[2])["query"] == "query 2"

    def test_log_user_feedback(self, tmp_path: Path):
        """User feedback should be logged correctly."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        logger.log_user_feedback(
            query="revenue growth",
            rating=4,
            selected_indices=[0, 2],
        )

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["type"] == "user_feedback"
        assert entry["query"] == "revenue growth"
        assert entry["user_rating"] == 4
        assert entry["user_selected_indices"] == [0, 2]

    def test_log_handles_write_errors_gracefully(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Write errors should be logged but not raise."""
        logger = FeedbackLogger(tmp_path, "AAPL")

        # Mock open to raise an error
        def mock_open(*args, **kwargs):
            raise PermissionError("Cannot write")

        monkeypatch.setattr("builtins.open", mock_open)

        # Should not raise
        logger.log_search(
            query="test",
            mode="hybrid",
            top_k=5,
            vector_weight=0.5,
            query_type="default",
            query_confidence=0.5,
            results=[],
        )


class TestAnalyzeFeedback:
    """Tests for analyze_feedback function."""

    def test_nonexistent_file_returns_error(self, tmp_path: Path):
        """Missing file should return error dict."""
        result = analyze_feedback(tmp_path / "nonexistent.jsonl")
        assert "error" in result
        assert "No feedback log found" in result["error"]

    def test_empty_file_returns_error(self, tmp_path: Path):
        """Empty file should return error."""
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")

        result = analyze_feedback(log_path)
        assert "error" in result
        assert "No search entries found" in result["error"]

    def test_only_user_feedback_returns_error(self, tmp_path: Path):
        """File with only user feedback should return error."""
        log_path = tmp_path / "feedback.jsonl"
        log_path.write_text(json.dumps({"type": "user_feedback", "query": "test", "rating": 5}) + "\n")

        result = analyze_feedback(log_path)
        assert "error" in result
        assert "No search entries found" in result["error"]

    def test_counts_total_searches(self, tmp_path: Path):
        """Should count total search entries."""
        log_path = tmp_path / "feedback.jsonl"
        entries = [
            {"mode": "hybrid", "query_type": "default", "num_results": 5},
            {"mode": "vector", "query_type": "conceptual", "num_results": 3},
            {"mode": "fts", "query_type": "exact_metric", "num_results": 10},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        result = analyze_feedback(log_path)
        assert result["total_searches"] == 3

    def test_query_type_distribution(self, tmp_path: Path):
        """Should calculate query type distribution."""
        log_path = tmp_path / "feedback.jsonl"
        entries = [
            {"mode": "hybrid", "query_type": "exact_metric", "num_results": 5},
            {"mode": "hybrid", "query_type": "conceptual", "num_results": 3},
            {"mode": "hybrid", "query_type": "exact_metric", "num_results": 7},
            {"mode": "hybrid", "query_type": "entity", "num_results": 4},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = analyze_feedback(log_path)
        assert result["query_type_distribution"]["exact_metric"] == 2
        assert result["query_type_distribution"]["conceptual"] == 1
        assert result["query_type_distribution"]["entity"] == 1

    def test_mode_distribution(self, tmp_path: Path):
        """Should calculate mode distribution."""
        log_path = tmp_path / "feedback.jsonl"
        entries = [
            {"mode": "hybrid", "query_type": "default", "num_results": 5},
            {"mode": "vector", "query_type": "default", "num_results": 3},
            {"mode": "hybrid", "query_type": "default", "num_results": 7},
            {"mode": "fts", "query_type": "default", "num_results": 4},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = analyze_feedback(log_path)
        assert result["mode_distribution"]["hybrid"] == 2
        assert result["mode_distribution"]["vector"] == 1
        assert result["mode_distribution"]["fts"] == 1

    def test_low_result_queries(self, tmp_path: Path):
        """Should identify queries with few results."""
        log_path = tmp_path / "feedback.jsonl"
        entries = [
            {"mode": "hybrid", "query_type": "default", "num_results": 10, "query": "good query"},
            {"mode": "hybrid", "query_type": "entity", "num_results": 0, "query": "no results"},
            {"mode": "hybrid", "query_type": "default", "num_results": 2, "query": "few results"},
            {"mode": "hybrid", "query_type": "default", "num_results": 5, "query": "ok query"},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = analyze_feedback(log_path)
        low = result["low_result_queries"]
        assert len(low) == 2
        queries = [q["query"] for q in low]
        assert "no results" in queries
        assert "few results" in queries

    def test_avg_results_by_query_type(self, tmp_path: Path):
        """Should calculate average results by query type."""
        log_path = tmp_path / "feedback.jsonl"
        entries = [
            {"mode": "hybrid", "query_type": "exact_metric", "num_results": 10},
            {"mode": "hybrid", "query_type": "exact_metric", "num_results": 6},
            {"mode": "hybrid", "query_type": "conceptual", "num_results": 3},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = analyze_feedback(log_path)
        assert result["avg_results_by_query_type"]["exact_metric"] == 8.0  # (10+6)/2
        assert result["avg_results_by_query_type"]["conceptual"] == 3.0

    def test_limits_low_result_queries(self, tmp_path: Path):
        """Should limit low result queries to 20."""
        log_path = tmp_path / "feedback.jsonl"
        entries = [
            {"mode": "hybrid", "query_type": "default", "num_results": 0, "query": f"query {i}"} for i in range(30)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = analyze_feedback(log_path)
        assert len(result["low_result_queries"]) == 20

    def test_handles_blank_lines(self, tmp_path: Path):
        """Should handle blank lines in JSONL."""
        log_path = tmp_path / "feedback.jsonl"
        content = (
            json.dumps({"mode": "hybrid", "query_type": "default", "num_results": 5})
            + "\n\n"
            + json.dumps({"mode": "vector", "query_type": "conceptual", "num_results": 3})
            + "\n"
        )
        log_path.write_text(content)

        result = analyze_feedback(log_path)
        assert result["total_searches"] == 2

    def test_handles_mixed_entry_types(self, tmp_path: Path):
        """Should filter out user feedback entries."""
        log_path = tmp_path / "feedback.jsonl"
        entries = [
            {"mode": "hybrid", "query_type": "default", "num_results": 5},
            {"type": "user_feedback", "query": "test", "rating": 4},
            {"mode": "vector", "query_type": "conceptual", "num_results": 3},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = analyze_feedback(log_path)
        assert result["total_searches"] == 2  # Only search entries
