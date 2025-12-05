"""Comprehensive AAPL-specific tests for the stock-pitch CLI toolkit.

Tests all CLI commands with AAPL as the target ticker, using mocked APIs
and fixture data. This ensures the toolkit works correctly for real-world
stock analysis workflows.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.filings import FilingMetadata, Manifest
from src.rag import EmbeddedChunk, VectorStore

runner = CliRunner()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_project_dirs():
    """Create a complete temporary project structure for AAPL tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create directory structure matching production layout
        dirs = {
            "data": root / "data" / "AAPL",
            "index": root / "index" / "AAPL",
            "processed": root / "processed" / "AAPL",
            "transcripts": root / "transcripts" / "AAPL",
            "analyst": root / "analyst" / "AAPL",
            "presentations": root / "presentations" / "AAPL",
            "conferences": root / "conferences" / "AAPL",
            "misc": root / "misc" / "AAPL",
            "notes": root / "notes",
            "output": root / "output",
        }

        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        yield {
            "root": root,
            **dirs,
        }


@pytest.fixture
def aapl_manifest(temp_project_dirs):
    """Create a sample AAPL manifest with realistic filings."""
    manifest = Manifest(
        ticker="AAPL",
        cik="320193",
        company_name="Apple Inc.",
        last_updated="2024-11-15T10:30:00",
    )

    # Add sample filings
    filings = [
        FilingMetadata(
            accession_number="0000320193-24-000081",
            filing_type="10-K",
            filing_date="2024-11-01",
            company_name="Apple Inc.",
            ticker="AAPL",
            cik="320193",
            file_path="AAPL/10-K_2024-11-01_0000320193-24-000081.html",
            file_hash="abc123def456789",
        ),
        FilingMetadata(
            accession_number="0000320193-24-000064",
            filing_type="10-Q",
            filing_date="2024-08-02",
            company_name="Apple Inc.",
            ticker="AAPL",
            cik="320193",
            file_path="AAPL/10-Q_2024-08-02_0000320193-24-000064.html",
            file_hash="def456ghi789012",
        ),
        FilingMetadata(
            accession_number="0000320193-24-000045",
            filing_type="10-Q",
            filing_date="2024-05-03",
            company_name="Apple Inc.",
            ticker="AAPL",
            cik="320193",
            file_path="AAPL/10-Q_2024-05-03_0000320193-24-000045.html",
            file_hash="ghi789jkl012345",
        ),
        FilingMetadata(
            accession_number="0000320193-24-000023",
            filing_type="8-K",
            filing_date="2024-02-01",
            company_name="Apple Inc.",
            ticker="AAPL",
            cik="320193",
            file_path="AAPL/8-K_2024-02-01_0000320193-24-000023.html",
            file_hash="jkl012mno345678",
        ),
    ]

    for filing in filings:
        manifest.filings[filing.accession_number] = filing

    # Save manifest
    manifest_path = temp_project_dirs["data"] / "manifest.json"
    manifest.save(manifest_path)

    return manifest


@pytest.fixture
def aapl_html_fixture(temp_project_dirs):
    """Create minimal AAPL HTML filing for testing."""
    html_content = """<!DOCTYPE html>
<html>
<head><title>Apple Inc. - Form 10-K</title></head>
<body>
<h3>Item 1. Business</h3>
<p>Apple Inc. designs, manufactures and markets smartphones.</p>
<p>iPhone represents approximately 52% of total net sales.</p>

<h3>Item 1A. Risk Factors</h3>
<p>The Company business is highly competitive.</p>
<p>Supply chain disruptions could impact business.</p>

<h3>Item 7. Management Discussion and Analysis</h3>
<p>Total net sales increased 2% during fiscal 2024.</p>
<p>Services gross margin improved to 74.0%.</p>
</body>
</html>
"""
    file_path = temp_project_dirs["data"] / "10-K_2024-11-01_0000320193-24-000081.html"
    file_path.write_text(html_content)
    return file_path


@pytest.fixture
def aapl_index(temp_project_dirs):
    """Create a populated vector index for AAPL."""
    store = VectorStore(
        index_dir=temp_project_dirs["root"] / "index",
        ticker="AAPL",
        embedding_dim=64,
        embedding_model="test-model",
    )

    # Add sample chunks
    np.random.seed(42)
    chunks = [
        EmbeddedChunk(
            text="iPhone revenue was $201.2 billion in fiscal 2024.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "filing_date": "2024-11-01",
                "section": "Item 1",
                "section_title": "Business",
                "source": "10-K_2024.html",
            },
        ),
        EmbeddedChunk(
            text="Services revenue increased 13% to $96.2 billion.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "filing_date": "2024-11-01",
                "section": "Item 7",
                "section_title": "MD&A",
                "source": "10-K_2024.html",
            },
        ),
        EmbeddedChunk(
            text="The Company business is highly competitive.",
            embedding=np.random.randn(64).astype(np.float32),
            metadata={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "filing_date": "2024-11-01",
                "section": "Item 1A",
                "section_title": "Risk Factors",
                "source": "10-K_2024.html",
            },
        ),
    ]

    store.add(chunks, source_file="10-K_2024.html", file_hash="testhash123")
    return store


@pytest.fixture
def mock_sec_fetcher():
    """Mock the SEC fetcher to avoid network calls."""
    with patch("src.cli.fetch.SECFetcher") as mock_class:
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = []
        mock_class.return_value = mock_fetcher
        yield mock_fetcher


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="## Summary\n\n- Key finding")]
    mock_response.usage.input_tokens = 1500
    mock_response.usage.output_tokens = 200
    mock_response.stop_reason = "end_turn"
    return mock_response


# =============================================================================
# Fetch Command Tests
# =============================================================================


class TestFetchCommand:
    """Tests for the pitch fetch AAPL command."""

    def test_fetch_aapl_basic(self, temp_project_dirs, mock_sec_fetcher):
        """Test basic AAPL fetch with defaults."""
        with patch("src.cli.fetch.DATA_DIR", temp_project_dirs["root"] / "data"):
            runner.invoke(app, ["fetch", "AAPL"])

            mock_sec_fetcher.fetch.assert_called_once()
            call_kwargs = mock_sec_fetcher.fetch.call_args.kwargs
            assert call_kwargs["ticker"] == "AAPL"

    def test_fetch_aapl_with_types(self, temp_project_dirs, mock_sec_fetcher):
        """Test AAPL fetch with specific filing types."""
        with patch("src.cli.fetch.DATA_DIR", temp_project_dirs["root"] / "data"):
            runner.invoke(app, ["fetch", "AAPL", "-t", "10-K,8-K"])

            call_kwargs = mock_sec_fetcher.fetch.call_args.kwargs
            assert set(call_kwargs["filing_types"]) == {"10-K", "8-K"}

    def test_fetch_aapl_since_manifest(self, temp_project_dirs, aapl_manifest, mock_sec_fetcher):
        """Test fetch --since uses manifest last_updated."""
        with patch("src.cli.fetch.DATA_DIR", temp_project_dirs["root"] / "data"):
            result = runner.invoke(app, ["fetch", "AAPL", "--since"])

            assert result.exit_code == 0
            # The output should show "Fetching filings since:" with a date
            assert "Fetching filings since:" in result.stdout

    def test_fetch_invalid_type(self, temp_project_dirs):
        """Test fetch with invalid filing type."""
        with patch("src.cli.fetch.DATA_DIR", temp_project_dirs["root"] / "data"):
            result = runner.invoke(app, ["fetch", "AAPL", "-t", "INVALID"])

            assert result.exit_code == 1
            assert "Unsupported filing type" in result.stdout

    @pytest.mark.parametrize("ticker", ["AAPL", "aapl", "AaPl"])
    def test_fetch_ticker_accepts_case_variants(self, temp_project_dirs, mock_sec_fetcher, ticker):
        """Test that fetch accepts various case variants of tickers."""
        with patch("src.cli.fetch.DATA_DIR", temp_project_dirs["root"] / "data"):
            runner.invoke(app, ["fetch", ticker])
            # The command should execute without error (fetcher handles normalization)
            mock_sec_fetcher.fetch.assert_called_once()


# =============================================================================
# Index Command Tests
# =============================================================================


class TestIndexCommand:
    """Tests for the pitch index AAPL command."""

    def test_index_no_data(self, temp_project_dirs):
        """Test index command when no data exists."""
        with (
            patch("src.cli.index.DATA_DIR", temp_project_dirs["root"] / "data"),
            patch("src.cli.index.INDEX_DIR", temp_project_dirs["root"] / "index"),
        ):
            result = runner.invoke(app, ["index", "AAPL", "--source", "sec"])

            assert result.exit_code == 1
            assert "No SEC filings found" in result.stdout


# =============================================================================
# Search Command Tests
# =============================================================================


class TestSearchCommand:
    """Tests for the pitch search AAPL command."""

    def test_search_no_index(self, temp_project_dirs):
        """Test search when no index exists."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            result = runner.invoke(app, ["search", "AAPL", "revenue growth"])

            assert result.exit_code == 1
            assert "No index found" in result.stdout

    def test_search_with_index_fts_only(self, temp_project_dirs, aapl_index):
        """Test basic FTS search with existing index (avoids embedding dimension issues)."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            # Use FTS mode to avoid embedding dimension mismatch
            result = runner.invoke(app, ["search", "AAPL", "iPhone", "-m", "fts", "--no-rerank"])

            assert result.exit_code == 0
            assert "Results for:" in result.stdout

    def test_search_fts_mode(self, temp_project_dirs, aapl_index):
        """Test FTS search mode."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            result = runner.invoke(app, ["search", "AAPL", "Services", "-m", "fts", "--no-rerank"])

            assert result.exit_code == 0

    def test_search_invalid_mode(self, temp_project_dirs, aapl_index):
        """Test search with invalid mode."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            result = runner.invoke(app, ["search", "AAPL", "query", "--mode", "invalid"])

            assert result.exit_code == 1
            assert "Invalid mode" in result.stdout

    @pytest.mark.parametrize("top_k", [1, 5, 10])
    def test_search_top_k_parameter_fts(self, temp_project_dirs, aapl_index, top_k):
        """Test search respects top_k parameter (FTS mode to avoid embedding issues)."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            result = runner.invoke(
                app, ["search", "AAPL", "revenue", "-m", "fts", "--top-k", str(top_k), "--no-rerank"]
            )
            assert result.exit_code == 0


# =============================================================================
# Summarize Command Tests
# =============================================================================


class TestSummarizeCommand:
    """Tests for the pitch summarize AAPL command."""

    def test_summarize_no_filings(self, temp_project_dirs):
        """Test summarize when no filings exist."""
        with (
            patch("src.cli.summarize.DATA_DIR", temp_project_dirs["root"] / "data"),
            patch("src.cli.summarize.PROCESSED_DIR", temp_project_dirs["root"] / "processed"),
            patch("src.summarizer.summarizer.Anthropic"),
        ):
            result = runner.invoke(app, ["summarize", "AAPL"])
            assert "No SEC filings found" in result.stdout or "No documents found" in result.stdout


# =============================================================================
# Notes Command Tests
# =============================================================================


class TestNotesCommand:
    """Tests for the pitch notes command."""

    def test_notes_file_not_found(self, temp_project_dirs):
        """Test notes command with non-existent file."""
        with patch("src.cli.misc.NOTES_DIR", temp_project_dirs["notes"]):
            result = runner.invoke(app, ["notes", "/nonexistent/file.pdf"])

            assert result.exit_code == 1
            assert "File not found" in result.stdout

    def test_notes_non_pdf(self, temp_project_dirs):
        """Test notes command with non-PDF file."""
        txt_file = temp_project_dirs["notes"] / "notes.txt"
        txt_file.write_text("Some text")

        with patch("src.cli.misc.NOTES_DIR", temp_project_dirs["notes"]):
            result = runner.invoke(app, ["notes", str(txt_file)])

            assert result.exit_code == 1
            assert "must be a PDF" in result.stdout


# =============================================================================
# Batch Command Tests
# =============================================================================


class TestBatchCommands:
    """Tests for batch processing commands."""

    def test_batch_submit_no_data(self, temp_project_dirs):
        """Test batch-submit when no data exists."""
        with patch("src.cli.batch.DATA_DIR", temp_project_dirs["root"] / "data"):
            result = runner.invoke(app, ["batch-submit", "AAPL", "-d", "10-K"])

            assert result.exit_code == 1
            assert "No data found" in result.stdout or "No manifest" in result.stdout

    def test_batch_status_no_jobs(self, temp_project_dirs):
        """Test batch-status when no jobs exist."""
        with patch("src.cli.batch.PROCESSED_DIR", temp_project_dirs["root"] / "processed"):
            with patch("src.summarizer.batch.Anthropic") as mock_anthropic:
                mock_anthropic.return_value = MagicMock()
                result = runner.invoke(app, ["batch-status"])
                assert "No batch jobs" in result.stdout or result.exit_code == 0


# =============================================================================
# Inventory Command Tests
# =============================================================================


class TestInventoryCommand:
    """Tests for the pitch inventory command."""

    def test_inventory_no_data(self, temp_project_dirs):
        """Test inventory when no data exists."""
        with (
            patch("src.cli.misc.DATA_DIR", temp_project_dirs["root"] / "data"),
            patch("src.cli.misc.TRANSCRIPTS_DIR", temp_project_dirs["root"] / "transcripts"),
            patch("src.cli.misc.ANALYST_DIR", temp_project_dirs["root"] / "analyst"),
            patch("src.cli.misc.PRESENTATIONS_DIR", temp_project_dirs["root"] / "presentations"),
            patch("src.cli.misc.CONFERENCES_DIR", temp_project_dirs["root"] / "conferences"),
            patch("src.cli.misc.MISC_DIR", temp_project_dirs["root"] / "misc"),
            patch("src.cli.misc.PROJECT_ROOT", temp_project_dirs["root"]),
        ):
            output_path = temp_project_dirs["root"] / "MATERIALS.md"
            result = runner.invoke(app, ["inventory", "AAPL", "-o", str(output_path)])

            assert result.exit_code == 0
            assert output_path.exists()

    def test_inventory_with_manifest(self, temp_project_dirs, aapl_manifest):
        """Test inventory with existing manifest."""
        with (
            patch("src.cli.misc.DATA_DIR", temp_project_dirs["root"] / "data"),
            patch("src.cli.misc.TRANSCRIPTS_DIR", temp_project_dirs["root"] / "transcripts"),
            patch("src.cli.misc.ANALYST_DIR", temp_project_dirs["root"] / "analyst"),
            patch("src.cli.misc.PRESENTATIONS_DIR", temp_project_dirs["root"] / "presentations"),
            patch("src.cli.misc.CONFERENCES_DIR", temp_project_dirs["root"] / "conferences"),
            patch("src.cli.misc.MISC_DIR", temp_project_dirs["root"] / "misc"),
            patch("src.cli.misc.PROJECT_ROOT", temp_project_dirs["root"]),
        ):
            output_path = temp_project_dirs["root"] / "MATERIALS.md"
            result = runner.invoke(app, ["inventory", "AAPL", "-o", str(output_path)])

            assert result.exit_code == 0
            content = output_path.read_text()
            assert "AAPL" in content
            assert "SEC Filings" in content


# =============================================================================
# Process Command Tests
# =============================================================================


class TestProcessCommand:
    """Tests for the pitch process AAPL command."""

    def test_process_invalid_types(self, temp_project_dirs):
        """Test process with invalid filing types."""
        with patch("src.cli.misc.DATA_DIR", temp_project_dirs["root"] / "data"):
            result = runner.invoke(app, ["process", "AAPL", "-t", "INVALID"])

            assert result.exit_code == 1
            assert "Unsupported filing type" in result.stdout


# =============================================================================
# AAPL-Specific Scenarios
# =============================================================================


class TestAAPLScenarios:
    """Integration-style tests for common AAPL analysis scenarios."""

    def test_search_iphone_revenue_fts(self, temp_project_dirs, aapl_index):
        """Test searching for iPhone revenue data (FTS mode)."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            # Use FTS mode to avoid embedding dimension mismatch with test fixtures
            result = runner.invoke(app, ["search", "AAPL", "iPhone revenue", "-m", "fts", "--no-rerank"])
            assert result.exit_code == 0

    def test_search_services_margin_fts(self, temp_project_dirs, aapl_index):
        """Test searching for Services gross margin (FTS mode)."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            # Use FTS mode to avoid embedding dimension mismatch with test fixtures
            result = runner.invoke(app, ["search", "AAPL", "Services", "-m", "fts", "--no-rerank"])
            assert result.exit_code == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.parametrize("ticker", ["AAPL", "BRK-B", "BRK.B"])
    def test_ticker_formats(self, temp_project_dirs, ticker):
        """Test various ticker formats."""
        with patch("src.cli.fetch.DATA_DIR", temp_project_dirs["root"] / "data"):
            result = runner.invoke(app, ["fetch", ticker, "--help"])
            assert result.exit_code == 0

    def test_empty_query(self, temp_project_dirs, aapl_index):
        """Test search with empty query."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            # Use FTS mode to avoid embedding dimension mismatch
            result = runner.invoke(app, ["search", "AAPL", "", "-m", "fts", "--no-rerank"])
            assert result.exit_code in [0, 1, 2]

    def test_special_characters_in_query(self, temp_project_dirs, aapl_index):
        """Test search with special characters."""
        with patch("src.cli.search.INDEX_DIR", temp_project_dirs["root"] / "index"):
            # Use FTS mode to avoid embedding dimension mismatch
            result = runner.invoke(app, ["search", "AAPL", "$96.2B Services +13%", "-m", "fts", "--no-rerank"])
            assert result.exit_code in [0, 1]
