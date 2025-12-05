"""Tests for the summarizer module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag import Section
from src.summarizer.summarizer import (
    DEFAULT_SUMMARY_MODEL,
    SECTION_WEIGHTS,
    FilingSummary,
    SectionSummary,
    Summarizer,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_section():
    """Create a sample section for testing."""
    return Section(
        item="Item 1A",
        title="Risk Factors",
        content="The company faces various risks including market competition, regulatory changes, and economic uncertainty. These factors could materially affect our business operations and financial results.",
        start_pos=0,
        end_pos=200,
    )


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text="## Summary\n\n- Key risk: market competition\n- Regulatory concerns noted")
    ]
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 50
    return mock_response


class TestSectionSummary:
    """Tests for SectionSummary dataclass."""

    def test_section_summary_creation(self):
        summary = SectionSummary(
            section_id="Item 1A",
            section_title="Risk Factors",
            summary="Key risks include competition.",
            weight="high",
            input_tokens=100,
            output_tokens=20,
        )

        assert summary.section_id == "Item 1A"
        assert summary.section_title == "Risk Factors"
        assert summary.weight == "high"
        assert "competition" in summary.summary


class TestFilingSummary:
    """Tests for FilingSummary dataclass."""

    def test_filing_summary_creation(self):
        sections = [
            SectionSummary("Item 1", "Business", "Summary of business", "high", 100, 20),
            SectionSummary("Item 1A", "Risk Factors", "Key risks", "high", 100, 20),
        ]
        summary = FilingSummary(
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            source_path="/path/to/filing.html",
            executive_summary="Company is performing well.",
            section_summaries=sections,
            total_input_tokens=300,
            total_output_tokens=60,
        )

        assert summary.ticker == "AAPL"
        assert summary.filing_type == "10-K"
        assert len(summary.section_summaries) == 2

    def test_to_markdown(self):
        sections = [
            SectionSummary("Item 1", "Business", "Business overview", "high", 100, 20),
        ]
        summary = FilingSummary(
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            source_path="/path/to/file.html",
            executive_summary="Executive overview here.",
            section_summaries=sections,
            total_input_tokens=200,
            total_output_tokens=40,
        )

        markdown = summary.to_markdown()

        assert "# AAPL 10-K Analysis" in markdown
        assert "Executive overview here" in markdown
        assert "Item 1: Business" in markdown
        assert "Business overview" in markdown

    def test_save(self, temp_dir):
        summary = FilingSummary(
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            source_path="/path/to/file.html",
            executive_summary="Test summary.",
            section_summaries=[],
            total_input_tokens=100,
            total_output_tokens=20,
        )

        output_path = temp_dir / "summary.md"
        summary.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "AAPL 10-K Analysis" in content


class TestSummarizer:
    """Tests for Summarizer class."""

    def test_init_requires_api_key(self):
        """Summarizer should fail without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                Summarizer()

    @patch("src.summarizer.summarizer.Anthropic")
    def test_init_with_api_key(self, mock_anthropic_class):
        """Summarizer should initialize with API key."""
        summarizer = Summarizer(api_key="test-key")
        assert summarizer.model == DEFAULT_SUMMARY_MODEL

    @patch("src.summarizer.summarizer.Anthropic")
    def test_summarize_section(self, mock_anthropic_class, sample_section, mock_anthropic_response):
        """Should summarize a section using Claude."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize_section(sample_section, "AAPL 10-K (2024-01-15)")

        assert isinstance(result, SectionSummary)
        assert result.section_id == "Item 1A"
        assert result.section_title == "Risk Factors"
        assert result.weight == "high"  # Item 1A is high weight
        assert result.input_tokens == 500
        assert result.output_tokens == 50

    @patch("src.summarizer.summarizer.Anthropic")
    def test_create_executive_summary(self, mock_anthropic_class, mock_anthropic_response):
        """Should create executive summary from section summaries."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        summarizer = Summarizer(api_key="test-key")

        section_summaries = [
            SectionSummary("Item 1", "Business", "Business desc", "high", 100, 20),
            SectionSummary("Item 1A", "Risk Factors", "Risk desc", "high", 100, 20),
        ]

        summary, input_tokens, output_tokens = summarizer.create_executive_summary(
            section_summaries, "AAPL 10-K (2024-01-15)"
        )

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert input_tokens == 500
        assert output_tokens == 50

    @patch("src.summarizer.summarizer.Anthropic")
    def test_get_output_path(self, mock_anthropic_class, temp_dir):
        """Should generate correct output path."""
        summarizer = Summarizer(api_key="test-key", processed_dir=temp_dir)

        path = summarizer.get_output_path("AAPL", "10-K", "2024-01-15")

        assert path.parent.name == "AAPL"
        assert "10-K" in path.name
        assert "2024-01-15" in path.name
        assert path.suffix == ".md"

    @patch("src.summarizer.summarizer.Anthropic")
    def test_get_output_path_no_dir(self, mock_anthropic_class):
        """Should raise error if processed_dir not set."""
        summarizer = Summarizer(api_key="test-key")

        with pytest.raises(ValueError, match="processed_dir not set"):
            summarizer.get_output_path("AAPL", "10-K", "2024-01-15")


class TestSummarizeFilingFunction:
    """Tests for convenience function."""

    @patch("src.summarizer.summarizer.Anthropic")
    @patch("src.summarizer.summarizer.SECFilingParser")
    def test_summarize_filing_basic(self, mock_parser_class, mock_anthropic_class, temp_dir, mock_anthropic_response):
        """Should summarize a filing and optionally save."""
        # Setup mock parser
        mock_parser = MagicMock()
        mock_parsed = MagicMock()
        mock_parsed.sections = [
            Section("Item 1", "Business", "Business content", 0, 100),
        ]
        mock_parser.parse_file.return_value = mock_parsed
        mock_parser_class.return_value = mock_parser

        # Setup mock Anthropic
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        # Create a dummy file
        dummy_file = temp_dir / "test.html"
        dummy_file.write_text("<html>Test</html>")

        output_path = temp_dir / "output.md"

        # Use sequential (parallel=False) since we're only mocking sync client
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize_filing(
            file_path=dummy_file,
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            parallel=False,
        )
        if output_path:
            result.save(output_path)

        assert isinstance(result, FilingSummary)
        assert result.ticker == "AAPL"
        assert output_path.exists()


class TestSummarizerWithProgress:
    """Tests for progress callback functionality."""

    @patch("src.summarizer.summarizer.Anthropic")
    @patch("src.summarizer.summarizer.SECFilingParser")
    def test_progress_callback_called(self, mock_parser_class, mock_anthropic_class, temp_dir, mock_anthropic_response):
        """Should call progress callback for each section."""
        # Setup mock parser
        mock_parser = MagicMock()
        mock_parsed = MagicMock()
        mock_parsed.sections = [
            Section("Item 1", "Business", "Content 1", 0, 100),
            Section("Item 1A", "Risk Factors", "Content 2", 100, 200),
        ]
        mock_parser.parse_file.return_value = mock_parsed
        mock_parser_class.return_value = mock_parser

        # Setup mock Anthropic
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        # Create dummy file
        dummy_file = temp_dir / "test.html"
        dummy_file.write_text("<html>Test</html>")

        summarizer = Summarizer(api_key="test-key")

        progress_calls = []

        def callback(current, total, section):
            progress_calls.append((current, total, section))

        # Use sequential (parallel=False) since we're only mocking sync client
        summarizer.summarize_filing(
            file_path=dummy_file,
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            progress_callback=callback,
            parallel=False,
        )

        # Should have calls for 2 sections + 1 executive summary
        assert len(progress_calls) == 3
        # Progress now includes weight info
        assert "Item 1" in progress_calls[0][2]
        assert "Item 1A" in progress_calls[1][2]
        assert "Executive" in progress_calls[2][2]


class TestSectionWeights:
    """Tests for section weight configuration."""

    def test_high_weight_sections(self):
        """High priority sections should be configured."""
        high_weight_sections = ["Item 1", "Item 1A", "Item 7"]
        for section in high_weight_sections:
            assert section in SECTION_WEIGHTS
            assert SECTION_WEIGHTS[section]["weight"] == "high"

    def test_low_weight_sections(self):
        """Boilerplate sections should be low weight."""
        low_weight_sections = ["Item 15", "Item 16", "Item 4"]
        for section in low_weight_sections:
            assert section in SECTION_WEIGHTS
            assert SECTION_WEIGHTS[section]["weight"] == "low"

    def test_max_tokens_vary_by_weight(self):
        """Higher weight sections should allow more output tokens."""
        high_tokens = SECTION_WEIGHTS["Item 7"]["max_tokens"]
        low_tokens = SECTION_WEIGHTS["Item 15"]["max_tokens"]
        assert high_tokens > low_tokens
