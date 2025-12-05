"""Tests for analyst report parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.analyst_parser import (
    ANALYST_SECTION_PATTERNS,
    BANK_PATTERNS,
    PRICE_TARGET_PATTERNS,
    RATING_PATTERNS,
    AnalystReportMetadata,
    AnalystReportParser,
    ParsedAnalystReport,
    ReportSection,
    parse_analyst_report,
)

# Sample analyst report text
SAMPLE_REPORT = """
Goldman Sachs
Equity Research

Apple Inc. (AAPL)
Rating: Buy
Price Target: $250

January 15, 2024

Investment Thesis

Apple remains our top pick in the technology sector. The company's strong
ecosystem and growing services revenue provide a durable competitive moat.
We see continued upside from the iPhone 16 cycle and potential AI integration
across the product lineup.

Key Highlights:
- Services revenue growing at 15% CAGR
- Strong installed base of 2B+ active devices
- Leading position in premium smartphones

Valuation

Our $250 price target is based on a DCF analysis with the following assumptions:
- Revenue CAGR of 8% through 2028
- Terminal growth rate of 3%
- WACC of 9%

We also cross-check with comparable analysis, where Apple trades at 28x forward
P/E versus peers at 25x.

Financial Estimates

FY24E Revenue: $400B (+5% YoY)
FY24E EPS: $7.00 (+10% YoY)
FY25E Revenue: $430B (+7.5% YoY)
FY25E EPS: $7.80 (+11% YoY)

Our estimates are above consensus due to stronger iPhone and Services growth.

Catalysts

Near-term catalysts include:
- Q1 earnings release (late January)
- WWDC 2024 with potential AI announcements
- iPhone 16 launch in September 2024

Longer-term catalysts:
- Vision Pro ecosystem development
- Automotive initiative progress

Risks to Our Thesis

Key risks to our bullish view include:
- Slower than expected iPhone replacement cycles
- Regulatory pressure in EU and China
- Competition in services from Google and Amazon
- Supply chain disruptions in China

Industry Overview

The global smartphone market is expected to grow modestly at 2-3% annually.
Apple's premium positioning insulates it from low-end competition. The
services market continues to expand, with Apple capturing significant
wallet share.

Disclosures

Goldman Sachs has performed investment banking services for Apple Inc.
in the past 12 months.
"""


class TestSectionPatterns:
    """Test section detection patterns."""

    def test_thesis_pattern(self):
        """Test Investment Thesis detection."""
        pattern = ANALYST_SECTION_PATTERNS["thesis"]
        assert pattern.search("Investment Thesis")
        assert pattern.search("Investment Summary")
        assert pattern.search("Key Takeaways")
        assert pattern.search("Our View")

    def test_valuation_pattern(self):
        """Test Valuation section detection."""
        pattern = ANALYST_SECTION_PATTERNS["valuation"]
        assert pattern.search("Valuation")
        assert pattern.search("Price Target Analysis")
        assert pattern.search("DCF Analysis")
        assert pattern.search("Sum-of-the-Parts")

    def test_estimates_pattern(self):
        """Test Estimates section detection."""
        pattern = ANALYST_SECTION_PATTERNS["estimates"]
        assert pattern.search("Financial Estimates")
        assert pattern.search("Earnings Model")
        assert pattern.search("Revenue Model")

    def test_catalysts_pattern(self):
        """Test Catalysts section detection."""
        pattern = ANALYST_SECTION_PATTERNS["catalysts"]
        assert pattern.search("Catalysts")
        assert pattern.search("Near-Term Catalysts")
        assert pattern.search("Key Events")

    def test_risks_pattern(self):
        """Test Risks section detection."""
        pattern = ANALYST_SECTION_PATTERNS["risks"]
        assert pattern.search("Risks")
        assert pattern.search("Risks to Our Thesis")
        assert pattern.search("Key Risks")
        assert pattern.search("Downside Risks")

    def test_industry_pattern(self):
        """Test Industry section detection."""
        pattern = ANALYST_SECTION_PATTERNS["industry"]
        assert pattern.search("Industry Overview")
        assert pattern.search("Competitive Landscape")
        assert pattern.search("Market Share")


class TestMetadataPatterns:
    """Test metadata extraction patterns."""

    def test_rating_patterns(self):
        """Test rating detection."""
        assert RATING_PATTERNS.search("Rating: Buy")
        assert RATING_PATTERNS.search("Overweight")
        assert RATING_PATTERNS.search("OUTPERFORM")
        assert RATING_PATTERNS.search("Hold")
        assert RATING_PATTERNS.search("Strong Buy")

    def test_price_target_patterns(self):
        """Test price target detection."""
        match = PRICE_TARGET_PATTERNS.search("Price Target: $250")
        assert match is not None
        assert match.group(1) == "250"

        match = PRICE_TARGET_PATTERNS.search("PT $180.50")
        assert match is not None

        match = PRICE_TARGET_PATTERNS.search("Target Price: $1,200")
        assert match is not None

    def test_bank_patterns(self):
        """Test bank/firm detection."""
        assert BANK_PATTERNS.search("Goldman Sachs")
        assert BANK_PATTERNS.search("Morgan Stanley")
        assert BANK_PATTERNS.search("JPMorgan")
        assert BANK_PATTERNS.search("J.P. Morgan")
        assert BANK_PATTERNS.search("Bank of America")
        assert BANK_PATTERNS.search("Jefferies")


class TestMetadataParsing:
    """Test metadata extraction from reports."""

    def test_parse_rating(self):
        """Test rating extraction."""
        parser = AnalystReportParser()
        metadata = parser._parse_metadata(SAMPLE_REPORT, [SAMPLE_REPORT], "")
        assert metadata.rating.lower() == "buy"

    def test_parse_price_target(self):
        """Test price target extraction."""
        parser = AnalystReportParser()
        metadata = parser._parse_metadata(SAMPLE_REPORT, [SAMPLE_REPORT], "")
        assert "250" in metadata.price_target

    def test_parse_firm(self):
        """Test firm extraction."""
        parser = AnalystReportParser()
        metadata = parser._parse_metadata(SAMPLE_REPORT, [SAMPLE_REPORT], "")
        assert "Goldman" in metadata.firm

    def test_parse_ticker(self):
        """Test ticker extraction."""
        parser = AnalystReportParser()
        metadata = parser._parse_metadata(SAMPLE_REPORT, [SAMPLE_REPORT], "")
        assert metadata.ticker == "AAPL"

    def test_provided_ticker_used(self):
        """Test that provided ticker is used."""
        parser = AnalystReportParser()
        metadata = parser._parse_metadata("No ticker here", ["No ticker here"], "MSFT")
        assert metadata.ticker == "MSFT"


class TestSectionDetection:
    """Test section detection from report text."""

    def test_detect_thesis_section(self):
        """Test thesis section detection."""
        parser = AnalystReportParser()
        sections = parser._detect_sections([SAMPLE_REPORT])

        thesis_sections = [s for s in sections if s.title == "thesis"]
        assert len(thesis_sections) >= 1
        assert "top pick" in thesis_sections[0].content.lower()

    def test_detect_valuation_section(self):
        """Test valuation section detection."""
        parser = AnalystReportParser()
        sections = parser._detect_sections([SAMPLE_REPORT])

        valuation_sections = [s for s in sections if s.title == "valuation"]
        assert len(valuation_sections) >= 1
        # Content should have valuation-related terms
        content = valuation_sections[0].content.lower()
        assert any(term in content for term in ["revenue", "target", "growth", "wacc"])

    def test_detect_risks_section(self):
        """Test risks section detection."""
        parser = AnalystReportParser()
        sections = parser._detect_sections([SAMPLE_REPORT])

        risks_sections = [s for s in sections if s.title == "risks"]
        assert len(risks_sections) >= 1

    def test_detect_catalysts_section(self):
        """Test catalysts section detection."""
        parser = AnalystReportParser()
        sections = parser._detect_sections([SAMPLE_REPORT])

        catalyst_sections = [s for s in sections if s.title == "catalysts"]
        assert len(catalyst_sections) >= 1


class TestFullParsing:
    """Test full report parsing with mocked PDF."""

    @patch("src.rag.analyst_parser.fitz")
    def test_parse_file(self, mock_fitz):
        """Test full file parsing with mocked PDF."""
        # Mock PDF extraction
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = SAMPLE_REPORT
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        pdf_path = Path("/tmp/fake_report.pdf")

        with patch.object(Path, "exists", return_value=True):
            parser = AnalystReportParser()
            result = parser.parse_file(pdf_path, "AAPL")

        assert isinstance(result, ParsedAnalystReport)
        assert result.metadata.ticker == "AAPL"
        assert result.metadata.rating.lower() == "buy"
        assert len(result.sections) >= 3

    @patch("src.rag.analyst_parser.fitz")
    def test_convenience_function(self, mock_fitz):
        """Test parse_analyst_report convenience function."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = SAMPLE_REPORT
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        pdf_path = Path("/tmp/fake_report.pdf")

        with patch.object(Path, "exists", return_value=True):
            result = parse_analyst_report(pdf_path, "AAPL")

        assert isinstance(result, ParsedAnalystReport)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self):
        """Test handling of empty report."""
        parser = AnalystReportParser()
        metadata = parser._parse_metadata("", [""], "")
        assert metadata.ticker == ""
        assert metadata.rating == ""

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        parser = AnalystReportParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/path.pdf"))

    def test_no_sections_detected(self):
        """Test report with no recognizable sections."""
        parser = AnalystReportParser()
        sections = parser._detect_sections(["Just some random text without headers."])
        assert sections == []


class TestDataClasses:
    """Test dataclass functionality."""

    def test_metadata_defaults(self):
        """Test AnalystReportMetadata default values."""
        metadata = AnalystReportMetadata()
        assert metadata.ticker == ""
        assert metadata.rating == ""
        assert metadata.firm == ""

    def test_section_creation(self):
        """Test ReportSection creation."""
        section = ReportSection(
            title="thesis",
            content="Our investment thesis...",
            page_num=1,
        )
        assert section.title == "thesis"
        assert section.page_num == 1
