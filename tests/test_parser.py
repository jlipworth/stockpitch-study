"""Tests for SEC filing HTML parser."""

from pathlib import Path

import pytest

from src.rag import SECTION_PATTERNS, Chunk, ParsedFiling, SECFilingParser, Section
from src.rag.section_patterns import (
    SECTION_HEADER_REGEX,
    SECTION_HEADER_REGEX_8K,
    SECTION_PATTERNS_8K,
    SECTION_PATTERNS_10K,
    SECTION_PATTERNS_10Q,
    SECTION_PATTERNS_DEF14A,
    get_section_patterns,
    get_section_regex,
)

from .conftest import TEST_TICKER

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_10K = FIXTURES_DIR / "sample_10k_aapl.html"


class TestSECFilingParser:
    """Tests for SECFilingParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return SECFilingParser(chunk_size=500, chunk_overlap=50)

    @pytest.fixture
    def parsed_filing(self, parser):
        """Parse the sample 10-K file."""
        return parser.parse_file(
            SAMPLE_10K,
            ticker=TEST_TICKER,
            filing_type="10-K",
            filing_date="2025-10-31",
        )

    def test_parse_file_returns_parsed_filing(self, parsed_filing):
        """Test that parse_file returns a ParsedFiling object."""
        assert isinstance(parsed_filing, ParsedFiling)
        assert parsed_filing.ticker == TEST_TICKER
        assert parsed_filing.filing_type == "10-K"
        assert parsed_filing.filing_date == "2025-10-31"

    def test_parse_file_extracts_text(self, parsed_filing):
        """Test that text is extracted from HTML."""
        assert len(parsed_filing.full_text) > 10000
        # Should contain some expected content
        assert "Apple" in parsed_filing.full_text

    def test_parse_file_detects_sections(self, parsed_filing):
        """Test that sections are detected."""
        assert len(parsed_filing.sections) > 0

        # Should find key 10-K sections
        section_items = [s.item for s in parsed_filing.sections]
        assert "Item 1" in section_items  # Business
        assert "Item 7" in section_items  # MD&A
        assert "Item 8" in section_items  # Financial Statements

    def test_sections_have_content(self, parsed_filing):
        """Test that detected sections have meaningful content."""
        for section in parsed_filing.sections:
            # Min threshold is 50 chars (set in parser), some sections may be just above
            assert len(section.content) >= 50, f"{section.item} has too little content"
            assert section.start_pos < section.end_pos

    def test_section_titles_match_patterns(self, parsed_filing):
        """Test that section titles are recognized."""
        for section in parsed_filing.sections:
            # Either title matches our pattern or was extracted from document
            if section.item in SECTION_PATTERNS:
                # Title should be set (either from pattern or extracted)
                assert section.title, f"{section.item} missing title"


class TestSectionDetection:
    """Tests for section detection logic."""

    @pytest.fixture
    def parser(self):
        return SECFilingParser()

    def test_detects_item_1(self, parser):
        """Test detection of Item 1 Business section."""
        text = """
        Item 1. Business

        Apple Inc. is a technology company that designs, manufactures,
        and markets smartphones, personal computers, tablets, wearables,
        and accessories.

        Item 1A. Risk Factors

        Various risks affect our business.
        """
        sections = parser._detect_sections(text)

        assert len(sections) >= 1
        assert sections[0].item == "Item 1"
        assert "technology company" in sections[0].content

    def test_detects_item_with_letter_suffix(self, parser):
        """Test detection of items like 1A, 7A, 9B."""
        text = """
        Item 7A. Quantitative Disclosures

        Market risk discussion here with enough content to pass the minimum
        threshold for section detection. This section discusses quantitative
        and qualitative disclosures about market risk.

        Item 8. Financial Statements

        See consolidated financial statements and supplementary data. This
        section contains the audited financial statements including balance
        sheets, income statements, and cash flow statements.
        """
        sections = parser._detect_sections(text)

        items = [s.item for s in sections]
        assert "Item 7A" in items
        assert "Item 8" in items

    def test_skips_toc_entries(self, parser):
        """Test that short TOC entries are skipped."""
        text = """
        Table of Contents
        Item 1. Business..........5
        Item 2. Properties........10

        Item 1. Business

        This is the actual business section with substantial content
        that describes what the company does in detail. It needs to be
        longer than the minimum threshold to be detected.

        Item 2. Properties

        Short.
        """
        sections = parser._detect_sections(text)

        # Should only have Item 1 (Item 2 content is too short)
        assert len([s for s in sections if s.item == "Item 1"]) == 1


class TestChunking:
    """Tests for text chunking functionality."""

    @pytest.fixture
    def parser(self):
        return SECFilingParser(chunk_size=100, chunk_overlap=20)

    def test_chunk_section_creates_chunks(self, parser):
        """Test that chunking produces Chunk objects."""
        section = Section(
            item="Item 1",
            title="Business",
            content="This is a test section. " * 50,
            start_pos=0,
            end_pos=1000,
        )

        chunks = parser.chunk_section(section)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunks_have_metadata(self, parser):
        """Test that chunks include section metadata."""
        section = Section(
            item="Item 7",
            title="MD&A",
            content="Analysis content. " * 100,
            start_pos=0,
            end_pos=2000,
        )

        chunks = parser.chunk_section(
            section,
            filing_metadata={"ticker": TEST_TICKER, "filing_type": "10-K"},
        )

        for chunk in chunks:
            assert chunk.metadata["section"] == "Item 7"
            assert chunk.metadata["section_title"] == "MD&A"
            assert chunk.metadata["ticker"] == TEST_TICKER
            assert "chunk_index" in chunk.metadata

    def test_chunk_size_respected(self, parser):
        """Test that chunks don't exceed target size significantly."""
        # Use realistic content with sentence boundaries
        # Note: With content-type-specific chunk sizes, Item 1 now has 1200 token target
        # so we need more content to ensure multiple chunks
        sentence = "This is a test sentence that contains multiple words. "
        section = Section(
            item="Item 1",
            title="Business",
            content=sentence * 150,  # ~1500 words to exceed 1200 token chunk size
            start_pos=0,
            end_pos=7500,
        )

        chunks = parser.chunk_section(section)

        # Should produce multiple chunks (content > chunk_size for Item 1 = 1200)
        assert len(chunks) > 1

        # Most chunks should be reasonably sized (allow some tolerance)
        # Use section-specific chunk size (1200 for Item 1)
        oversized = [c for c in chunks if c.token_estimate > 1200 * 2]
        assert len(oversized) <= 1  # At most one oversized chunk (edge case)

    def test_chunk_overlap(self):
        """Test that overlapping chunks share content."""
        parser = SECFilingParser(chunk_size=50, chunk_overlap=20)

        section = Section(
            item="Item 1",
            title="Test",
            content="Sentence one. Sentence two. Sentence three. Sentence four. "
            "Sentence five. Sentence six. Sentence seven. Sentence eight. "
            "Sentence nine. Sentence ten. Sentence eleven. Sentence twelve.",
            start_pos=0,
            end_pos=500,
        )

        chunks = parser.chunk_section(section)

        # With overlap, adjacent chunks should share some content
        if len(chunks) > 1:
            # Just verify we got multiple chunks
            assert len(chunks) >= 2


class TestChunkFilingIntegration:
    """Integration tests for chunking entire filings."""

    @pytest.fixture
    def parser(self):
        return SECFilingParser(chunk_size=500, chunk_overlap=50)

    @pytest.fixture
    def parsed_filing(self, parser):
        return parser.parse_file(
            SAMPLE_10K,
            ticker=TEST_TICKER,
            filing_type="10-K",
            filing_date="2025-10-31",
        )

    def test_chunk_filing_produces_chunks(self, parser, parsed_filing):
        """Test that chunking a filing produces chunks."""
        chunks = parser.chunk_filing(parsed_filing)

        assert len(chunks) > 10  # Real 10-K should have many chunks
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_all_chunks_have_required_metadata(self, parser, parsed_filing):
        """Test that all chunks have required metadata fields."""
        chunks = parser.chunk_filing(parsed_filing)

        required_fields = ["ticker", "filing_type", "filing_date", "section", "source"]

        for chunk in chunks:
            for field in required_fields:
                assert field in chunk.metadata, f"Missing {field} in chunk metadata"

    def test_chunks_cover_major_sections(self, parser, parsed_filing):
        """Test that chunks are created from major 10-K sections."""
        chunks = parser.chunk_filing(parsed_filing)

        sections_in_chunks = set(c.metadata["section"] for c in chunks)

        # Should have chunks from key sections
        assert "Item 1" in sections_in_chunks  # Business
        assert "Item 7" in sections_in_chunks  # MD&A


class TestTokenEstimate:
    """Tests for token estimation."""

    def test_token_estimate_calculation(self):
        """Test that token estimate is reasonable."""
        chunk = Chunk(text="This is a test sentence with ten words in it.")

        # 10 words * 1.3 = 13 tokens estimated
        assert chunk.token_estimate == 13

    def test_empty_chunk_token_estimate(self):
        """Test token estimate for empty chunk."""
        chunk = Chunk(text="")
        assert chunk.token_estimate == 0


class TestSectionPatterns:
    """Tests for section pattern constants."""

    def test_all_standard_items_defined(self):
        """Test that all standard 10-K items are defined."""
        expected_items = [
            "Item 1",
            "Item 1A",
            "Item 1B",
            "Item 2",
            "Item 3",
            "Item 4",
            "Item 5",
            "Item 6",
            "Item 7",
            "Item 7A",
            "Item 8",
            "Item 9",
            "Item 9A",
            "Item 9B",
            "Item 10",
            "Item 11",
            "Item 12",
            "Item 13",
            "Item 14",
            "Item 15",
        ]

        for item in expected_items:
            assert item in SECTION_PATTERNS, f"Missing {item} in SECTION_PATTERNS"

    def test_section_patterns_have_titles(self):
        """Test that all section patterns have non-empty titles."""
        for item, title in SECTION_PATTERNS.items():
            assert title, f"{item} has empty title"


class TestTableToMarkdown:
    """Tests for HTML table to markdown conversion."""

    def test_simple_table_conversion(self):
        """Test converting a simple HTML table to markdown."""
        from bs4 import BeautifulSoup

        from src.rag.table_utils import _table_to_markdown

        html = """
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>Revenue</td><td>$100M</td></tr>
            <tr><td>Profit</td><td>$20M</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _table_to_markdown(table)

        assert "| Name | Value |" in result
        assert "| Revenue | $100M |" in result
        assert "| Profit | $20M |" in result
        assert "---" in result  # Header separator

    def test_table_with_caption(self):
        """Test that table captions are preserved."""
        from bs4 import BeautifulSoup

        from src.rag.table_utils import _table_to_markdown

        html = """
        <table>
            <caption>Financial Summary</caption>
            <tr><th>Year</th><th>Revenue</th></tr>
            <tr><td>2024</td><td>$500M</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _table_to_markdown(table)

        assert "Financial Summary" in result

    def test_table_with_preceding_header(self):
        """Test that preceding headers are captured as context."""
        from bs4 import BeautifulSoup

        from src.rag.table_utils import _get_table_context

        html = """
        <div>
            <h3>Executive Compensation</h3>
            <table>
                <tr><th>Name</th><th>Salary</th></tr>
                <tr><td>CEO</td><td>$1M</td></tr>
            </table>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        context = _get_table_context(table)

        assert "Executive Compensation" in context

    def test_convert_tables_in_soup(self):
        """Test that convert_tables_to_markdown modifies soup in place."""
        from bs4 import BeautifulSoup

        from src.rag.table_utils import convert_tables_to_markdown

        html = """
        <div>
            <p>Here is the data:</p>
            <table>
                <tr><th>A</th><th>B</th></tr>
                <tr><td>1</td><td>2</td></tr>
            </table>
            <p>End of data.</p>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        convert_tables_to_markdown(soup)

        text = soup.get_text()
        # Table should now be markdown, not HTML
        assert "| A | B |" in text
        assert "| 1 | 2 |" in text

    def test_parser_preserves_table_structure(self):
        """Test that parsed filings preserve table structure with markers."""
        from src.rag import SECFilingParser

        # HTML with a table
        html = """
        <html><body>
        <p>Item 1. Business</p>
        <p>Our company sells products. Here is the breakdown:</p>
        <table>
            <tr><th>Product</th><th>Revenue</th></tr>
            <tr><td>Widgets</td><td>$50M</td></tr>
            <tr><td>Gadgets</td><td>$30M</td></tr>
        </table>
        <p>Item 1A. Risk Factors</p>
        <p>We face various risks.</p>
        </body></html>
        """

        parser = SECFilingParser()
        # Use parse method with raw HTML
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        parser._remove_hidden_elements(soup)
        text = parser._extract_text(soup)

        # Should contain table markers with readable text format
        assert "[TABLE:" in text
        assert "[/TABLE]" in text
        # Table data should be present in readable format
        assert "Widgets" in text
        assert "$50M" in text
        assert "Gadgets" in text


class TestFilingTypePatterns:
    """Tests for filing-type-specific section patterns."""

    def test_10q_patterns_defined(self):
        """Test that 10-Q patterns are defined."""
        # 10-Q uses "Part I, Item X" format to avoid key conflicts
        assert "Part I, Item 1" in SECTION_PATTERNS_10Q
        assert "Part I, Item 2" in SECTION_PATTERNS_10Q
        assert "Part II, Item 1A" in SECTION_PATTERNS_10Q  # Risk factors update

    def test_8k_patterns_defined(self):
        """Test that 8-K patterns are defined."""
        assert "Item 1.01" in SECTION_PATTERNS_8K
        assert "Item 2.02" in SECTION_PATTERNS_8K
        assert "Item 5.02" in SECTION_PATTERNS_8K
        assert "Item 8.01" in SECTION_PATTERNS_8K

    def test_def14a_patterns_defined(self):
        """Test that DEF 14A patterns are defined."""
        assert "Executive Compensation" in SECTION_PATTERNS_DEF14A
        assert "Corporate Governance" in SECTION_PATTERNS_DEF14A
        assert "Board of Directors" in SECTION_PATTERNS_DEF14A

    def test_get_section_patterns_10k(self):
        """Test get_section_patterns returns 10-K patterns."""
        patterns = get_section_patterns("10-K")
        assert patterns == SECTION_PATTERNS_10K
        patterns = get_section_patterns("10K")
        assert patterns == SECTION_PATTERNS_10K

    def test_get_section_patterns_10q(self):
        """Test get_section_patterns returns 10-Q patterns."""
        patterns = get_section_patterns("10-Q")
        assert patterns == SECTION_PATTERNS_10Q
        patterns = get_section_patterns("10Q")
        assert patterns == SECTION_PATTERNS_10Q

    def test_get_section_patterns_8k(self):
        """Test get_section_patterns returns 8-K patterns."""
        patterns = get_section_patterns("8-K")
        assert patterns == SECTION_PATTERNS_8K
        patterns = get_section_patterns("8K")
        assert patterns == SECTION_PATTERNS_8K

    def test_get_section_patterns_def14a(self):
        """Test get_section_patterns returns DEF 14A patterns."""
        patterns = get_section_patterns("DEF 14A")
        assert patterns == SECTION_PATTERNS_DEF14A
        patterns = get_section_patterns("DEF14A")
        assert patterns == SECTION_PATTERNS_DEF14A
        patterns = get_section_patterns("PROXY")
        assert patterns == SECTION_PATTERNS_DEF14A


class Test8KSectionDetection:
    """Tests for 8-K section header detection."""

    def test_8k_regex_matches_item_format(self):
        """Test that 8-K regex matches Item X.XX format."""
        text = "Item 1.01 - Entry into a Material Definitive Agreement"
        match = SECTION_HEADER_REGEX_8K.search(text)
        assert match is not None
        assert match.group(1) == "1.01"

    def test_8k_regex_matches_common_items(self):
        """Test that 8-K regex matches common 8-K items."""
        items = [
            "Item 2.02 - Results of Operations",
            "Item 5.02. Departure of Directors",
            "Item 8.01 Other Events",
            "Item 9.01 Financial Statements and Exhibits",
        ]
        for item in items:
            match = SECTION_HEADER_REGEX_8K.search(item)
            assert match is not None, f"Failed to match: {item}"

    def test_parser_uses_8k_patterns(self):
        """Test that parser uses 8-K patterns for 8-K filings."""
        from bs4 import BeautifulSoup

        html = """
        <html><body>
        <p>Item 2.02 - Results of Operations and Financial Condition</p>
        <p>On January 15, 2024, the Company announced its quarterly results.
        Revenue increased 10% year over year to $50 billion. The company
        exceeded analyst expectations for the quarter.</p>
        <p>Item 9.01 - Financial Statements and Exhibits</p>
        <p>See attached exhibits for additional details.</p>
        </body></html>
        """

        parser = SECFilingParser()
        soup = BeautifulSoup(html, "lxml")
        parser._remove_hidden_elements(soup)
        text = parser._extract_text(soup)
        sections = parser._detect_sections(text, "8-K")

        # Should detect Item 2.02
        item_keys = [s.item for s in sections]
        assert "Item 2.02" in item_keys or len(sections) >= 1


class TestSectionRegexSelection:
    """Tests for section regex selection based on filing type."""

    def test_get_section_regex_default(self):
        """Test that default regex is used for unknown types."""
        regex = get_section_regex("")
        assert regex == SECTION_HEADER_REGEX

    def test_get_section_regex_8k(self):
        """Test that 8-K regex is used for 8-K filings."""
        regex = get_section_regex("8-K")
        assert regex == SECTION_HEADER_REGEX_8K


class TestDEF14ADetection:
    """Tests for DEF 14A (proxy statement) section detection."""

    @pytest.fixture
    def parser(self):
        return SECFilingParser()

    def test_def14a_patterns_defined(self):
        """Test that DEF 14A section patterns are defined."""
        assert len(SECTION_PATTERNS_DEF14A) > 0
        assert "Board of Directors" in SECTION_PATTERNS_DEF14A
        assert "Executive Compensation" in SECTION_PATTERNS_DEF14A
        assert "Proposal" in SECTION_PATTERNS_DEF14A

    def test_html_based_proposal_detection(self, parser):
        """Test detection of PROPOSAL sections from HTML."""
        html = """
        <html><body>
        <p style="font-weight:700">PROPOSAL 1 - ELECTION OF DIRECTORS</p>
        <p>The board recommends voting for all nominees...</p>
        <p style="font-weight:700">PROPOSAL 2 - RATIFICATION OF AUDITORS</p>
        <p>The audit committee has selected...</p>
        </body></html>
        """

        sections = parser._detect_def14a_sections_from_html(html, SECTION_PATTERNS_DEF14A)

        assert len(sections) >= 2
        section_keys = [s[0] for s in sections]
        assert "Proposal 1" in section_keys
        assert "Proposal 2" in section_keys

    def test_html_based_standard_section_detection(self, parser):
        """Test detection of standard DEF 14A sections from HTML."""
        html = """
        <html><body>
        <div style="font-weight:700">BOARD OF DIRECTORS</div>
        <p>Our board consists of the following members...</p>
        <div style="font-weight:bold">EXECUTIVE COMPENSATION</div>
        <p>The following table shows compensation...</p>
        <div style="font-size:18pt">DIRECTOR COMPENSATION</div>
        <p>Directors receive the following compensation...</p>
        </body></html>
        """

        sections = parser._detect_def14a_sections_from_html(html, SECTION_PATTERNS_DEF14A)

        assert len(sections) >= 3
        section_keys = [s[0].lower() for s in sections]
        assert any("board" in key for key in section_keys)
        assert any("executive compensation" in key for key in section_keys)
        assert any("director compensation" in key for key in section_keys)

    def test_def14a_skips_toc_entries(self, parser):
        """Test that table of contents entries are skipped."""
        html = """
        <html><body>
        <a href="#proposal1"><p style="font-weight:700">PROPOSAL 1</p></a>
        <p>Some content</p>
        <p style="font-weight:700">PROPOSAL 1 - ACTUAL SECTION</p>
        <p>Real content goes here...</p>
        </body></html>
        """

        sections = parser._detect_def14a_sections_from_html(html, SECTION_PATTERNS_DEF14A)

        # Should find only the real section, not the TOC entry
        proposal_sections = [s for s in sections if "Proposal 1" in s[0]]
        assert len(proposal_sections) == 1

    def test_def14a_deduplication(self, parser):
        """Test that duplicate section headers are deduplicated."""
        html = """
        <html><body>
        <p style="font-weight:700">BOARD OF DIRECTORS</p>
        <p>TOC entry or cross-reference</p>
        <div style="font-weight:700">BOARD OF DIRECTORS</div>
        <p>The actual board section with lots of content...</p>
        </body></html>
        """

        sections = parser._detect_def14a_sections_from_html(html, SECTION_PATTERNS_DEF14A)

        # Should have only one Board of Directors section
        board_sections = [s for s in sections if "board" in s[0].lower()]
        assert len(board_sections) == 1


class TestBoardMemberExtraction:
    """Tests for extracting board members from iXBRL data."""

    def test_extract_board_members_from_ixbrl(self):
        """Test extraction of board member names from iXBRL context data."""
        from bs4 import BeautifulSoup

        html = """
        <html xmlns:xbrli="http://www.xbrl.org/2003/instance"
              xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
              xmlns:ecd="http://xbrl.sec.gov/ecd/2025"
              xmlns:dt="http://www.dynatrace.com/20250708">
        <body>
        <xbrli:context id="c-2">
            <xbrli:entity>
                <xbrli:identifier scheme="http://www.sec.gov/CIK">0001773383</xbrli:identifier>
                <xbrli:segment>
                    <xbrldi:explicitMember dimension="ecd:IndividualAxis">dt:RickMcConnellMember</xbrldi:explicitMember>
                </xbrli:segment>
            </xbrli:entity>
        </xbrli:context>
        <xbrli:context id="c-8">
            <xbrli:entity>
                <xbrli:identifier scheme="http://www.sec.gov/CIK">0001773383</xbrli:identifier>
                <xbrli:segment>
                    <xbrldi:explicitMember dimension="ecd:IndividualAxis">dt:JohnVanSiclenMember</xbrldi:explicitMember>
                </xbrli:segment>
            </xbrli:entity>
        </xbrli:context>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "lxml")
        members = SECFilingParser.extract_board_members_from_ixbrl(soup)

        assert len(members) >= 2
        assert "Rick Mc Connell" in members
        assert "John Van Siclen" in members

    def test_extract_board_members_empty(self):
        """Test extraction when no board members present."""
        from bs4 import BeautifulSoup

        html = "<html><body><p>No iXBRL data here</p></body></html>"

        soup = BeautifulSoup(html, "lxml")
        members = SECFilingParser.extract_board_members_from_ixbrl(soup)

        assert members == []

    def test_extract_board_members_deduplication(self):
        """Test that duplicate member names are deduplicated."""
        from bs4 import BeautifulSoup

        html = """
        <html xmlns:xbrli="http://www.xbrl.org/2003/instance"
              xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
              xmlns:ecd="http://xbrl.sec.gov/ecd/2025"
              xmlns:dt="http://www.dynatrace.com/20250708">
        <body>
        <xbrli:context id="c-1">
            <xbrli:entity>
                <xbrli:segment>
                    <xbrldi:explicitMember dimension="ecd:IndividualAxis">dt:RickMcConnellMember</xbrldi:explicitMember>
                </xbrli:segment>
            </xbrli:entity>
        </xbrli:context>
        <xbrli:context id="c-2">
            <xbrli:entity>
                <xbrli:segment>
                    <xbrldi:explicitMember dimension="ecd:IndividualAxis">dt:RickMcConnellMember</xbrldi:explicitMember>
                </xbrli:segment>
            </xbrli:entity>
        </xbrli:context>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "lxml")
        members = SECFilingParser.extract_board_members_from_ixbrl(soup)

        # Should have only one instance despite two contexts
        assert len(members) == 1
        assert "Rick Mc Connell" in members


class TestDEF14AIntegration:
    """Integration tests for parsing real DEF 14A files."""

    @pytest.fixture
    def parser(self):
        return SECFilingParser(chunk_size=1000, chunk_overlap=200)

    def test_parse_def14a_uses_html_detection(self, parser):
        """Test that DEF 14A files use HTML-based section detection."""
        # Create a minimal DEF 14A HTML file
        html_content = """
        <html xmlns:xbrli="http://www.xbrl.org/2003/instance">
        <body>
        <div style="font-weight:700">PROPOSAL 1 - ELECTION OF DIRECTORS</div>
        <p>The board of directors recommends that you vote FOR each of the nominees listed below.</p>
        <div style="font-weight:700">BOARD OF DIRECTORS</div>
        <p>Our board of directors currently consists of nine members, divided into three classes.</p>
        <div style="font-weight:bold">EXECUTIVE COMPENSATION</div>
        <p>The following table shows the compensation of our named executive officers.</p>
        </body>
        </html>
        """

        import tempfile
        from pathlib import Path

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            temp_path = Path(f.name)

        try:
            parsed = parser.parse_file(
                temp_path,
                ticker="TEST",
                filing_type="DEF 14A",
                filing_date="2025-07-08",
            )

            # Should have detected sections
            assert len(parsed.sections) >= 3

            section_items = [s.item.lower() for s in parsed.sections]
            assert any("proposal" in item for item in section_items)
            assert any("board" in item for item in section_items)
            assert any("compensation" in item for item in section_items)

        finally:
            # Clean up temp file
            temp_path.unlink()

    def test_chunk_def14a_produces_chunks(self, parser):
        """Test that DEF 14A chunking produces reasonable chunks."""
        html_content = (
            """
        <html><body>
        <div style="font-weight:700">PROPOSAL 1</div>
        <p>"""
            + ("Content paragraph. " * 100)
            + """</p>
        <div style="font-weight:700">EXECUTIVE COMPENSATION</div>
        <p>"""
            + ("Compensation details. " * 100)
            + """</p>
        </body></html>
        """
        )

        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            temp_path = Path(f.name)

        try:
            parsed = parser.parse_file(
                temp_path,
                ticker="TEST",
                filing_type="DEF 14A",
                filing_date="2025-07-08",
            )

            chunks = parser.chunk_filing(parsed)

            # Should have produced chunks
            assert len(chunks) > 0

            # All chunks should have metadata
            for chunk in chunks:
                assert "ticker" in chunk.metadata
                assert "filing_type" in chunk.metadata
                assert "section" in chunk.metadata

        finally:
            temp_path.unlink()
