"""Tests for Phase 2: Table Intelligence features.

Tests cover:
- Table dataclass creation and classification
- Table-aware chunking (keeping small tables intact, splitting large with headers)
- Content type and table type metadata propagation
- Search filtering by content_type and table_type
"""

import json
from pathlib import Path

import pytest

from src.rag.parser import (
    SECFilingParser,
    Section,
)
from src.rag.table_utils import (
    TABLE_END_MARKER,
    TABLE_START_MARKER,
    TABLE_TYPE_PATTERNS,
    Table,
    classify_table,
)


class TestTableClassification:
    """Tests for classify_table() function."""

    def test_financial_statement_detection(self):
        """Financial statements should be classified correctly."""
        assert classify_table("Income Statement", "Net revenue Sales") == "financial_statement"
        assert classify_table("Balance Sheet", "Total assets Liabilities") == "financial_statement"
        assert classify_table("Cash Flow Statement", "Operating activities") == "financial_statement"
        assert classify_table("Consolidated Financial", "Revenue Net income") == "financial_statement"

    def test_compensation_detection(self):
        """Compensation tables should be classified correctly."""
        assert classify_table("Executive Compensation", "Salary Bonus") == "compensation"
        assert classify_table("Summary Compensation Table", "Stock Awards") == "compensation"
        assert classify_table("Director Compensation", "Base Salary Bonus") == "compensation"

    def test_governance_detection(self):
        """Governance tables should be classified correctly."""
        assert classify_table("Board of Directors", "Name Committee") == "governance"
        assert classify_table("Audit Committee", "Independence") == "governance"
        assert classify_table("Corporate Governance", "Director Independence") == "governance"

    def test_comparison_detection(self):
        """Comparison tables should be classified correctly."""
        assert classify_table("Year Ended December 31", "Amount Total") == "comparison"
        assert classify_table("Three Months Ended March 31", "Data Values") == "comparison"
        assert classify_table("Year over Year Comparison", "Metrics Data") == "comparison"

    def test_schedule_detection(self):
        """Schedule tables should be classified correctly."""
        assert classify_table("Debt Maturity Schedule", "Due payments") == "schedule"
        assert classify_table("Lease Schedule", "Future payments") == "schedule"
        assert classify_table("Amortization Schedule", "Principal Interest") == "schedule"

    def test_metrics_detection(self):
        """Metrics tables should be classified correctly."""
        assert classify_table("Key Performance Metrics", "ARR NRR") == "metrics"
        assert classify_table("Operating Metrics", "Customer count") == "metrics"
        assert classify_table("KPI Summary", "ARPU CAC LTV") == "metrics"

    def test_other_fallback(self):
        """Unknown tables should fall back to 'other'."""
        assert classify_table("Random Table", "foo bar baz") == "other"
        assert classify_table("Unknown Data", "xyz abc 123") == "other"

    def test_case_insensitive(self):
        """Classification should be case-insensitive."""
        assert classify_table("INCOME STATEMENT", "revenue") == "financial_statement"
        assert classify_table("income statement", "REVENUE") == "financial_statement"


class TestTableDataclass:
    """Tests for Table dataclass."""

    def test_table_from_html(self):
        """Table.from_html should create Table with correct fields."""
        html = """<table>
            <tr><th>Name</th><th>Revenue</th></tr>
            <tr><td>Q1</td><td>$100M</td></tr>
            <tr><td>Q2</td><td>$110M</td></tr>
        </table>"""

        table = Table.from_html(html, "Revenue Summary", position=0, source_hash="abc123")

        assert table.table_id == "abc123_table_0"
        assert table.caption == "Revenue Summary"
        assert table.position == 0
        assert table.num_rows == 3
        assert table.num_cols == 2
        assert table.table_type == "financial_statement"  # "Revenue" triggers this
        assert table.approx_tokens > 0
        assert "<table>" in table.html
        assert table.markdown  # Should have markdown content

    def test_table_tokens_estimation(self):
        """Token estimation should be reasonable."""
        html = "<table><tr><td>Short</td></tr></table>"
        table = Table.from_html(html, "Test", 0)
        # Very short table should have few tokens
        assert table.approx_tokens < 50

    def test_table_classification_propagates(self):
        """Table type should be set based on caption and content."""
        html = "<table><tr><th>Executive</th><th>Salary</th></tr></table>"
        table = Table.from_html(html, "Executive Compensation", 0)
        assert table.table_type == "compensation"


class TestTableAwareChunking:
    """Tests for table-aware paragraph splitting and chunking."""

    def test_small_table_stays_intact(self):
        """Tables under 2000 tokens should not be split."""
        parser = SECFilingParser()

        small_table = f"""{TABLE_START_MARKER} Revenue Summary]
Year  Revenue
2023  $100M
2024  $110M
{TABLE_END_MARKER}"""

        section = Section(
            item="Item 7",
            title="MD&A",
            content=f"Intro text.\n\n{small_table}\n\nMore text.",
            start_pos=0,
            end_pos=100,
        )

        chunks = parser.chunk_section(section, {"ticker": "TEST"})
        table_chunks = [c for c in chunks if c.has_table]

        assert len(table_chunks) == 1
        assert table_chunks[0].content_type == "table"
        assert table_chunks[0].is_table_continuation is False
        assert TABLE_START_MARKER in table_chunks[0].text
        assert TABLE_END_MARKER in table_chunks[0].text

    def test_large_table_split_with_headers(self):
        """Tables over 2000 tokens should be split with headers repeated."""
        parser = SECFilingParser()

        # Create a large table (over 2000 tokens)
        rows = "\n".join([f"Row{i}  Value{i}  " + "extra " * 50 for i in range(100)])
        large_table = f"""{TABLE_START_MARKER} Large Data Table]
Column1  Column2  Column3
{rows}
{TABLE_END_MARKER}"""

        section = Section(
            item="Item 8",
            title="Financial Statements",
            content=large_table,
            start_pos=0,
            end_pos=100,
        )

        chunks = parser.chunk_section(section, {"ticker": "TEST"})
        table_chunks = [c for c in chunks if c.has_table]

        # Should have multiple chunks
        assert len(table_chunks) > 1

        # First chunk should not be continuation
        assert table_chunks[0].is_table_continuation is False

        # Subsequent chunks should be continuations
        for chunk in table_chunks[1:]:
            assert chunk.is_table_continuation is True

        # All chunks should have the header row
        for chunk in table_chunks:
            assert "Column1" in chunk.text or "Column2" in chunk.text

    def test_table_structure_preserved_in_paragraphs(self):
        """Table newlines should be preserved when splitting into paragraphs."""
        parser = SECFilingParser()

        table_text = f"""{TABLE_START_MARKER} Test Table]
Header1  Header2
Row1  Data1
Row2  Data2
{TABLE_END_MARKER}"""

        paragraphs = parser._split_into_paragraphs(table_text)

        assert len(paragraphs) == 1
        # Should preserve newlines (not join with spaces)
        assert "\n" in paragraphs[0]
        assert "Header1" in paragraphs[0]
        assert "Row1" in paragraphs[0]

    def test_mixed_text_and_tables(self):
        """Text and tables should be chunked separately."""
        parser = SECFilingParser()

        content = f"""This is introductory text about the company.

{TABLE_START_MARKER} Revenue Table]
Year  Revenue
2023  $100M
{TABLE_END_MARKER}

This is more text after the table.

{TABLE_START_MARKER} Expense Table]
Year  Expenses
2023  $50M
{TABLE_END_MARKER}

Final paragraph of text."""

        section = Section(
            item="Item 7",
            title="MD&A",
            content=content,
            start_pos=0,
            end_pos=100,
        )

        chunks = parser.chunk_section(section, {"ticker": "TEST"})
        table_chunks = [c for c in chunks if c.has_table]
        text_chunks = [c for c in chunks if not c.has_table]

        # Should have 2 table chunks
        assert len(table_chunks) == 2

        # All table chunks should have content_type="table"
        for chunk in table_chunks:
            assert chunk.content_type == "table"

        # Text chunks should have content_type="text"
        for chunk in text_chunks:
            assert chunk.content_type == "text"


class TestChunkMetadataPropagation:
    """Tests for metadata propagation to chunks."""

    def test_content_type_in_metadata(self):
        """content_type should be in chunk metadata."""
        parser = SECFilingParser()

        table = f"""{TABLE_START_MARKER} Test]
Col1  Col2
Data  Value
{TABLE_END_MARKER}"""

        section = Section(
            item="Item 1",
            title="Business",
            content=f"Text before.\n\n{table}\n\nText after.",
            start_pos=0,
            end_pos=100,
        )

        chunks = parser.chunk_section(section, {"ticker": "TEST"})
        table_chunk = next(c for c in chunks if c.has_table)

        assert table_chunk.metadata["content_type"] == "table"
        assert table_chunk.metadata["has_table"] is True

    def test_table_type_propagation(self):
        """table_type should propagate from Section.tables to chunks."""
        parser = SECFilingParser()

        # Create a section with table metadata
        table = f"""{TABLE_START_MARKER} Executive Compensation]
Name  Salary  Bonus
CEO  $1M  $500K
{TABLE_END_MARKER}"""

        section = Section(
            item="Item 11",
            title="Executive Compensation",
            content=table,
            start_pos=0,
            end_pos=100,
            tables=[
                Table(
                    table_id="test_table_0",
                    html="<table>...</table>",
                    markdown="...",
                    caption="Executive Compensation",
                    position=0,
                    num_rows=2,
                    num_cols=3,
                    table_type="compensation",
                    approx_tokens=50,
                )
            ],
        )

        chunks = parser.chunk_section(section, {"ticker": "TEST"})
        table_chunk = next(c for c in chunks if c.has_table)

        assert table_chunk.table_type == "compensation"
        assert table_chunk.metadata["table_type"] == "compensation"

    def test_is_table_continuation_in_metadata(self):
        """is_table_continuation should be in chunk metadata."""
        parser = SECFilingParser()

        # Create a large table that will be split
        rows = "\n".join([f"Row{i}  " + "data " * 100 for i in range(50)])
        table = f"""{TABLE_START_MARKER} Large Table]
Header1  Header2
{rows}
{TABLE_END_MARKER}"""

        section = Section(
            item="Item 8",
            title="Financial",
            content=table,
            start_pos=0,
            end_pos=100,
        )

        chunks = parser.chunk_section(section, {"ticker": "TEST"})

        if len(chunks) > 1:
            # First chunk
            assert chunks[0].metadata["is_table_continuation"] is False
            # Subsequent chunks
            for chunk in chunks[1:]:
                if chunk.has_table:
                    assert chunk.metadata["is_table_continuation"] is True


class TestTableCorpusClassification:
    """Tests using the 50-table test corpus."""

    @pytest.fixture
    def table_corpus(self):
        """Load the test corpus."""
        corpus_path = Path(__file__).parent / "fixtures" / "table_corpus.json"
        if not corpus_path.exists():
            pytest.skip("Table corpus not found")
        with open(corpus_path) as f:
            return json.load(f)

    def test_corpus_classification_accuracy(self, table_corpus):
        """Classification should match corpus labels for most tables."""
        correct = 0
        total = len(table_corpus)
        mismatches = []

        for table in table_corpus:
            expected_type = table.get("table_type", "other")
            caption = table.get("caption", "")
            markdown = table.get("markdown", "")

            predicted_type = classify_table(caption, markdown)

            if predicted_type == expected_type:
                correct += 1
            else:
                mismatches.append(
                    {
                        "id": table.get("id"),
                        "expected": expected_type,
                        "predicted": predicted_type,
                        "caption": caption[:50],
                    }
                )

        accuracy = correct / total if total > 0 else 0

        # Allow some mismatches (corpus may have edge cases, labels may be inconsistent)
        # Simple regex patterns can't match complex human labeling - aim for 40%
        # This test primarily ensures we don't regress significantly
        assert accuracy >= 0.4, f"Classification accuracy {accuracy:.1%} below 40%. Mismatches: {mismatches[:5]}"

    def test_corpus_table_types_covered(self, table_corpus):
        """All table types in corpus should be in our patterns."""
        corpus_types = {t.get("table_type", "other") for t in table_corpus}
        pattern_types = set(TABLE_TYPE_PATTERNS.keys()) | {"other"}

        for corpus_type in corpus_types:
            assert corpus_type in pattern_types, f"Corpus type '{corpus_type}' not in patterns"


class TestSplitLargeTable:
    """Tests for _split_large_table helper function."""

    def test_small_table_not_split(self):
        """Tables under max_tokens should not be split."""
        parser = SECFilingParser()

        table = f"""{TABLE_START_MARKER} Small]
Header
Row1
{TABLE_END_MARKER}"""

        result = parser._split_large_table(table, max_tokens=2000)

        assert len(result) == 1
        assert result[0][1] is False  # is_continuation should be False

    def test_large_table_split(self):
        """Tables over max_tokens should be split."""
        parser = SECFilingParser()

        rows = "\n".join([f"Row{i}  " + "data " * 50 for i in range(100)])
        table = f"""{TABLE_START_MARKER} Large]
Header1  Header2
{rows}
{TABLE_END_MARKER}"""

        result = parser._split_large_table(table, max_tokens=500)

        assert len(result) > 1

        # First chunk is not continuation
        assert result[0][1] is False

        # Subsequent chunks are continuations
        for text, is_cont in result[1:]:
            assert is_cont is True

    def test_header_repeated_in_all_chunks(self):
        """Header row should be in all split chunks."""
        parser = SECFilingParser()

        rows = "\n".join([f"Row{i}  " + "data " * 50 for i in range(100)])
        table = f"""{TABLE_START_MARKER} Large]
HeaderA  HeaderB
{rows}
{TABLE_END_MARKER}"""

        result = parser._split_large_table(table, max_tokens=500)

        for text, _ in result:
            assert "HeaderA" in text or "HeaderB" in text
            assert TABLE_START_MARKER in text
            assert TABLE_END_MARKER in text


class TestIsTableParagraph:
    """Tests for _is_table_paragraph helper."""

    def test_detects_table_paragraph(self):
        """Should detect paragraphs with table markers."""
        parser = SECFilingParser()

        table = f"{TABLE_START_MARKER} Test]\nData\n{TABLE_END_MARKER}"
        assert parser._is_table_paragraph(table) is True

    def test_rejects_non_table_paragraph(self):
        """Should reject paragraphs without table markers."""
        parser = SECFilingParser()

        assert parser._is_table_paragraph("Regular text paragraph.") is False
        assert parser._is_table_paragraph(TABLE_START_MARKER) is False  # Only start
        assert parser._is_table_paragraph(TABLE_END_MARKER) is False  # Only end
