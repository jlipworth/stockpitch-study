"""Tests for transcript parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.transcript_parser import (
    SECTION_PATTERNS,
    ParsedTranscript,
    Speaker,
    SpeakerBlock,
    TranscriptParser,
    parse_transcript,
)

# Sample transcript text in S&P Capital IQ format
SAMPLE_TRANSCRIPT = """
S&P Global Market Intelligence

Apple Inc. (AAPL)
Q3 2024 Earnings Call
August 1, 2024

CALL PARTICIPANTS

EXECUTIVES

Tim Cook - CEO
Luca Maestri - CFO, Senior Vice President

ANALYSTS

Amit Daryanani - Evercore ISI
Samik Chatterjee - JPMorgan

PRESENTATION

Tim Cook

Good afternoon, everyone. Thank you for joining us today. I'm pleased to report another strong quarter for Apple. Our revenue came in at $85.8 billion, which was above the high end of our guidance range.

Let me start with iPhone. We continue to see strong demand for iPhone 15 Pro and Pro Max. Customer satisfaction remains incredibly high.

Luca Maestri

Thank you, Tim. Let me provide more detail on our financial performance.

Total revenue for the quarter was $85.8 billion, up 5% year over year. We generated strong operating cash flow of $26 billion during the quarter.

QUESTION AND ANSWER

Amit Daryanani

Thanks for taking my question. Tim, can you talk about the trajectory you're seeing in China? And how do you think about the competitive environment there?

Tim Cook

Thanks, Amit. China is an incredibly important market for us. We saw some sequential improvement during the quarter. The competitive environment is always intense, but we feel good about our position.

Samik Chatterjee

Great, thanks for taking my question. Luca, can you discuss the gross margin outlook for next quarter?

Luca Maestri

Sure, Samik. We expect gross margin to be in the range of 46% to 47% for the September quarter.
"""


class TestSectionPatterns:
    """Test section detection patterns."""

    def test_participants_pattern(self):
        """Test CALL PARTICIPANTS detection."""
        pattern = SECTION_PATTERNS["participants"]
        assert pattern.search("CALL PARTICIPANTS")
        assert pattern.search("CORPORATE PARTICIPANTS")
        assert pattern.search("conference call participants")

    def test_executives_pattern(self):
        """Test EXECUTIVES section detection."""
        pattern = SECTION_PATTERNS["executives"]
        assert pattern.search("EXECUTIVES")
        assert pattern.search("Executive")

    def test_analysts_pattern(self):
        """Test ANALYSTS section detection."""
        pattern = SECTION_PATTERNS["analysts"]
        assert pattern.search("ANALYSTS")
        assert pattern.search("Analyst")

    def test_presentation_pattern(self):
        """Test PRESENTATION section detection."""
        pattern = SECTION_PATTERNS["presentation"]
        assert pattern.search("PRESENTATION")
        assert pattern.search("Prepared Remarks")
        assert pattern.search("MANAGEMENT DISCUSSION")

    def test_qa_pattern(self):
        """Test Q&A section detection."""
        pattern = SECTION_PATTERNS["qa"]
        assert pattern.search("QUESTION AND ANSWER")
        assert pattern.search("Q&A")
        assert pattern.search("Q & A")


class TestTranscriptMetadataParsing:
    """Test metadata extraction from transcripts."""

    def test_parse_ticker(self):
        """Test ticker extraction from transcript."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata(SAMPLE_TRANSCRIPT, "")
        assert metadata.ticker == "AAPL"

    def test_parse_ticker_provided(self):
        """Test ticker when provided explicitly."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata("No ticker here", "MSFT")
        assert metadata.ticker == "MSFT"

    def test_parse_event_type(self):
        """Test event type extraction."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata(SAMPLE_TRANSCRIPT, "")
        assert "Q3 2024" in metadata.event_type
        assert "Earnings" in metadata.event_type

    def test_parse_event_date(self):
        """Test event date extraction."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata(SAMPLE_TRANSCRIPT, "")
        assert "August" in metadata.event_date or "2024" in metadata.event_date


class TestParticipantParsing:
    """Test participant parsing."""

    def test_parse_executives(self):
        """Test executive parsing from participants section."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata(SAMPLE_TRANSCRIPT, "AAPL")

        assert len(metadata.executives) >= 1
        names = [e.name for e in metadata.executives]
        assert any("Tim Cook" in n for n in names) or any("Cook" in n for n in names)

    def test_parse_analysts(self):
        """Test analyst parsing from participants section."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata(SAMPLE_TRANSCRIPT, "AAPL")

        assert len(metadata.analysts) >= 1
        names = [a.name for a in metadata.analysts]
        assert any("Amit" in n or "Daryanani" in n for n in names)


class TestSectionSplitting:
    """Test section splitting."""

    def test_split_sections(self):
        """Test splitting transcript into sections."""
        parser = TranscriptParser()
        sections = parser._split_sections(SAMPLE_TRANSCRIPT)

        assert "presentation" in sections
        assert "qa" in sections

    def test_presentation_content(self):
        """Test presentation section contains expected content."""
        parser = TranscriptParser()
        sections = parser._split_sections(SAMPLE_TRANSCRIPT)

        assert "strong quarter" in sections["presentation"]
        assert "revenue" in sections["presentation"].lower()

    def test_qa_content(self):
        """Test Q&A section contains expected content."""
        parser = TranscriptParser()
        sections = parser._split_sections(SAMPLE_TRANSCRIPT)

        assert "China" in sections["qa"]
        assert "gross margin" in sections["qa"].lower()


class TestSpeakerBlockParsing:
    """Test speaker block parsing."""

    def test_parse_presentation_blocks(self):
        """Test parsing speaker blocks from presentation."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata(SAMPLE_TRANSCRIPT, "AAPL")
        sections = parser._split_sections(SAMPLE_TRANSCRIPT)

        blocks = parser._parse_speaker_blocks(sections["presentation"], "presentation", metadata)

        # Should have blocks for Tim Cook and Luca Maestri
        assert len(blocks) >= 2
        speakers = [b.speaker.name for b in blocks]
        assert any("Cook" in s or "Tim" in s for s in speakers)

    def test_parse_qa_blocks(self):
        """Test parsing speaker blocks from Q&A."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata(SAMPLE_TRANSCRIPT, "AAPL")
        sections = parser._split_sections(SAMPLE_TRANSCRIPT)

        blocks = parser._parse_speaker_blocks(sections["qa"], "qa", metadata)

        # Should have analyst questions and executive answers
        assert len(blocks) >= 2


class TestFullParsing:
    """Test full transcript parsing with mocked PDF."""

    @patch("src.rag.transcript_parser.fitz")
    def test_parse_file(self, mock_fitz):
        """Test full file parsing with mocked PDF."""
        # Mock PDF extraction
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = SAMPLE_TRANSCRIPT
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        # Create temp path
        pdf_path = Path("/tmp/fake_transcript.pdf")

        with patch.object(Path, "exists", return_value=True):
            parser = TranscriptParser()
            result = parser.parse_file(pdf_path, "AAPL")

        assert isinstance(result, ParsedTranscript)
        assert result.metadata.ticker == "AAPL"
        assert len(result.presentation_blocks) >= 1
        assert len(result.qa_blocks) >= 1

    @patch("src.rag.transcript_parser.fitz")
    def test_convenience_function(self, mock_fitz):
        """Test parse_transcript convenience function."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = SAMPLE_TRANSCRIPT
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        pdf_path = Path("/tmp/fake_transcript.pdf")

        with patch.object(Path, "exists", return_value=True):
            result = parse_transcript(pdf_path, "AAPL")

        assert isinstance(result, ParsedTranscript)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self):
        """Test handling of empty transcript."""
        parser = TranscriptParser()
        metadata = parser._parse_metadata("", "")
        assert metadata.ticker == ""
        assert metadata.executives == []

    def test_missing_qa_section(self):
        """Test transcript with no Q&A section."""
        text = """
        PRESENTATION

        Speaker One
        Hello everyone.
        """
        parser = TranscriptParser()
        sections = parser._split_sections(text)

        assert "presentation" in sections
        assert sections.get("qa", "") == ""

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        parser = TranscriptParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/path.pdf"))


class TestSpeakerClass:
    """Test Speaker dataclass."""

    def test_speaker_creation(self):
        """Test creating a Speaker."""
        speaker = Speaker(
            name="Tim Cook",
            title="CEO",
            company="Apple Inc.",
            role="executive",
        )
        assert speaker.name == "Tim Cook"
        assert speaker.role == "executive"

    def test_speaker_defaults(self):
        """Test Speaker default values."""
        speaker = Speaker(name="John Doe")
        assert speaker.title == ""
        assert speaker.company == ""
        assert speaker.role == ""


class TestSpeakerBlock:
    """Test SpeakerBlock dataclass."""

    def test_speaker_block_creation(self):
        """Test creating a SpeakerBlock."""
        speaker = Speaker(name="Tim Cook", role="executive")
        block = SpeakerBlock(
            speaker=speaker,
            text="Hello everyone.",
            section="presentation",
        )
        assert block.speaker.name == "Tim Cook"
        assert block.section == "presentation"
