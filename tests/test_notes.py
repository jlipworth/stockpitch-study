"""Tests for the handwritten notes converter module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import pytest

from src.notes.converter import (
    DEFAULT_VISION_MODEL,
    ConversionResult,
    NotesConverter,
    PageResult,
    convert_notes,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf(temp_dir):
    """Create a simple test PDF with text."""
    pdf_path = temp_dir / "test_notes.pdf"

    # Create a simple PDF with PyMuPDF
    doc = fitz.open()

    # Add first page with some text
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page 1: Meeting Notes", fontsize=16)
    page1.insert_text((72, 100), "- Discussed project timeline", fontsize=12)
    page1.insert_text((72, 120), "- Budget approved", fontsize=12)

    # Add second page
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page 2: Action Items", fontsize=16)
    page2.insert_text((72, 100), "1. Complete analysis", fontsize=12)
    page2.insert_text((72, 120), "2. Send report", fontsize=12)

    doc.save(pdf_path)
    doc.close()

    return pdf_path


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response for vision."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="# Meeting Notes\n\n- Discussed project timeline\n- Budget approved")]
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 50
    return mock_response


class TestPageResult:
    """Tests for PageResult dataclass."""

    def test_page_result_creation(self):
        result = PageResult(
            page_num=1,
            text="# Notes\n\nSome content",
            input_tokens=500,
            output_tokens=25,
        )

        assert result.page_num == 1
        assert "Notes" in result.text
        assert result.input_tokens == 500
        assert result.output_tokens == 25


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_conversion_result_creation(self):
        pages = [
            PageResult(page_num=1, text="Page 1 content", input_tokens=100, output_tokens=20),
            PageResult(page_num=2, text="Page 2 content", input_tokens=100, output_tokens=20),
        ]
        result = ConversionResult(
            source_path="/path/to/notes.pdf",
            num_pages=2,
            markdown="Page 1 content\n\nPage 2 content",
            pages=pages,
            total_input_tokens=200,
            total_output_tokens=40,
        )

        assert result.num_pages == 2
        assert result.total_input_tokens == 200
        assert len(result.pages) == 2

    def test_save_result(self, temp_dir):
        result = ConversionResult(
            source_path="/path/to/notes.pdf",
            num_pages=1,
            markdown="# Test Notes\n\nThis is a test.",
            pages=[],
            total_input_tokens=100,
            total_output_tokens=20,
        )

        output_path = temp_dir / "output.md"
        result.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Test Notes" in content


class TestNotesConverter:
    """Tests for NotesConverter class."""

    def test_init_requires_api_key(self):
        """Converter should fail without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                NotesConverter()

    @patch("src.notes.converter.Anthropic")
    def test_init_with_api_key(self, mock_anthropic_class):
        """Converter should initialize with API key."""
        converter = NotesConverter(api_key="test-key")
        assert converter.model == DEFAULT_VISION_MODEL
        assert converter.dpi == 200

    @patch("src.notes.converter.Anthropic")
    def test_init_custom_dpi(self, mock_anthropic_class):
        """Converter should accept custom DPI."""
        converter = NotesConverter(api_key="test-key", dpi=300)
        assert converter.dpi == 300

    @patch("src.notes.converter.Anthropic")
    def test_extract_pages(self, mock_anthropic_class, sample_pdf):
        """Should extract pages from PDF as images."""
        converter = NotesConverter(api_key="test-key")
        images = converter.extract_pages(sample_pdf)

        assert len(images) == 2
        # Each image should be PNG bytes
        for img in images:
            assert isinstance(img, bytes)
            # PNG magic bytes
            assert img[:8] == b"\x89PNG\r\n\x1a\n"

    @patch("src.notes.converter.Anthropic")
    def test_transcribe_page(self, mock_anthropic_class, mock_anthropic_response):
        """Should transcribe a page using Claude vision."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        converter = NotesConverter(api_key="test-key")

        # Create dummy image bytes (minimal valid PNG)
        import base64

        # 1x1 transparent PNG
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        image_bytes = base64.b64decode(png_b64)

        result = converter.transcribe_page(image_bytes, page_num=1)

        assert isinstance(result, PageResult)
        assert result.page_num == 1
        assert "Meeting Notes" in result.text
        assert result.input_tokens == 1000
        assert result.output_tokens == 50

    @patch("src.notes.converter.Anthropic")
    def test_convert_full_pdf(self, mock_anthropic_class, sample_pdf, mock_anthropic_response):
        """Should convert entire PDF to markdown."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        converter = NotesConverter(api_key="test-key")
        result = converter.convert(sample_pdf)

        assert isinstance(result, ConversionResult)
        assert result.num_pages == 2
        assert len(result.pages) == 2
        assert result.total_input_tokens == 2000  # 1000 per page
        assert result.total_output_tokens == 100  # 50 per page
        assert len(result.markdown) > 0

    @patch("src.notes.converter.Anthropic")
    def test_convert_with_progress_callback(self, mock_anthropic_class, sample_pdf, mock_anthropic_response):
        """Should call progress callback for each page."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        converter = NotesConverter(api_key="test-key")
        progress_calls = []

        def callback(page_num, total):
            progress_calls.append((page_num, total))

        converter.convert(sample_pdf, progress_callback=callback)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2)
        assert progress_calls[1] == (2, 2)

    @patch("src.notes.converter.Anthropic")
    def test_convert_nonexistent_file(self, mock_anthropic_class):
        """Should raise error for missing file."""
        converter = NotesConverter(api_key="test-key")

        with pytest.raises(FileNotFoundError):
            converter.convert(Path("/nonexistent/file.pdf"))


class TestConvertNotesFunction:
    """Tests for convenience function."""

    @patch("src.notes.converter.Anthropic")
    def test_convert_notes_saves_output(self, mock_anthropic_class, sample_pdf, temp_dir, mock_anthropic_response):
        """Should save markdown to output path."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        output_path = temp_dir / "output.md"
        result = convert_notes(sample_pdf, output_path, api_key="test-key")

        assert output_path.exists()
        assert isinstance(result, ConversionResult)

    @patch("src.notes.converter.Anthropic")
    def test_convert_notes_default_output(self, mock_anthropic_class, sample_pdf, mock_anthropic_response):
        """Should default to same name with .md extension."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        convert_notes(sample_pdf, api_key="test-key")

        expected_output = sample_pdf.with_suffix(".md")
        assert expected_output.exists()

        # Cleanup
        expected_output.unlink()
