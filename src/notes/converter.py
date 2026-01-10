"""Handwritten notes converter using Claude vision."""

import base64
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# Default model for vision tasks
DEFAULT_VISION_MODEL = "claude-sonnet-4-5-20250929"

# System prompt for handwriting transcription
TRANSCRIPTION_PROMPT = """You are an expert at transcribing handwritten notes \
into clean, well-formatted markdown.

Instructions:
1. Transcribe ALL text from the handwritten notes exactly as written
2. Preserve the original structure (headings, lists, paragraphs)
3. Use markdown formatting appropriately:
   - # for main headings
   - ## for subheadings
   - - or * for bullet points
   - 1. 2. 3. for numbered lists
   - **bold** for emphasized text
   - > for quotes or callouts
4. If text is unclear, use [unclear] or your best interpretation with [?]
5. Preserve any diagrams or drawings as descriptions in [brackets]
6. Maintain the logical flow and organization of the notes

IMPORTANT - Handwritten Tables:
- Look for grid-like structures, aligned columns, or boxed areas with tabular data
- Convert to markdown tables:
  | Header 1 | Header 2 | Header 3 |
  |----------|----------|----------|
  | data     | data     | data     |
- If the table structure is unclear, describe it: [Table: approximate description of content]
- For comparison lists (side-by-side items), use tables even without drawn lines

IMPORTANT - User's Note-Taking Style:
- The user marks questions/follow-ups with a HIGHLIGHTER STREAK (a blank colored streak
  with nothing written underneath it). The actual question is written to the LEFT or RIGHT
  of the highlighter mark (varies), usually spanning 1-2 lines.
- When you see a highlighter streak with text adjacent to it (either side), format as a question:
  > **Q:** [the question text]
- At the end of each page, collect all questions in a "Questions" section for easy reference

Output ONLY the transcribed markdown content, no explanations or preamble."""


@dataclass
class PageResult:
    """Result of transcribing a single page."""

    page_num: int
    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class ConversionResult:
    """Result of converting an entire PDF."""

    source_path: str
    num_pages: int
    markdown: str
    pages: list[PageResult]
    total_input_tokens: int
    total_output_tokens: int

    def save(self, output_path: Path) -> None:
        """Save markdown to file."""
        output_path.write_text(self.markdown, encoding="utf-8")


class NotesConverter:
    """Converts handwritten PDF notes to markdown using Claude vision."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_VISION_MODEL,
        dpi: int = 200,
    ):
        """
        Initialize the converter.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use for vision
            dpi: Resolution for rendering PDF pages (higher = better quality but slower)
        """
        self.model = model
        self.dpi = dpi

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or pass api_key parameter.")
        self.client = Anthropic(api_key=api_key)

    def extract_pages(self, pdf_path: Path) -> list[bytes]:
        """
        Extract pages from PDF as PNG images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PNG image bytes for each page
        """
        doc = fitz.open(pdf_path)
        images = []

        # Calculate zoom factor for target DPI (default PDF is 72 DPI)
        zoom = self.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            # Render page to pixmap (image)
            pix = page.get_pixmap(matrix=matrix)
            # Convert to PNG bytes
            png_bytes = pix.tobytes("png")
            images.append(png_bytes)

        doc.close()
        return images

    def transcribe_page(
        self,
        image_bytes: bytes,
        page_num: int,
        context: str = "",
    ) -> PageResult:
        """
        Transcribe a single page using Claude vision.

        Args:
            image_bytes: PNG image bytes
            page_num: Page number (1-indexed)
            context: Optional context from previous pages

        Returns:
            PageResult with transcribed text
        """
        # Encode image as base64
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Build message with image
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            },
            {
                "type": "text",
                "text": f"Please transcribe the handwritten notes on this page (page {page_num})."
                + (f"\n\nContext from previous pages:\n{context[:500]}..." if context else ""),
            },
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=TRANSCRIPTION_PROMPT,
            messages=[{"role": "user", "content": user_content}],  # type: ignore[typeddict-item]
        )

        # Extract text from response (first content block is always text for our prompts)
        text_content: str = getattr(response.content[0], "text", "")
        return PageResult(
            page_num=page_num,
            text=text_content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def convert(
        self,
        pdf_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ConversionResult:
        """
        Convert a PDF of handwritten notes to markdown.

        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback(page_num, total_pages) for progress

        Returns:
            ConversionResult with full markdown and page details
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Converting PDF: {pdf_path}")

        # Extract pages as images
        page_images = self.extract_pages(pdf_path)
        num_pages = len(page_images)
        logger.debug(f"Extracted {num_pages} pages from PDF")

        if num_pages == 0:
            raise ValueError(f"No pages found in PDF: {pdf_path}")

        # Transcribe each page
        pages: list[PageResult] = []
        context = ""

        for i, image_bytes in enumerate(page_images):
            page_num = i + 1

            if progress_callback:
                progress_callback(page_num, num_pages)

            result = self.transcribe_page(
                image_bytes=image_bytes,
                page_num=page_num,
                context=context,
            )
            pages.append(result)

            # Build context for next page (last ~500 chars)
            context = result.text[-500:] if result.text else ""

        # Combine all pages into single markdown
        markdown_parts = []
        for page in pages:
            if num_pages > 1:
                markdown_parts.append(f"<!-- Page {page.page_num} -->\n")
            markdown_parts.append(page.text)
            markdown_parts.append("\n\n")

        markdown = "".join(markdown_parts).strip()

        total_in = sum(p.input_tokens for p in pages)
        total_out = sum(p.output_tokens for p in pages)
        logger.info(f"Conversion complete: {num_pages} pages, {total_in} input / {total_out} output tokens")

        return ConversionResult(
            source_path=str(pdf_path),
            num_pages=num_pages,
            markdown=markdown,
            pages=pages,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )


def convert_notes(
    pdf_path: Path,
    output_path: Path | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_VISION_MODEL,
    dpi: int = 200,
) -> ConversionResult:
    """
    Convenience function to convert handwritten notes PDF to markdown.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save markdown (defaults to same name with .md)
        api_key: Anthropic API key
        model: Claude model to use
        dpi: Resolution for PDF rendering

    Returns:
        ConversionResult with markdown content
    """
    converter = NotesConverter(api_key=api_key, model=model, dpi=dpi)
    result = converter.convert(pdf_path)

    if output_path is None:
        output_path = pdf_path.with_suffix(".md")

    result.save(output_path)
    return result
