"""Hierarchical document summarization using Claude."""

import asyncio
import logging
import os
import re
import shutil
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic, AsyncAnthropic

from src.rag import SECFilingParser, Section
from src.summarizer.config import (
    DEFAULT_SECTION_WEIGHT,
    SECTION_WEIGHTS,
    SectionWeight,
)

logger = logging.getLogger(__name__)

# Chunking thresholds - chars per chunk before splitting
# Context window is ~200k tokens (~800k chars), so these are conservative
# Larger = fewer chunks = less overhead, but very large sections may lose coherence
MAX_CHUNK_CHARS_HIGH = 250_000  # ~62k tokens - for high-weight sections (MD&A, Risk Factors)
MAX_CHUNK_CHARS_DEFAULT = 150_000  # ~37k tokens - for medium/low weight sections


def sanitize_custom_id_component(text: str) -> str:
    """Sanitize text for use in Anthropic Batch API custom_id.

    Per Anthropic API: custom_id must be alphanumeric with limited special chars (-, _, .)

    Args:
        text: Raw text that may contain special characters

    Returns:
        Sanitized text safe for custom_id use

    Examples:
        >>> sanitize_custom_id_component("Q&A Session")
        'Q-and-A-Session'
        >>> sanitize_custom_id_component("Part I, Item 1A")
        'Part-I-Item-1A'
        >>> sanitize_custom_id_component("Operator/Other")
        'Operator-Other'
    """
    # Replace common special characters with readable alternatives
    text = text.replace("&", "-and-")
    text = text.replace("/", "-")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(":", "")
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace(" ", "-")

    # Collapse multiple hyphens and strip leading/trailing
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")

    return text


def wrap_markdown_lines(text: str, width: int = 100) -> str:
    """Wrap long lines in markdown while preserving structure.

    Handles:
    - Headers (# lines) - not wrapped
    - Bullet points (- or *) - wrapped with proper indentation
    - Numbered lists (1.) - wrapped with proper indentation
    - Tables (| lines) - not wrapped
    - Code blocks (``` or indented) - not wrapped
    - Regular paragraphs - wrapped at word boundaries

    Args:
        text: Markdown text to wrap
        width: Maximum line width (default 100)

    Returns:
        Text with long lines wrapped
    """
    lines = text.split("\n")
    result = []
    in_code_block = False
    in_table = False

    for line in lines:
        # Track code block state
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        # Don't wrap inside code blocks
        if in_code_block:
            result.append(line)
            continue

        # Don't wrap indented code (4+ spaces)
        if line.startswith("    ") and not line.strip().startswith("-"):
            result.append(line)
            continue

        # Track table state (lines starting with |)
        if line.strip().startswith("|"):
            in_table = True
            result.append(line)
            continue
        elif in_table and not line.strip():
            in_table = False
            result.append(line)
            continue
        elif in_table:
            result.append(line)
            continue

        # Don't wrap headers
        if line.strip().startswith("#"):
            result.append(line)
            continue

        # Don't wrap horizontal rules
        if line.strip() in ("---", "***", "___"):
            result.append(line)
            continue

        # Don't wrap short lines
        if len(line) <= width:
            result.append(line)
            continue

        # Handle bullet points with proper indentation
        bullet_match = re.match(r"^(\s*)([*-]|\d+\.)\s+", line)
        if bullet_match:
            prefix = bullet_match.group(0)
            content = line[len(prefix) :]
            # Wrap the content with subsequent indent
            subsequent_indent = " " * len(prefix)
            wrapped = textwrap.fill(
                content,
                width=width,
                initial_indent=prefix,
                subsequent_indent=subsequent_indent,
            )
            result.append(wrapped)
            continue

        # Handle bold/strong starts (like **Bull Case:** )
        # Only match short bold prefixes to avoid issues with long bold text
        bold_match = re.match(r"^(\*\*[^*]{1,30}\*\*:?\s*)", line)
        if bold_match:
            prefix = bold_match.group(0)
            content = line[len(prefix) :]
            if content and len(prefix) < width - 20:  # Ensure reasonable wrap width
                wrapped = textwrap.fill(
                    content,
                    width=width - len(prefix),
                    initial_indent="",
                    subsequent_indent="",
                )
                # Re-add prefix to first line
                wrapped_lines = wrapped.split("\n")
                wrapped_lines[0] = prefix + wrapped_lines[0]
                result.extend(wrapped_lines)
            else:
                # Just wrap the whole line normally
                wrapped = textwrap.fill(line, width=width)
                result.append(wrapped)
            continue

        # Regular paragraph - wrap normally
        wrapped = textwrap.fill(line, width=width)
        result.append(wrapped)

    return "\n".join(result)


def display_width(text: str) -> int:
    """Calculate display width of text, ignoring markdown emphasis markers.

    Args:
        text: Text potentially containing **bold** or *italic* markers

    Returns:
        Width as it would display (without the marker characters)
    """
    # Remove emphasis markers for width calculation
    stripped = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    stripped = re.sub(r"\*([^*]+)\*", r"\1", stripped)  # *italic*
    stripped = re.sub(r"__([^_]+)__", r"\1", stripped)  # __bold__
    stripped = re.sub(r"_([^_]+)_", r"\1", stripped)  # _italic_
    return len(stripped)


def align_tables_by_display_width(text: str) -> str:
    """Align markdown table columns based on display width, preserving emphasis.

    mdformat aligns tables by raw character count, but **bold** markers add
    characters that don't display, breaking visual alignment. This realigns
    tables based on how text actually displays while keeping bold/italic intact.

    Args:
        text: Markdown text potentially containing tables

    Returns:
        Text with tables aligned by display width
    """
    lines = text.split("\n")
    result = []
    table_lines: list[str] = []

    def process_table():
        """Align and output accumulated table lines."""
        if not table_lines:
            return

        # Parse table into rows of cells (preserving original content)
        rows = []
        for line in table_lines:
            # Split by | and strip whitespace
            cells = [c.strip() for c in line.split("|")]
            # Remove empty strings from leading/trailing pipes
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            rows.append(cells)

        if not rows:
            result.extend(table_lines)
            return

        # Calculate max DISPLAY width for each column
        num_cols = max(len(row) for row in rows)
        col_widths = [0] * num_cols
        for row in rows:
            for i, cell in enumerate(row):
                if i < num_cols:
                    col_widths[i] = max(col_widths[i], display_width(cell))

        # Rebuild table with proper alignment
        for row in rows:
            # Check if separator row (all dashes/colons)
            is_separator = all(set(cell.strip()).issubset({"-", ":"}) for cell in row if cell.strip())

            aligned_cells = []
            for i in range(num_cols):
                cell = row[i] if i < len(row) else ""
                if is_separator:
                    # Rebuild separator to match column width
                    # Preserve alignment colons if present
                    left_colon = cell.startswith(":")
                    right_colon = cell.endswith(":")
                    dash_count = col_widths[i] - (1 if left_colon else 0) - (1 if right_colon else 0)
                    sep = (":" if left_colon else "") + "-" * dash_count + (":" if right_colon else "")
                    aligned_cells.append(sep)
                else:
                    # Pad based on display width, not raw length
                    dw = display_width(cell)
                    padding = col_widths[i] - dw
                    aligned_cells.append(cell + " " * padding)

            result.append("| " + " | ".join(aligned_cells) + " |")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(line)
        else:
            if table_lines:
                process_table()
                table_lines = []
            result.append(line)

    # Process any remaining table at end of file
    if table_lines:
        process_table()

    return "\n".join(result)


# Default model for summarization
DEFAULT_SUMMARY_MODEL = "claude-sonnet-4-5-20250929"

# ==========================================================================
# TOKEN SCALING CONFIGURATION
# ==========================================================================
# Section-type aware scaling for batch API requests
# Different content types have different compression characteristics:
# - Q&A: Dense multi-speaker content, needs higher ratio
# - Prepared Remarks: More structured, can compress more
# - SEC filings: Highly structured, most compressible

# Base prompts by weight level - section-specific guidance is appended
PROMPT_HIGH_WEIGHT_BASE = """You are a senior financial analyst conducting deep-dive analysis for investment research.

CRITICAL RULES:
- Use ONLY information from the provided document. Do not hallucinate or invent data.
- Use the exact company name/ticker from the Filing context. Do not substitute competitor names.
- When discussing fiscal periods, note the company's fiscal year end (e.g., "FY2026 ends March 31, 2026")
- All figures, percentages, and dates must come directly from the document.

For this HIGH-PRIORITY section, provide a thorough analysis that:
1. Captures ALL material information - don't leave out important details
2. Identifies key trends, changes from prior periods, and inflection points
3. Preserves critical numbers, percentages, dates, and named entities
4. Provides your analytical interpretation, not just summarization

IMPORTANT - Table Handling:
- Convert data tables to clean markdown tables with context
- Preserve key figures and relationships
- For financial tables: include all time periods and key metrics
- For compensation tables: name, title, salary, bonus, stock awards, total

Format your response as markdown with:
- A substantive overview paragraph (3-5 sentences)
- Detailed bullet points organized by theme
- Important tables in clean markdown format
- Analytical observations and flags for further investigation

Be thorough - this is a critical section for investment analysis."""

PROMPT_MEDIUM_WEIGHT_BASE = """You are a financial analyst summarizing documents for investment research.

CRITICAL RULES:
- Use ONLY information from the provided document. Do not hallucinate or invent data.
- Use the exact company name/ticker from the Filing context. Do not substitute competitor names.
- All figures must come directly from the document.

For this section, provide a balanced summary that:
1. Captures the key points and material information
2. Highlights any risks, opportunities, or notable changes
3. Preserves important numbers, percentages, and dates
4. Flags anything unusual or noteworthy for investors

IMPORTANT - Table Handling:
- Convert data tables to clean markdown tables
- Add brief context above each table
- Preserve key figures and relationships

Format your response as markdown with:
- A brief overview (2-3 sentences)
- Organized bullet points for key details
- Important tables in clean markdown format"""

PROMPT_LOW_WEIGHT_BASE = """You are a financial analyst reviewing documents for investment research.

CRITICAL RULES:
- Use ONLY information from the provided document. Do not hallucinate or invent data.
- Use the exact company name/ticker from the Filing context.

For this STANDARD/BOILERPLATE section, provide a brief summary that:
1. Notes any material information (most will be routine)
2. Flags ONLY items that deviate from typical boilerplate
3. Keeps coverage proportional to actual importance

Format your response as markdown with:
- 1-2 sentence overview
- Bullet points ONLY for non-routine items
- Skip if entirely standard boilerplate (just note "Standard disclosure, nothing notable")"""


def build_section_prompt(weight: str, guidance: str | None = None) -> str:
    """Build the full system prompt for a section, including section-specific guidance."""
    if weight == "high":
        base = PROMPT_HIGH_WEIGHT_BASE
    elif weight == "low":
        base = PROMPT_LOW_WEIGHT_BASE
    else:
        base = PROMPT_MEDIUM_WEIGHT_BASE

    if guidance:
        return f"{base}\n\n--- SECTION-SPECIFIC GUIDANCE ---\n{guidance}"
    return base


# Executive synthesis prompt for transcripts/presentations (no investment thesis)
TRANSCRIPT_SUMMARY_PROMPT = """You are a senior financial analyst synthesizing an earnings call or presentation analysis.

CRITICAL RULES:
- Use the exact company name/ticker from the context provided. Do not substitute competitor names.
- All data points must come from the section summaries provided - do not hallucinate figures.
- When referencing fiscal periods, use the company's fiscal year notation consistently.
- ALWAYS use markdown tables for financial data - tables are easier to scan than prose.

Based on the section summaries provided, create a cohesive executive synthesis that:
1. Identifies the 3-5 most critical takeaways from this document
2. Highlights key metrics and guidance shared - USE TABLES
3. Tracks analyst participation and focus areas (for earnings calls)
4. Notes risks and concerns raised or implied
5. Flags items requiring further investigation

Format as markdown:

## Key Takeaways
[3-5 most important points from this document]

## Financial Highlights
**REQUIRED: Use tables for all financial data.** Include:
- Revenue breakdown by segment/product with YoY growth
- Key profitability metrics (margins, operating income)
- Guidance summary table (current Q, full year)
- Key operating metrics (ARR, bookings, etc.)

## Analyst Focus Areas
If this is an earnings call with Q&A, list key analysts and their question topics:
- Name (Firm) - Primary question topic(s)
Highlight themes that multiple analysts asked about (signals institutional concern).
Skip this section if no Q&A or analyst questions documented.

## Risks & Concerns
[Issues raised or implied during the call/presentation]

## Follow-Up Items
[What needs more digging or monitoring]

Be direct and concise - focus on what's new or changed from this document."""


# Executive synthesis prompt - connects the dots
EXECUTIVE_SUMMARY_PROMPT = """You are a senior financial analyst synthesizing an SEC filing analysis for investment decision-making.

CRITICAL RULES:
- Use the exact company name/ticker from the Filing context provided. Do not substitute competitor names.
- All data points must come from the section summaries provided - do not hallucinate figures.
- When referencing fiscal periods, use the company's fiscal year notation consistently.
- ALWAYS use markdown tables for financial data - tables are easier to scan than prose.

Based on the section summaries provided, create a cohesive executive synthesis that:
1. Opens with a clear investment thesis summary (bull and bear case in 2-3 sentences each)
2. Highlights key financial performance - USE TABLES for revenue/profit breakdowns
3. Identifies the 3-5 most critical takeaways that would impact an investment decision
4. Connects dots across sections - what story do they tell together?
5. Highlights key risks ranked by severity and probability
6. Notes growth drivers and opportunities
7. Flags items requiring further investigation

Format as markdown:

## Investment Thesis Summary
**Bull Case:** [2-3 sentences]
**Bear Case:** [2-3 sentences]

## Financial Performance & Profitability
**REQUIRED: Use tables for all financial data.** Include:
- Revenue breakdown by segment with YoY growth %
- Revenue by geography if available
- Profitability metrics (gross margin, operating margin, net margin)
- Key operating metrics the company emphasizes

## Critical Takeaways
[3-5 most important points, ranked]

## Key Risks
[Categorized and prioritized - not just listed]

## Growth Drivers & Opportunities
[What could go right]

## Red Flags & Investigation Items
[What needs more digging]

Be direct and analytical - this synthesis drives investment decisions."""


@dataclass
class SectionSummary:
    """Summary of a single section."""

    section_id: str  # e.g., "Item 1A"
    section_title: str
    summary: str
    weight: str  # high, medium, low
    input_tokens: int
    output_tokens: int
    truncated: bool = False  # True if response hit max_tokens limit


@dataclass
class FilingSummary:
    """Complete summary of a filing."""

    ticker: str
    filing_type: str
    filing_date: str
    source_path: str
    executive_summary: str
    section_summaries: list[SectionSummary]
    total_input_tokens: int
    total_output_tokens: int
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self, wrap_lines: bool = True, line_width: int = 100) -> str:
        """Convert to formatted markdown.

        Args:
            wrap_lines: Wrap long lines for readability (default True)
            line_width: Maximum line width when wrapping (default 100)
        """
        # Wrap executive summary
        exec_summary = self.executive_summary
        if wrap_lines:
            exec_summary = wrap_markdown_lines(exec_summary, line_width)

        # Check if any sections were truncated
        truncated_sections = [s for s in self.section_summaries if s.truncated]

        lines = [
            f"# {self.ticker} {self.filing_type} Analysis",
            f"**Filing Date:** {self.filing_date}",
            f"**Generated:** {self.generated_at[:10]}",
            f"**Tokens Used:** {self.total_input_tokens:,} in / {self.total_output_tokens:,} out",
        ]

        # Add truncation warning if any sections were cut off
        if truncated_sections:
            section_names = ", ".join(s.section_id for s in truncated_sections)
            lines.append("")
            lines.append(
                f"⚠️ **WARNING:** {len(truncated_sections)} section(s) were truncated "
                f"due to token limits: {section_names}"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                exec_summary,
                "",
                "---",
                "",
                "# Section Analysis",
                "",
            ]
        )

        # Group sections by weight for better organization
        high_weight = [s for s in self.section_summaries if s.weight == "high"]
        medium_weight = [s for s in self.section_summaries if s.weight == "medium"]
        low_weight = [s for s in self.section_summaries if s.weight == "low"]

        if high_weight:
            lines.append("## Key Sections (Deep Analysis)")
            lines.append("")
            for section in high_weight:
                summary = section.summary
                if wrap_lines:
                    summary = wrap_markdown_lines(summary, line_width)
                # Add truncation warning if section was cut off
                header = f"### {section.section_id}: {section.section_title}"
                if section.truncated:
                    header += " ⚠️ TRUNCATED"
                lines.extend(
                    [
                        header,
                        "",
                        summary,
                        "",
                    ]
                )

        if medium_weight:
            lines.append("## Supporting Sections")
            lines.append("")
            for section in medium_weight:
                summary = section.summary
                if wrap_lines:
                    summary = wrap_markdown_lines(summary, line_width)
                header = f"### {section.section_id}: {section.section_title}"
                if section.truncated:
                    header += " ⚠️ TRUNCATED"
                lines.extend(
                    [
                        header,
                        "",
                        summary,
                        "",
                    ]
                )

        if low_weight:
            lines.append("## Other Sections (Brief Coverage)")
            lines.append("")
            for section in low_weight:
                summary = section.summary
                if wrap_lines:
                    summary = wrap_markdown_lines(summary, line_width)
                header = f"### {section.section_id}: {section.section_title}"
                if section.truncated:
                    header += " ⚠️ TRUNCATED"
                lines.extend(
                    [
                        header,
                        "",
                        summary,
                        "",
                    ]
                )

        return "\n".join(lines)

    def save(self, output_path: Path) -> None:
        """Save summary to markdown file using atomic write.

        Uses tempfile + shutil.move to prevent corruption if interrupted.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_markdown()
        # Align tables by display width (accounts for **bold** markers)
        content = align_tables_by_display_width(content)

        # Atomic write: write to temp file, then rename (atomic on POSIX)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".md", dir=output_path.parent, prefix=f".{output_path.stem}_")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(content)
            shutil.move(temp_path, output_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    @classmethod
    def from_markdown(cls, path: Path) -> "FilingSummary":
        """
        Load a FilingSummary from a markdown file.

        This parses the markdown format back into a FilingSummary object,
        enabling the resubmit workflow where we merge new sections into
        existing summaries.

        Args:
            path: Path to the markdown file

        Returns:
            FilingSummary object
        """
        import re

        content = path.read_text(encoding="utf-8")
        # Strip BOM if present
        content = content.lstrip("\ufeff")
        lines = content.split("\n")

        # Skip empty lines to find header
        start_idx = 0
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1

        if start_idx >= len(lines):
            raise ValueError("No content found in markdown file")

        # Parse header: # {ticker} {filing_type} Analysis
        # Support tickers with hyphens (e.g., BRK-A, BRK-B)
        header_match = re.match(r"^# ([\w\-\.]+) (.+?) Analysis$", lines[start_idx])
        if not header_match:
            raise ValueError(f"Could not parse header from: {lines[start_idx]}")
        ticker = header_match.group(1)
        filing_type = header_match.group(2)

        # Parse metadata lines
        filing_date = ""
        generated_at = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for line in lines[start_idx + 1 : start_idx + 11]:  # Metadata is in first few lines after header
            if line.startswith("**Filing Date:**"):
                filing_date = line.replace("**Filing Date:**", "").strip()
            elif line.startswith("**Generated:**"):
                generated_at = line.replace("**Generated:**", "").strip()
            elif line.startswith("**Tokens Used:**"):
                tokens_match = re.search(r"([\d,]+) in / ([\d,]+) out", line)
                if tokens_match:
                    total_input_tokens = int(tokens_match.group(1).replace(",", ""))
                    total_output_tokens = int(tokens_match.group(2).replace(",", ""))

        # Find section analysis header to bound exec summary search
        # This prevents accidentally including section content if there's a missing HR
        section_analysis_idx = next(
            (i for i, line in enumerate(lines) if line.strip() == "# Section Analysis"), len(lines)
        )

        # Find executive summary (between first --- and second --- before Section Analysis)
        hr_indices = [i for i, line in enumerate(lines[:section_analysis_idx]) if line.strip() == "---"]
        if len(hr_indices) >= 2:
            exec_start = hr_indices[0] + 1
            exec_end = hr_indices[1]
            executive_summary = "\n".join(lines[exec_start:exec_end]).strip()
        else:
            executive_summary = ""

        # Parse sections - look for ### headers AFTER "# Section Analysis"
        section_summaries: list[SectionSummary] = []
        current_weight = "medium"

        # Find where sections start (after "# Section Analysis")
        section_start_idx = 0
        for idx, line in enumerate(lines):
            if line.strip() == "# Section Analysis":
                section_start_idx = idx + 1
                break

        i = section_start_idx
        while i < len(lines):
            line = lines[i]

            # Track which weight category we're in
            if line.startswith("## Key Sections"):
                current_weight = "high"
            elif line.startswith("## Supporting Sections"):
                current_weight = "medium"
            elif line.startswith("## Other Sections"):
                current_weight = "low"

            # Parse section headers: ### {section_id}: {section_title} [⚠️ TRUNCATED]
            # Valid section headers match patterns like:
            #   ### Item 1: Business Description
            #   ### Prepared Remarks: Management's prepared comments
            #   ### Q and A: Q and A
            # NOT headers inside summary content like:
            #   ### FY2025 Performance vs. Guidance
            section_match = re.match(
                r"^### (Item \d+[A-Z]?|Part [I]+, Item \d+[A-Z]?|"
                r"Prepared Remarks|Q and A|Q&A|Operator/Other|"
                r"Proxy Summary|Voting Matters|Director Nominees|Executive Compensation|"
                r"Stock Ownership|Related Party Transactions|Audit Committee|Governance|"
                r"Presentation [\w]+|Conference [\w]+|Misc Document|Analyst [\w]+): (.+?)( ⚠️ TRUNCATED)?$",
                line,
            )
            if section_match:
                section_id = section_match.group(1)
                section_title = section_match.group(2).strip()
                truncated = section_match.group(3) is not None

                # Collect summary text until next valid section header or weight header
                summary_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    # Stop at weight headers
                    if (
                        next_line.startswith("## Key Sections")
                        or next_line.startswith("## Supporting Sections")
                        or next_line.startswith("## Other Sections")
                    ):
                        break
                    # Stop at next valid section header (use same regex)
                    if re.match(
                        r"^### (Item \d+[A-Z]?|Part [I]+, Item \d+[A-Z]?|"
                        r"Prepared Remarks|Q and A|Q&A|Operator/Other|"
                        r"Proxy Summary|Voting Matters|Director Nominees|Executive Compensation|"
                        r"Stock Ownership|Related Party Transactions|Audit Committee|Governance|"
                        r"Presentation [\w]+|Conference [\w]+|Misc Document|Analyst [\w]+): ",
                        next_line,
                    ):
                        break
                    summary_lines.append(next_line)
                    i += 1

                summary = "\n".join(summary_lines).strip()

                section_summaries.append(
                    SectionSummary(
                        section_id=section_id,
                        section_title=section_title,
                        summary=summary,
                        weight=current_weight,
                        input_tokens=0,  # Not stored in markdown
                        output_tokens=0,  # Not stored in markdown
                        truncated=truncated,
                    )
                )
                continue  # Don't increment i again, already done in inner loop

            i += 1

        return cls(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            source_path=str(path),
            executive_summary=executive_summary,
            section_summaries=section_summaries,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            generated_at=generated_at,
        )


class Summarizer:
    """Hierarchical document summarizer using Claude."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_SUMMARY_MODEL,
        processed_dir: Path | None = None,
        use_batch: bool = False,
        max_concurrent: int = 8,
    ):
        """
        Initialize summarizer.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use
            processed_dir: Directory for saving summaries
            use_batch: Use batch API for 50% cost savings (async, slower)
            max_concurrent: Maximum concurrent API calls for parallel processing
        """
        self.model = model
        self.processed_dir = processed_dir
        self.use_batch = use_batch
        self.max_concurrent = max_concurrent

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or pass api_key parameter.")
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        self.parser = SECFilingParser()

    def _get_section_config(self, section_id: str) -> SectionWeight:
        """Get weight configuration for a section."""
        return SECTION_WEIGHTS.get(section_id, DEFAULT_SECTION_WEIGHT)

    def _format_fiscal_year_end(self, fiscal_year_end: str) -> str:
        """Convert fiscal year end from 'M/D' format to 'Month D' format.

        Args:
            fiscal_year_end: Date string like "3/31" or "12/31"

        Returns:
            Human-readable format like "March 31" or "December 31"
        """
        month_names = {
            "1": "January",
            "2": "February",
            "3": "March",
            "4": "April",
            "5": "May",
            "6": "June",
            "7": "July",
            "8": "August",
            "9": "September",
            "10": "October",
            "11": "November",
            "12": "December",
        }
        try:
            parts = fiscal_year_end.split("/")
            if len(parts) == 2:
                month, day = parts
                month_name = month_names.get(month, month)
                return f"{month_name} {day}"
        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse fiscal year end '{fiscal_year_end}': {e}")
        return fiscal_year_end  # Return as-is if parsing fails

    def _split_content_by_paragraphs(
        self,
        content: str,
        max_chunk_chars: int = MAX_CHUNK_CHARS_HIGH,
    ) -> list[str]:
        """Split content into chunks at natural paragraph breaks.

        Args:
            content: Full section content
            max_chunk_chars: Maximum characters per chunk (default ~62k tokens)

        Returns:
            List of content chunks
        """
        if len(content) <= max_chunk_chars:
            return [content]

        # Split by paragraph breaks (double newlines or --- separators)
        paragraphs = []
        for part in content.split("\n\n"):
            # Also split on our merged section separators
            for subpart in part.split("\n\n---\n\n"):
                if subpart.strip():
                    paragraphs.append(subpart.strip())

        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para) + 2  # +2 for \n\n separator

            # If single paragraph exceeds limit, split it by sentences
            if para_size > max_chunk_chars:
                # Flush current chunk first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sentences = para.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    if current_size + len(sentence) > max_chunk_chars and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    current_chunk.append(sentence)
                    current_size += len(sentence)
            elif current_size + para_size > max_chunk_chars:
                # Start new chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _combine_chunk_summaries(
        self,
        summaries: list[str],
        section_item: str,
        section_title: str,
        filing_context: str,
    ) -> str:
        """Combine multiple chunk summaries into a cohesive section summary.

        Args:
            summaries: List of summaries from each chunk
            section_item: Section identifier (e.g., "Item 1A")
            section_title: Section title
            filing_context: Context about the filing

        Returns:
            Combined summary text
        """
        if len(summaries) == 1:
            return summaries[0]

        combined_input = "\n\n---\n\n".join(
            f"**Part {i+1} of {len(summaries)}:**\n{s}" for i, s in enumerate(summaries)
        )

        system_prompt = f"""You are combining multiple partial summaries of SEC filing section {section_item} ({section_title}) into a single cohesive analysis.

Guidelines:
- Synthesize the key points from all parts without excessive repetition
- Maintain the analytical depth and investment focus
- Organize by theme rather than by part number
- Flag any contradictions or notable patterns across parts
- Keep the combined summary comprehensive but not redundant

Output a single unified analysis."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,  # Larger limit for combined summary
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""Filing: {filing_context}

Please combine these partial summaries into a unified section analysis:

{combined_input}""",
                }
            ],
        )

        return getattr(response.content[0], "text", "")

    async def _combine_chunk_summaries_async(
        self,
        summaries: list[str],
        section_item: str,
        section_title: str,
        filing_context: str,
    ) -> str:
        """Async version: Combine multiple chunk summaries into a cohesive section summary.

        Args:
            summaries: List of summaries from each chunk
            section_item: Section identifier (e.g., "Item 1A")
            section_title: Section title
            filing_context: Context about the filing

        Returns:
            Combined summary text
        """
        if len(summaries) == 1:
            return summaries[0]

        combined_input = "\n\n---\n\n".join(
            f"**Part {i+1} of {len(summaries)}:**\n{s}" for i, s in enumerate(summaries)
        )

        system_prompt = f"""You are combining multiple partial summaries of SEC filing section {section_item} ({section_title}) into a single cohesive analysis.

Guidelines:
- Synthesize the key points from all parts without excessive repetition
- Maintain the analytical depth and investment focus
- Organize by theme rather than by part number
- Flag any contradictions or notable patterns across parts
- Keep the combined summary comprehensive but not redundant

Output a single unified analysis."""

        response = await self.async_client.messages.create(
            model=self.model,
            max_tokens=4000,  # Larger limit for combined summary
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""Filing: {filing_context}

Please combine these partial summaries into a unified section analysis:

{combined_input}""",
                }
            ],
        )

        return getattr(response.content[0], "text", "")

    def _get_prompt_for_section(self, section_id: str, weight_override: str | None = None) -> str:
        """Get the full prompt for a section, including section-specific guidance."""
        config = self._get_section_config(section_id)
        weight = weight_override or config["weight"]
        guidance = config.get("guidance")
        return build_section_prompt(weight, guidance)

    def summarize_section(
        self,
        section: Section,
        filing_context: str = "",
        weight_override: str | None = None,
    ) -> SectionSummary:
        """
        Summarize a single section with weight-appropriate depth.

        For very long sections, splits content into chunks, summarizes each,
        then combines the summaries.

        Args:
            section: Section to summarize
            filing_context: Context about the filing (ticker, type, date)
            weight_override: Override the default weight for this section

        Returns:
            SectionSummary with condensed content
        """
        config = self._get_section_config(section.item)
        weight = weight_override or config["weight"]
        max_tokens = config["max_tokens"]
        prompt = self._get_prompt_for_section(section.item, weight_override)

        # Split very long sections into chunks
        max_chunk_chars = MAX_CHUNK_CHARS_HIGH if weight == "high" else MAX_CHUNK_CHARS_DEFAULT
        chunks = self._split_content_by_paragraphs(section.content, max_chunk_chars)

        if len(chunks) > 1:
            logger.debug(
                f"Section '{section.item}' split into {len(chunks)} chunks "
                f"(content length: {len(section.content):,} chars, max per chunk: {max_chunk_chars:,})"
            )

        total_input_tokens = 0
        total_output_tokens = 0
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            chunk_label = f" (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""

            user_message = f"""Filing: {filing_context}
Section: {section.item} - {section.title}{chunk_label}
Section Weight: {weight.upper()} priority

Content:
{chunk}

Please analyze this section according to the guidelines."""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            summary_text: str = getattr(response.content[0], "text", "")
            chunk_summaries.append(summary_text)
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            logger.debug(
                f"Summarized section '{section.item}'{chunk_label}: "
                f"input={response.usage.input_tokens:,} output={response.usage.output_tokens:,}"
            )

        # Combine chunk summaries if there were multiple
        if len(chunk_summaries) > 1:
            logger.debug(f"Combining {len(chunk_summaries)} chunk summaries for section '{section.item}'")
            final_summary = self._combine_chunk_summaries(chunk_summaries, section.item, section.title, filing_context)
            # Note: We don't track tokens from the combine call separately
        else:
            final_summary = chunk_summaries[0]

        logger.info(
            f"✓ Section '{section.item}' summarized: "
            f"total_input={total_input_tokens:,} total_output={total_output_tokens:,}"
        )

        return SectionSummary(
            section_id=section.item,
            section_title=section.title,
            summary=final_summary,
            weight=weight,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    async def summarize_section_async(
        self,
        section: Section,
        filing_context: str = "",
        weight_override: str | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> SectionSummary:
        """
        Async version of summarize_section for parallel processing.

        For very long sections, splits content into chunks, summarizes each,
        then combines the summaries.

        Args:
            section: Section to summarize
            filing_context: Context about the filing (ticker, type, date)
            weight_override: Override the default weight for this section
            semaphore: Optional semaphore to limit concurrent requests

        Returns:
            SectionSummary with condensed content
        """
        config = self._get_section_config(section.item)
        weight = weight_override or config["weight"]
        max_tokens = config["max_tokens"]
        prompt = self._get_prompt_for_section(section.item, weight_override)

        # Split very long sections into chunks
        max_chunk_chars = MAX_CHUNK_CHARS_HIGH if weight == "high" else MAX_CHUNK_CHARS_DEFAULT
        chunks = self._split_content_by_paragraphs(section.content, max_chunk_chars)

        if len(chunks) > 1:
            logger.debug(
                f"Section '{section.item}' split into {len(chunks)} chunks "
                f"(content length: {len(section.content):,} chars, max per chunk: {max_chunk_chars:,})"
            )

        total_input_tokens = 0
        total_output_tokens = 0
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            chunk_label = f" (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""

            user_message = f"""Filing: {filing_context}
Section: {section.item} - {section.title}{chunk_label}
Section Weight: {weight.upper()} priority

Content:
{chunk}

Please analyze this section according to the guidelines."""

            # Use semaphore to limit concurrent requests
            if semaphore:
                async with semaphore:
                    response = await self.async_client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=prompt,
                        messages=[{"role": "user", "content": user_message}],
                    )
            else:
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

            summary_text: str = getattr(response.content[0], "text", "")
            chunk_summaries.append(summary_text)
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        # Combine chunk summaries if there were multiple
        if len(chunk_summaries) > 1:
            final_summary = await self._combine_chunk_summaries_async(
                chunk_summaries, section.item, section.title, filing_context
            )
            # Note: We don't track tokens from the combine call separately
        else:
            final_summary = chunk_summaries[0]

        return SectionSummary(
            section_id=section.item,
            section_title=section.title,
            summary=final_summary,
            weight=weight,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def create_executive_summary(
        self,
        section_summaries: list[SectionSummary],
        filing_context: str,
        include_thesis: bool = True,
    ) -> tuple[str, int, int]:
        """
        Create executive synthesis from section summaries.

        Args:
            section_summaries: List of section summaries
            filing_context: Context about the filing
            include_thesis: If True, use full investment thesis prompt (10-K).
                          If False, use simpler transcript prompt (no bull/bear case).

        Returns:
            Tuple of (summary_text, input_tokens, output_tokens)
        """
        # Organize summaries by weight for context
        sections_text = ""

        high = [s for s in section_summaries if s.weight == "high"]
        medium = [s for s in section_summaries if s.weight == "medium"]
        low = [s for s in section_summaries if s.weight == "low"]

        if high:
            sections_text += "## KEY SECTIONS (High Priority)\n\n"
            sections_text += "\n\n".join([f"### {s.section_id}: {s.section_title}\n{s.summary}" for s in high])
            sections_text += "\n\n"

        if medium:
            sections_text += "## SUPPORTING SECTIONS\n\n"
            sections_text += "\n\n".join([f"### {s.section_id}: {s.section_title}\n{s.summary}" for s in medium])
            sections_text += "\n\n"

        if low:
            sections_text += "## OTHER SECTIONS (Brief)\n\n"
            sections_text += "\n\n".join([f"### {s.section_id}: {s.section_title}\n{s.summary}" for s in low])

        user_message = f"""Filing: {filing_context}

{sections_text}

Please create an executive synthesis that connects these analyses into actionable investment insight."""

        # Select prompt based on include_thesis
        prompt = EXECUTIVE_SUMMARY_PROMPT if include_thesis else TRANSCRIPT_SUMMARY_PROMPT

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract text from TextBlock
        exec_text: str = getattr(response.content[0], "text", "")

        return (
            exec_text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    async def create_executive_summary_async(
        self,
        section_summaries: list[SectionSummary],
        filing_context: str,
        include_thesis: bool = True,
        semaphore: asyncio.Semaphore | None = None,
    ) -> tuple[str, int, int]:
        """
        Async version of create_executive_summary for parallel processing.

        Args:
            section_summaries: List of section summaries
            filing_context: Context about the filing
            include_thesis: If True, use full investment thesis prompt (10-K).
                          If False, use simpler transcript prompt (no bull/bear case).
            semaphore: Optional semaphore for concurrency control

        Returns:
            Tuple of (summary_text, input_tokens, output_tokens)
        """
        # Organize summaries by weight for context
        sections_text = ""

        high = [s for s in section_summaries if s.weight == "high"]
        medium = [s for s in section_summaries if s.weight == "medium"]
        low = [s for s in section_summaries if s.weight == "low"]

        if high:
            sections_text += "## KEY SECTIONS (High Priority)\n\n"
            sections_text += "\n\n".join([f"### {s.section_id}: {s.section_title}\n{s.summary}" for s in high])
            sections_text += "\n\n"

        if medium:
            sections_text += "## SUPPORTING SECTIONS\n\n"
            sections_text += "\n\n".join([f"### {s.section_id}: {s.section_title}\n{s.summary}" for s in medium])
            sections_text += "\n\n"

        if low:
            sections_text += "## OTHER SECTIONS (Brief)\n\n"
            sections_text += "\n\n".join([f"### {s.section_id}: {s.section_title}\n{s.summary}" for s in low])

        user_message = f"""Filing: {filing_context}

{sections_text}

Please create an executive synthesis that connects these analyses into actionable investment insight."""

        # Select prompt based on include_thesis
        prompt = EXECUTIVE_SUMMARY_PROMPT if include_thesis else TRANSCRIPT_SUMMARY_PROMPT

        if semaphore:
            async with semaphore:
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=3000,
                    system=prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
        else:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=3000,
                system=prompt,
                messages=[{"role": "user", "content": user_message}],
            )

        # Extract text from TextBlock
        exec_text: str = getattr(response.content[0], "text", "")

        return (
            exec_text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def summarize_filing(
        self,
        file_path: Path,
        ticker: str,
        filing_type: str,
        filing_date: str,
        sections_to_include: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        parallel: bool = True,
    ) -> FilingSummary:
        """
        Summarize an entire SEC filing with weighted section analysis.

        Args:
            file_path: Path to the filing (HTML)
            ticker: Company ticker
            filing_type: Type of filing (10-K, 10-Q, etc.)
            filing_date: Filing date
            sections_to_include: Optional list of sections to summarize
            progress_callback: Optional callback(current, total, section_name)
            parallel: Use parallel API calls for faster processing (default: True)

        Returns:
            FilingSummary with all summaries
        """
        # Parse the filing
        parsed = self.parser.parse_file(
            file_path,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
        )

        # Build filing context with fiscal period info to prevent quarter confusion
        # e.g., "Company: DT (ticker symbol) | Filing: 10-Q Q2 FY2026 | Date: 2025-11-05 | FY End: March 31"
        filing_context = f"Company: {ticker} (ticker symbol) | Filing: {filing_type}"
        if parsed.fiscal_period and parsed.fiscal_year:
            filing_context += f" {parsed.fiscal_period} FY{parsed.fiscal_year}"
        filing_context += f" | Date: {filing_date}"
        if parsed.fiscal_year_end:
            # Convert "3/31" to "March 31" for clarity
            fiscal_end_display = self._format_fiscal_year_end(parsed.fiscal_year_end)
            filing_context += f" | FY End: {fiscal_end_display}"

        # Filter sections if specified
        sections = parsed.sections
        if sections_to_include:
            sections = [s for s in sections if s.item in sections_to_include]

        if not sections:
            raise ValueError(f"No sections found in filing: {file_path}")

        # Sort sections by weight (high first) for better progress feedback
        def section_sort_key(s: Section) -> int:
            weight = self._get_section_config(s.item)["weight"]
            return {"high": 0, "medium": 1, "low": 2}.get(weight, 1)

        sections = sorted(sections, key=section_sort_key)

        # Summarize sections (parallel or sequential)
        if parallel:
            section_summaries = self._summarize_sections_parallel(sections, filing_context, progress_callback)
        else:
            section_summaries = self._summarize_sections_sequential(sections, filing_context, progress_callback)

        total_input = sum(s.input_tokens for s in section_summaries)
        total_output = sum(s.output_tokens for s in section_summaries)

        # Create executive summary
        if progress_callback:
            progress_callback(len(sections) + 1, len(sections) + 1, "Executive Synthesis")

        exec_summary, exec_input, exec_output = self.create_executive_summary(section_summaries, filing_context)
        total_input += exec_input
        total_output += exec_output

        return FilingSummary(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            source_path=str(file_path),
            executive_summary=exec_summary,
            section_summaries=section_summaries,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
        )

    def _summarize_sections_sequential(
        self,
        sections: list[Section],
        filing_context: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[SectionSummary]:
        """Summarize sections one at a time (original behavior)."""
        section_summaries = []

        for i, section in enumerate(sections):
            config = self._get_section_config(section.item)
            if progress_callback:
                progress_callback(i + 1, len(sections) + 1, f"{section.item} ({config['weight']})")

            summary = self.summarize_section(section, filing_context)
            section_summaries.append(summary)

        return section_summaries

    def _summarize_sections_parallel(
        self,
        sections: list[Section],
        filing_context: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[SectionSummary]:
        """Summarize sections in parallel using async API calls."""

        async def run_parallel():
            semaphore = asyncio.Semaphore(self.max_concurrent)
            completed = 0

            async def summarize_with_progress(section: Section) -> SectionSummary:
                nonlocal completed
                result = await self.summarize_section_async(section, filing_context, semaphore=semaphore)
                completed += 1
                if progress_callback:
                    config = self._get_section_config(section.item)
                    progress_callback(completed, len(sections) + 1, f"{section.item} ({config['weight']})")
                return result

            # Create tasks for all sections
            tasks = [summarize_with_progress(section) for section in sections]

            # Run all tasks concurrently (limited by semaphore)
            results = await asyncio.gather(*tasks)
            return list(results)

        # Run the async function
        return asyncio.run(run_parallel())

    def get_output_path(self, ticker: str, filing_type: str, filing_date: str) -> Path:
        """Get the output path for a summary."""
        if self.processed_dir is None:
            raise ValueError("processed_dir not set")

        filename = f"{filing_type}_{filing_date}_summary.md"
        return self.processed_dir / ticker.upper() / filename

    def prepare_batch_requests(
        self,
        file_path: Path,
        ticker: str,
        filing_type: str,
        filing_date: str,
        sections_to_include: list[str] | None = None,
        token_multiplier: float = 1.0,
    ) -> list[dict]:
        """
        Prepare batch API requests for all sections.

        Returns list of request objects for the Batch API.
        Use this for 50% cost savings on large summarization jobs.

        Long sections (>80k chars for high-priority, >50k for others) are automatically
        split into multiple requests with _part1, _part2, etc. suffixes. Results are
        combined during retrieval.

        Token limits are set to 60% of input tokens (estimated at 4 chars/token) with
        no artificial caps, trusting prompts to keep output concise.

        Args:
            file_path: Path to the filing
            ticker: Company ticker
            filing_type: Type of filing
            filing_date: Filing date
            sections_to_include: Optional list of sections
            token_multiplier: Factor to increase token limits (for resubmit, default 1.0)

        Returns:
            List of batch request dicts with custom_id and params
        """
        parsed = self.parser.parse_file(
            file_path,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
        )

        # Build filing context with fiscal period info to prevent quarter confusion
        filing_context = f"Company: {ticker} (ticker symbol) | Filing: {filing_type}"
        if parsed.fiscal_period and parsed.fiscal_year:
            filing_context += f" {parsed.fiscal_period} FY{parsed.fiscal_year}"
        filing_context += f" | Date: {filing_date}"
        if parsed.fiscal_year_end:
            fiscal_end_display = self._format_fiscal_year_end(parsed.fiscal_year_end)
            filing_context += f" | FY End: {fiscal_end_display}"

        sections = parsed.sections
        if sections_to_include:
            sections = [s for s in sections if s.item in sections_to_include]

        requests = []
        for section in sections:
            config = self._get_section_config(section.item)
            weight = config["weight"]
            base_max_tokens = config["max_tokens"]
            prompt = self._get_prompt_for_section(section.item)

            # Split long sections into chunks instead of truncating
            max_chunk_chars = 80000 if weight == "high" else 50000
            chunks = self._split_content_by_paragraphs(section.content, max_chunk_chars)

            for chunk_idx, chunk_content in enumerate(chunks):
                # Token estimation: trust prompts to keep output lean
                estimated_input_tokens = len(chunk_content) // 4
                max_tokens = max(base_max_tokens, int(estimated_input_tokens * 0.6))

                # Apply multiplier for resubmits
                if token_multiplier > 1.0:
                    max_tokens = int(max_tokens * token_multiplier)

                # Only hard cap at API limit
                max_tokens = min(max_tokens, 32000)

                # Add part label if multiple chunks
                chunk_label = f" (Part {chunk_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""

                user_message = f"""Filing: {filing_context}
Section: {section.item} - {section.title}{chunk_label}
Section Weight: {weight.upper()} priority

Content:
{chunk_content}

Please analyze this section according to the guidelines."""

                safe_section = sanitize_custom_id_component(section.item)
                # Include part number in custom_id for multi-chunk sections
                if len(chunks) > 1:
                    custom_id = f"{ticker}_{filing_type}_{filing_date}_{safe_section}_part{chunk_idx + 1}"
                else:
                    custom_id = f"{ticker}_{filing_type}_{filing_date}_{safe_section}"

                requests.append(
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "system": prompt,
                            "messages": [{"role": "user", "content": user_message}],
                        },
                    }
                )

        return requests

    def prepare_batch_requests_pdf(
        self,
        file_path: Path,
        ticker: str,
        doc_type: str,
        doc_date: str,
        sections_to_include: list[str] | None = None,
        token_multiplier: float = 1.0,
    ) -> list[dict]:
        """
        Prepare batch API requests for PDF documents (transcripts, presentations).

        Long sections (>80k chars for high-priority, >50k for others) are automatically
        split into multiple requests with _part1, _part2, etc. suffixes. Results are
        combined during retrieval.

        Token limits are set to 60% of input tokens (estimated at 4 chars/token) with
        no artificial caps, trusting prompts to keep output concise.

        Args:
            file_path: Path to the PDF
            ticker: Company ticker
            doc_type: Document type (transcript, presentation, conference)
            doc_date: Document date
            sections_to_include: Optional list of section names to include (for resubmit)
            token_multiplier: Factor to increase token limits (for resubmit, default 1.0)

        Returns:
            List of batch request dicts
        """
        import fitz  # PyMuPDF

        requests = []

        if doc_type == "transcript":
            from src.rag.transcript_parser import TranscriptParser

            parser = TranscriptParser()
            parsed = parser.parse_file(file_path, ticker)

            # Map parser section names to SECTION_WEIGHTS keys
            section_name_map = {
                "presentation": "Prepared Remarks",
                "qa": "Q&A",
            }

            # Create sections from transcript
            sections_data = []
            for parser_section, weight_section in section_name_map.items():
                blocks = [b for b in parsed.all_blocks if b.section == parser_section]
                if blocks:
                    content = "\n\n".join(
                        [f"**{b.speaker.name}** ({b.speaker.title or 'N/A'}):\n{b.text}" for b in blocks]
                    )
                    sections_data.append((weight_section, content))

            filing_context = f"Company: {ticker} | Document: Earnings Call Transcript | Date: {doc_date}"

        elif doc_type == "presentation":
            # Extract full text from presentation PDF
            doc = fitz.open(file_path)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()

            # Single section for presentation
            sections_data = [("Presentation Full", full_text)]
            filing_context = f"Company: {ticker} | Document: Investor Presentation | Date: {doc_date}"

        elif doc_type == "conference":
            # Conference transcripts - similar to earnings transcripts
            from src.rag.transcript_parser import TranscriptParser

            parser = TranscriptParser()
            parsed = parser.parse_file(file_path, ticker)

            # Map parser section names to SECTION_WEIGHTS keys
            section_name_map = {
                "presentation": "Conference Overview",
                "qa": "Conference Q&A",
            }

            sections_data = []
            for parser_section, weight_section in section_name_map.items():
                blocks = [b for b in parsed.all_blocks if b.section == parser_section]
                if blocks:
                    content = "\n\n".join(
                        [f"**{b.speaker.name}** ({b.speaker.title or 'N/A'}):\n{b.text}" for b in blocks]
                    )
                    sections_data.append((weight_section, content))

            filing_context = f"Company: {ticker} | Document: Conference Transcript | Date: {doc_date}"

        else:
            # Generic PDF - extract full text
            doc = fitz.open(file_path)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()
            sections_data = [("Full Document", full_text)]
            filing_context = f"Company: {ticker} | Document: {doc_type.title()} | Date: {doc_date}"

        # Filter sections if specified (for resubmit)
        # Compare using sanitized names to handle "Q&A" vs "Q and A" etc.
        if sections_to_include:
            sanitized_includes = {
                sanitize_custom_id_component(s).replace("-", " ").lower() for s in sections_to_include
            }
            sections_data = [
                (name, content)
                for name, content in sections_data
                if (
                    name in sections_to_include
                    or sanitize_custom_id_component(name).replace("-", " ").lower() in sanitized_includes
                )
            ]

        # Create batch requests for each section
        for section_name, content in sections_data:
            config = self._get_section_config(section_name)
            base_max_tokens = config.get("max_tokens", 2000)
            prompt = self._get_prompt_for_section(section_name)

            # Split long sections into chunks instead of truncating
            max_chunk_chars = MAX_CHUNK_CHARS_HIGH
            chunks = self._split_content_by_paragraphs(content, max_chunk_chars)

            for chunk_idx, chunk_content in enumerate(chunks):
                # Token estimation: trust prompts to keep output lean
                estimated_input_tokens = len(chunk_content) // 4
                max_tokens = max(base_max_tokens, int(estimated_input_tokens * 0.6))

                # Apply multiplier for resubmits
                if token_multiplier > 1.0:
                    max_tokens = int(max_tokens * token_multiplier)

                # Only hard cap at API limit
                max_tokens = min(max_tokens, 32000)

                # Add part label if multiple chunks
                chunk_label = f" (Part {chunk_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""

                user_message = f"""Document: {filing_context}
Section: {section_name}{chunk_label}

Content:
{chunk_content}

Please analyze this section according to the guidelines."""

                safe_section = sanitize_custom_id_component(section_name)
                # Include part number in custom_id for multi-chunk sections
                if len(chunks) > 1:
                    custom_id = f"{ticker}_{doc_type}_{doc_date}_{safe_section}_part{chunk_idx + 1}"
                else:
                    custom_id = f"{ticker}_{doc_type}_{doc_date}_{safe_section}"

                requests.append(
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "system": prompt,
                            "messages": [{"role": "user", "content": user_message}],
                        },
                    }
                )

        return requests


def summarize_filing(
    file_path: Path,
    ticker: str,
    filing_type: str,
    filing_date: str,
    output_path: Path | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_SUMMARY_MODEL,
) -> FilingSummary:
    """
    Convenience function to summarize a filing.

    Args:
        file_path: Path to the filing
        ticker: Company ticker
        filing_type: Type of filing
        filing_date: Filing date
        output_path: Optional path to save summary
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        FilingSummary
    """
    summarizer = Summarizer(api_key=api_key, model=model)
    result = summarizer.summarize_filing(
        file_path=file_path,
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
    )

    if output_path:
        result.save(output_path)

    return result
