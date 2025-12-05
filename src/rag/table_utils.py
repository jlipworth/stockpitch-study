"""Table extraction, conversion, and classification utilities.

This module provides utilities for:
- Extracting tables from HTML (SEC filings, etc.)
- Converting HTML tables to markdown or readable text
- Classifying table types (financial, compensation, governance, etc.)
- Table markers for chunk identification
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Table marker constants for detection in chunked text
TABLE_START_MARKER = "[TABLE:"
TABLE_END_MARKER = "[/TABLE]"


# Table type classification patterns
# Priority order: compensation, governance > financial_statement > metrics, schedule > comparison > other
# Higher-weight patterns are more specific and should win ties
TABLE_TYPE_PATTERNS = {
    "compensation": [
        r"(?:executive|director)\s+compensation",
        r"summary\s+compensation\s+table",
        r"(?:stock|equity)\s+(?:awards?|options?|grants?)",
        r"(?:salary|bonus|incentive)\s+(?:plan|award)",
        r"(?:named\s+executive|NEO)",
        r"(?:pension|retirement)\s+(?:plan|benefit)",
        r"pay\s+(?:versus|vs\.?)\s+performance",
        r"(?:annual|long.?term)\s+incentive",
        r"(?:base\s+)?salary",
        r"(?:non.?equity|cash)\s+incentive",
    ],
    "governance": [
        r"board\s+(?:of\s+)?directors?",
        r"(?:director|committee)\s+(?:independence|composition)",
        r"audit\s+committee",
        r"compensation\s+committee",
        r"nominating\s+(?:and\s+)?governance",
        r"(?:corporate\s+)?governance",
        r"(?:stock|share)\s+ownership\s+(?:guidelines?|requirements?)",
        r"(?:director|board)\s+(?:nominees?|elections?)",
    ],
    "financial_statement": [
        r"(?:income|revenue|earnings)\s+statement",
        r"balance\s+sheet",
        r"cash\s+flow",
        r"statement\s+of\s+(?:operations|income|comprehensive)",
        r"consolidated\s+(?:statements?|financial)",
        r"(?:total\s+)?(?:assets|liabilities)\b",  # Removed equity to avoid conflict
        r"(?:net\s+)?(?:revenue|sales|income)\b",
    ],
    "metrics": [
        r"key\s+(?:performance\s+)?(?:metrics?|indicators?|KPIs?)",
        r"(?:operating|financial)\s+(?:metrics?|highlights?)",
        r"(?:ARR|MRR|NRR|ARPU|CAC|LTV)\b",
        r"(?:customer|subscriber)\s+(?:metrics?|count|growth)",
    ],
    "schedule": [
        r"schedule\s+(?:of|to)",
        r"(?:debt|loan|lease)\s+(?:maturity|schedule)",
        r"(?:amortization|depreciation)\s+schedule",
        r"(?:contractual|payment)\s+obligations?",
        r"(?:future|minimum)\s+(?:payments?|commitments?)",
        r"maturity\s+schedule",
    ],
    "comparison": [
        # More specific patterns to avoid false positives from bare years
        r"(?:year|quarter|period)\s+(?:ended|ending)",
        r"(?:three|six|nine|twelve)\s+months?\s+ended",
        r"(?:vs\.?|versus)\s+(?:prior|previous)",
        r"compared\s+to\s+(?:prior|previous)",
        r"(?:change|growth)\s+(?:over|from)\s+(?:prior|previous)",
        r"year.over.year",
        r"quarter.over.quarter",
    ],
}


@dataclass
class Table:
    """Represents an extracted table with metadata and classification.

    Phase 2 Table Intelligence: Stores tables with HTML source for rendering,
    markdown for search, and classification for smart filtering.
    """

    table_id: str  # Unique ID: "{source_hash}_table_{position}"
    html: str  # Original HTML for rendering
    markdown: str  # Clean text for search/embedding
    caption: str  # Table title/context
    position: int  # Position in document (0-indexed)
    num_rows: int
    num_cols: int
    # Classification
    table_type: str = "other"  # financial_statement, compensation, governance, comparison, schedule, metrics, other
    # Token estimate for chunking decisions
    approx_tokens: int = 0  # Approximate token count

    @classmethod
    def from_html(cls, html: str, caption: str, position: int, source_hash: str = "") -> "Table":
        """Create Table from HTML string."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        table_elem = soup.find("table")

        if not table_elem:
            # HTML might be the table directly
            table_elem = soup

        # Count rows and cols
        rows = table_elem.find_all("tr") if table_elem else []
        num_rows = len(rows)
        num_cols = 0
        if rows:
            first_row = rows[0]
            cells = first_row.find_all(["td", "th"])
            num_cols = len(cells)

        # Generate markdown
        markdown = _table_to_readable_text(table_elem) if table_elem else ""

        # Estimate tokens (chars / 4)
        approx_tokens = len(markdown) // 4

        # Classify table
        table_type = classify_table(caption, markdown)

        table_id = f"{source_hash}_table_{position}" if source_hash else f"table_{position}"

        return cls(
            table_id=table_id,
            html=html,
            markdown=markdown,
            caption=caption,
            position=position,
            num_rows=num_rows,
            num_cols=num_cols,
            table_type=table_type,
            approx_tokens=approx_tokens,
        )


def classify_table(caption: str, content: str) -> str:
    """
    Classify table type based on caption and content.

    Uses keyword matching on caption + content (headers, first rows).

    Args:
        caption: Table caption/title
        content: Table markdown/text content

    Returns:
        Table type: financial_statement, compensation, governance, comparison, schedule, metrics, other
    """
    # Combine caption and first portion of content for matching
    text_to_match = f"{caption}\n{content[:2000]}".lower()

    # Score each type
    scores = {}
    for table_type, patterns in TABLE_TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_to_match, re.IGNORECASE):
                score += 1
        scores[table_type] = score

    # Return type with highest score, or "other" if no matches
    if max(scores.values()) == 0:
        return "other"

    return max(scores, key=lambda k: scores[k])


def extract_tables(soup) -> list[dict]:
    """
    Extract tables from HTML as structured data.

    Args:
        soup: Parsed BeautifulSoup object

    Returns:
        List of dicts with table data
    """
    tables = []

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(cells)

        if rows:
            tables.append(
                {
                    "rows": rows,
                    "num_rows": len(rows),
                    "num_cols": max(len(r) for r in rows) if rows else 0,
                }
            )

    return tables


def _get_table_context(table) -> str:
    """
    Extract context for a table (caption, preceding header, or surrounding text).

    Args:
        table: BeautifulSoup table element

    Returns:
        Context string describing the table
    """
    context_parts = []

    # Check for <caption> element
    caption = table.find("caption")
    if caption:
        context_parts.append(caption.get_text(strip=True))

    # Check preceding siblings for headers or descriptive text
    for sibling in table.find_previous_siblings(limit=3):
        if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "b", "strong"]:
            text = sibling.get_text(strip=True)
            if text and len(text) < 200:  # Reasonable header length
                context_parts.insert(0, text)
                break

    return " - ".join(context_parts) if context_parts else ""


def _table_to_markdown(table) -> str:
    """
    Convert an HTML table to markdown format with context.

    Args:
        table: BeautifulSoup table element

    Returns:
        Markdown string representation of the table
    """
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            # Clean cell text: remove newlines, normalize whitespace
            cell_text = td.get_text(strip=True)
            cell_text = re.sub(r"\s+", " ", cell_text)
            # Escape pipe characters in cell content
            cell_text = cell_text.replace("|", "\\|")
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # Determine column count (handle ragged tables)
    num_cols = max(len(row) for row in rows)

    # Pad rows to consistent length
    for row in rows:
        while len(row) < num_cols:
            row.append("")

    # Build markdown table
    md_lines = []

    # Get table context/title
    context = _get_table_context(table)
    if context:
        md_lines.append(f"**{context}**")
        md_lines.append("")

    # Header row (first row)
    if rows:
        md_lines.append("| " + " | ".join(rows[0]) + " |")
        md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        # Data rows
        for row in rows[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("")  # Blank line after table
    return "\n".join(md_lines)


def _table_to_readable_text(table) -> str:
    """
    Convert HTML table to clean readable text format.

    Uses pandas for robust HTML parsing, then cleans up:
    - Removes spacer columns from SEC formatting
    - Merges currency symbols with values
    - Removes duplicate columns from colspan expansion
    - Produces readable, searchable text

    Args:
        table: BeautifulSoup table element

    Returns:
        Clean readable text representation
    """
    from io import StringIO

    import pandas as pd

    try:
        # Get HTML string for pandas
        html_str = str(table)
        dfs = pd.read_html(StringIO(html_str))
        if not dfs:
            return ""
        df = dfs[0]
    except Exception as e:
        # Fallback to simple text extraction if pandas fails
        logger.debug("pandas read_html failed, using text fallback: %s", e)
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append("  ".join(c for c in cells if c))
        return "\n".join(rows)

    # Clean NaN values
    df = df.fillna("")
    df = df.astype(str).replace("nan", "")

    lines = []
    for row_idx in range(len(df)):
        row_vals = df.iloc[row_idx].tolist()

        # Remove consecutive duplicates (from colspan expansion)
        unique_vals = []
        prev = None
        for v in row_vals:
            v = str(v).strip()
            if v and v != prev:
                unique_vals.append(v)
            prev = v

        if not unique_vals:
            continue

        # Merge currency symbols with following numbers
        merged = []
        i = 0
        while i < len(unique_vals):
            v = unique_vals[i]
            # Check if currency followed by number
            if v in ("$", "€", "£", "¥") and i + 1 < len(unique_vals):
                next_v = unique_vals[i + 1]
                # Check if next looks like a number (possibly with commas, parens)
                cleaned = next_v.replace(",", "").replace("(", "").replace(")", "")
                if cleaned.replace(".", "").replace("-", "").isdigit():
                    merged.append(f"{v}{next_v}")
                    i += 2
                    continue
            merged.append(v)
            i += 1

        if merged:
            lines.append("  ".join(merged))

    return "\n".join(lines)


def render_table_as_markdown(html: str) -> str:
    """
    Render HTML table as clean, aligned markdown grid.

    For display purposes - use when showing search results.
    Uses same cleanup logic as _table_to_readable_text but outputs markdown.

    Args:
        html: HTML table string

    Returns:
        Aligned markdown table string
    """
    from io import StringIO

    import pandas as pd

    try:
        dfs = pd.read_html(StringIO(html))
        if not dfs:
            return ""
        df = dfs[0]
    except Exception as e:
        logger.debug("Failed to render table as markdown: %s", e)
        return ""

    df = df.fillna("")
    df = df.astype(str).replace("nan", "")

    # Process each row - remove duplicates, merge currency
    processed_rows = []
    for row_idx in range(len(df)):
        row_vals = df.iloc[row_idx].tolist()

        # Remove consecutive duplicates (from colspan expansion)
        unique_vals = []
        prev = None
        for v in row_vals:
            v = str(v).strip()
            if v and v != prev:
                unique_vals.append(v)
            prev = v

        if not unique_vals:
            continue

        # Merge currency symbols with following numbers
        merged = []
        i = 0
        while i < len(unique_vals):
            v = unique_vals[i]
            if v in ("$", "€", "£", "¥") and i + 1 < len(unique_vals):
                next_v = unique_vals[i + 1]
                cleaned = next_v.replace(",", "").replace("(", "").replace(")", "")
                if cleaned.replace(".", "").replace("-", "").isdigit():
                    merged.append(f"{v}{next_v}")
                    i += 2
                    continue
            merged.append(v)
            i += 1

        if merged:
            processed_rows.append(merged)

    if not processed_rows:
        return ""

    # Normalize column count (pad shorter rows)
    max_cols = max(len(row) for row in processed_rows)
    for row in processed_rows:
        while len(row) < max_cols:
            row.append("")

    # Calculate column widths for alignment
    col_widths = []
    for col_idx in range(max_cols):
        max_width = max(len(row[col_idx]) for row in processed_rows)
        col_widths.append(max(max_width, 3))  # Minimum 3 for "---"

    # Build aligned markdown
    lines = []

    # Header (first row)
    header = [processed_rows[0][i].ljust(col_widths[i]) for i in range(max_cols)]
    lines.append("| " + " | ".join(header) + " |")

    # Separator
    sep = ["-" * col_widths[i] for i in range(max_cols)]
    lines.append("| " + " | ".join(sep) + " |")

    # Data rows
    for row in processed_rows[1:]:
        cells = [row[i].ljust(col_widths[i]) for i in range(max_cols)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def convert_tables_to_readable(soup, source_hash: str = "") -> list[Table]:
    """
    Convert all tables in soup to readable text format with markers.

    Replaces <table> elements with readable text wrapped in markers,
    and returns Table objects for storage.

    Phase 2: Now returns proper Table dataclass objects with classification.

    Args:
        soup: BeautifulSoup object (modified in place)
        source_hash: Optional source hash for Table ID generation

    Returns:
        List of Table objects with html, caption, classification, etc.
    """
    tables = []

    for idx, table in enumerate(soup.find_all("table")):
        # Skip very small tables (likely formatting artifacts)
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Get caption/context
        caption = _get_table_context(table)
        if not caption:
            caption = f"Table {idx + 1}"

        # Store original HTML
        original_html = str(table)

        # Convert to readable text
        readable = _table_to_readable_text(table)

        if readable:
            # Wrap with markers
            marked_text = f"\n\n{TABLE_START_MARKER} {caption}]\n{readable}\n{TABLE_END_MARKER}\n\n"

            # Create replacement tag
            new_tag = soup.new_tag("div")
            new_tag.string = marked_text
            table.replace_with(new_tag)

            # Create Table object with classification
            table_obj = Table(
                table_id=f"{source_hash}_table_{idx}" if source_hash else f"table_{idx}",
                html=original_html,
                markdown=readable,
                caption=caption,
                position=idx,
                num_rows=len(rows),
                num_cols=len(rows[0].find_all(["td", "th"])) if rows else 0,
                table_type=classify_table(caption, readable),
                approx_tokens=len(readable) // 4,
            )
            tables.append(table_obj)

    return tables


def convert_tables_to_markdown(soup) -> None:
    """
    Convert all tables in soup to markdown format in-place.

    Replaces <table> elements with their markdown representation
    so that get_text() preserves table structure.

    Args:
        soup: BeautifulSoup object (modified in place)
    """
    for table in soup.find_all("table"):
        # Skip very small tables (likely formatting artifacts)
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Convert to markdown
        md_text = _table_to_markdown(table)

        if md_text:
            # Create a new tag with the markdown text
            # Using <pre> to preserve formatting in get_text()
            new_tag = soup.new_tag("div")
            new_tag.string = f"\n\n{md_text}\n\n"
            table.replace_with(new_tag)
