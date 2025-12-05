#!/usr/bin/env python3
"""
Extract diverse tables from DT SEC filings for table intelligence test corpus.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


def html_table_to_markdown(table_elem) -> str:
    """Convert HTML table to markdown (simplified version)."""
    rows = table_elem.find_all("tr")
    if not rows:
        return ""

    md_lines = []
    max_cols = 0

    # First pass: determine max columns
    for row in rows:
        cells = row.find_all(["td", "th"])
        max_cols = max(max_cols, len(cells))

    # Second pass: build markdown
    for i, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        cell_texts = []
        for cell in cells:
            text = cell.get_text(strip=True)
            text = text.replace("|", "\\|")  # Escape pipes
            cell_texts.append(text)

        # Pad to max columns
        while len(cell_texts) < max_cols:
            cell_texts.append("")

        md_lines.append("| " + " | ".join(cell_texts) + " |")

        # Add header separator after first row
        if i == 0:
            md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    return "\n".join(md_lines)


def classify_table_type(caption: str, html: str, section: str) -> str:
    """Classify table type based on caption, content, and context."""
    caption_lower = caption.lower()
    _ = html.lower()  # Reserved for future content-based classification

    # Financial statements
    if any(
        term in caption_lower
        for term in [
            "balance sheet",
            "statements of operations",
            "income statement",
            "cash flow",
            "stockholders equity",
        ]
    ):
        return "financial_statement"

    # Compensation tables
    if any(
        term in caption_lower
        for term in ["compensation", "stock awards", "option", "equity", "salary", "bonus", "executive"]
    ):
        return "compensation"

    # Segment data
    if any(term in caption_lower for term in ["segment", "geographic", "product line"]):
        return "segment_data"

    # Schedule (tax, debt, etc.)
    if any(term in caption_lower for term in ["schedule", "reconciliation", "rollforward"]):
        return "schedule"

    # Risk factors / voting
    if any(term in caption_lower for term in ["vote", "proposal", "director", "nominee"]):
        return "governance"

    # Metrics / KPI table
    if any(term in caption_lower for term in ["metrics", "kpi", "summary", "highlights"]):
        return "metrics"

    # Default to comparison if has multiple columns with years/periods
    if re.search(r"20\d{2}", html):
        return "comparison"

    return "other"


def extract_section_from_html(soup, table_elem) -> str:
    """Try to extract the section/item where table appears."""
    # Look for nearest heading before the table
    for tag in ["h1", "h2", "h3", "h4"]:
        previous_headings = table_elem.find_all_previous(tag)
        if previous_headings:
            heading_text = previous_headings[0].get_text(strip=True)
            # Look for Item X patterns
            if re.search(r"Item\s+\d+[A-Z]?", heading_text, re.IGNORECASE):
                return heading_text[:100]  # Truncate if too long
            # Or Part X patterns
            if re.search(r"Part\s+[IVX]+", heading_text, re.IGNORECASE):
                return heading_text[:100]

    return "Unknown section"


def get_table_dimensions(table_elem) -> tuple:
    """Get approximate rows x cols for table."""
    rows = table_elem.find_all("tr")
    if not rows:
        return (0, 0)

    max_cols = 0
    for row in rows:
        cells = row.find_all(["td", "th"])
        # Account for colspan
        col_count = 0
        for cell in cells:
            colspan = cell.get("colspan", "1")
            try:
                col_count += int(colspan)
            except ValueError:
                col_count += 1
        max_cols = max(max_cols, col_count)

    return (len(rows), max_cols)


def extract_table_caption(table_elem) -> str:
    """Extract table caption or nearby heading."""
    # Check for caption tag
    caption = table_elem.find("caption")
    if caption:
        text = caption.get_text(strip=True)
        if len(text) > 5:
            return text

    # Check for title attribute
    title = table_elem.get("title")
    if title and len(title) > 5:
        return title

    # Check for previous paragraph or div with table title (look back further)
    for prev in table_elem.find_all_previous(["p", "div", "span", "b", "strong"]):
        text = prev.get_text(strip=True)
        # If short enough and looks like a title
        if 10 < len(text) < 300 and not text.startswith("("):
            # Skip if it looks like part of table content
            if not re.search(r"\d{4}", text[:20]):  # Skip if starts with year
                return text

    # Check first row if it looks like a title row
    first_row = table_elem.find("tr")
    if first_row:
        cells = first_row.find_all(["td", "th"])
        if len(cells) == 1:
            text = cells[0].get_text(strip=True)
            if len(text) > 5:
                return text

    # Try to get from parent div class or id
    parent = table_elem.find_parent(["div", "section"])
    if parent:
        div_id = parent.get("id", "")
        if div_id and "table" in div_id.lower():
            # Clean up ID to make it readable
            clean_id = div_id.replace("_", " ").title()
            return clean_id

    return "Untitled table"


def extract_tables_from_file(file_path: Path) -> list[dict[str, Any]]:
    """Extract all tables from a single HTML filing."""
    with open(file_path, encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "lxml")
    tables = soup.find_all("table")

    results = []

    # Parse filename: 10-K_2024-05-23_accession.html
    filename_parts = file_path.stem.split("_")
    filing_type = filename_parts[0]
    filing_date = filename_parts[1] if len(filename_parts) > 1 else "unknown"

    for idx, table in enumerate(tables, 1):
        # Get dimensions
        rows, cols = get_table_dimensions(table)

        # Skip tiny tables (likely layout tables)
        if rows < 2 or cols < 2:
            continue

        # Skip huge tables (> 100 rows - likely data dumps)
        if rows > 100:
            continue

        # Extract caption
        caption = extract_table_caption(table)

        # Extract section
        section = extract_section_from_html(soup, table)

        # Get HTML
        table_html = str(table)

        # Skip if table is too large to process
        if len(table_html) > 50000:
            continue

        # Convert to markdown (if reasonable size)
        markdown = ""
        if len(table_html) < 20000:
            try:
                markdown = html_table_to_markdown(table)
            except Exception:
                markdown = "[Markdown conversion failed]"
        else:
            markdown = "[Table too large for markdown conversion]"

        # Classify
        table_type = classify_table_type(caption, table_html, section)

        # Create ID (include accession number to ensure uniqueness across filings)
        # Filename format: 10-K_2024-05-23_0001773383-24-000076.html
        accession = file_path.stem.split("_")[-1] if len(file_path.stem.split("_")) > 2 else "unknown"
        table_id = f"{filing_type}_{filing_date}_{accession}_table{idx}"

        results.append(
            {
                "id": table_id,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "section": section,
                "caption": caption[:300],  # Truncate if too long
                "table_type": table_type,
                "rows": rows,
                "cols": cols,
                "html": table_html,
                "markdown": markdown,
                "source_file": str(file_path),
            }
        )

    return results


def select_diverse_tables(all_tables: list[dict], target_count: int = 50) -> list[dict]:
    """Select diverse tables based on filing type, size, and type."""

    # Group by filing type
    by_filing_type = defaultdict(list)
    for table in all_tables:
        by_filing_type[table["filing_type"]].append(table)

    # Target distribution
    targets = {
        "10-K": 12,
        "10-Q": 12,
        "8-K": 8,
        "DEF_14A": 18,  # Proxy statements (combining DEF and DEF_14A keys)
    }

    selected = []

    for filing_type, target in targets.items():
        type_tables = by_filing_type.get(filing_type, [])

        # Combine DEF and DEF_14A (they're the same filing type, just parsed differently)
        if filing_type == "DEF_14A":
            type_tables.extend(by_filing_type.get("DEF", []))

        if not type_tables:
            print(f"  No tables found for {filing_type}")
            continue

        print(f"  Selecting from {len(type_tables)} {filing_type} tables")

        # Filter out tables with poor captions (too short or generic)
        type_tables = [
            t
            for t in type_tables
            if len(t["caption"].strip()) > 10
            and t["caption"] != "Untitled table"
            and not t["caption"].startswith("☒")
            and not t["caption"].startswith("☐")
        ]

        if not type_tables:
            print(f"    No tables with good captions for {filing_type}")
            continue

        # Sort by variety: mix of sizes and types
        # Categorize by size
        small = [t for t in type_tables if t["rows"] <= 5]
        medium = [t for t in type_tables if 6 <= t["rows"] <= 20]
        large = [t for t in type_tables if t["rows"] > 20]

        # Try to get diverse selection
        size_buckets = [small, medium, large]
        tables_to_select = []

        # Round-robin through buckets
        bucket_idx = 0
        attempts = 0
        max_attempts = sum(len(b) for b in size_buckets) * 2

        while len(tables_to_select) < target and attempts < max_attempts:
            bucket = size_buckets[bucket_idx % len(size_buckets)]
            if bucket:
                tables_to_select.append(bucket.pop(0))
            bucket_idx += 1
            attempts += 1

        # If we need more, just take whatever is left
        if len(tables_to_select) < target:
            remaining = [t for bucket in size_buckets for t in bucket]
            tables_to_select.extend(remaining[: target - len(tables_to_select)])

        selected.extend(tables_to_select[:target])
        print(f"    Selected {min(len(tables_to_select), target)} tables")

    return selected[:target_count]


def main():
    data_dir = Path("/home/jlipworth/nextcloud-sync/Documents/Stock Pitches/Long/DT/data/DT")
    output_file = Path(
        "/home/jlipworth/nextcloud-sync/Documents/Stock Pitches/Long/DT/tests/fixtures/table_corpus.json"
    )

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect all tables
    all_tables = []

    # Process 10-K files
    print("Processing 10-K filings...")
    for file in sorted(data_dir.glob("10-K_*.html")):
        print(f"  {file.name}")
        tables = extract_tables_from_file(file)
        all_tables.extend(tables)
        print(f"    Found {len(tables)} tables")

    # Process 10-Q files
    print("\nProcessing 10-Q filings...")
    for file in sorted(data_dir.glob("10-Q_*.html")):
        print(f"  {file.name}")
        tables = extract_tables_from_file(file)
        all_tables.extend(tables)
        print(f"    Found {len(tables)} tables")

    # Process 8-K files (sample - there are many)
    print("\nProcessing 8-K filings (sample)...")
    eightk_files = sorted(data_dir.glob("8-K_*.html"))[:10]  # Just first 10
    for file in eightk_files:
        print(f"  {file.name}")
        tables = extract_tables_from_file(file)
        all_tables.extend(tables)
        print(f"    Found {len(tables)} tables")

    # Process DEF 14A files
    print("\nProcessing DEF 14A proxy statements...")
    for file in sorted(data_dir.glob("DEF_14A_*.html")):
        print(f"  {file.name}")
        tables = extract_tables_from_file(file)
        all_tables.extend(tables)
        print(f"    Found {len(tables)} tables")

    print(f"\nTotal tables extracted: {len(all_tables)}")

    # Select diverse subset
    print("\nSelecting diverse subset of 50 tables...")
    selected_tables = select_diverse_tables(all_tables, target_count=50)

    print(f"Selected {len(selected_tables)} tables")

    # Print summary
    by_type = defaultdict(int)
    by_filing = defaultdict(int)
    size_dist = {"small": 0, "medium": 0, "large": 0}

    for table in selected_tables:
        by_type[table["table_type"]] += 1
        by_filing[table["filing_type"]] += 1

        if table["rows"] <= 5:
            size_dist["small"] += 1
        elif table["rows"] <= 20:
            size_dist["medium"] += 1
        else:
            size_dist["large"] += 1

    print("\nDistribution by filing type:")
    for filing_type, count in sorted(by_filing.items()):
        print(f"  {filing_type}: {count}")

    print("\nDistribution by table type:")
    for table_type, count in sorted(by_type.items()):
        print(f"  {table_type}: {count}")

    print("\nDistribution by size:")
    for size, count in size_dist.items():
        print(f"  {size}: {count}")

    # Save to JSON
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(selected_tables, f, indent=2, ensure_ascii=False)

    print("Done!")

    # Print some example captions
    print("\nExample table captions:")
    for table in selected_tables[:10]:
        print(f"  [{table['filing_type']}] {table['caption'][:80]}")


if __name__ == "__main__":
    main()
