#!/usr/bin/env python3
"""
Validation script for table corpus.

Checks:
- All tables have required fields
- All tables have valid HTML
- Table dimensions match actual HTML structure
- No duplicate IDs
- Filing dates are valid
- Table types are from expected set
"""

import json
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup


def validate_corpus(corpus_path: Path = None):
    """Validate the table corpus."""
    if corpus_path is None:
        corpus_path = Path(__file__).parent / "table_corpus.json"

    print("=" * 60)
    print("TABLE CORPUS VALIDATION")
    print("=" * 60)

    with open(corpus_path) as f:
        tables = json.load(f)

    print(f"\nLoaded {len(tables)} tables\n")

    # Required fields
    required_fields = [
        "id",
        "filing_type",
        "filing_date",
        "section",
        "caption",
        "table_type",
        "rows",
        "cols",
        "html",
        "markdown",
        "source_file",
    ]

    # Valid table types
    valid_types = {
        "financial_statement",
        "compensation",
        "governance",
        "comparison",
        "metrics",
        "schedule",
        "segment_data",
        "other",
    }

    # Valid filing types
    valid_filing_types = {"10-K", "10-Q", "8-K", "DEF", "DEF_14A"}

    errors = []
    warnings = []

    # Check 1: All tables have required fields
    print("Checking required fields...")
    for i, table in enumerate(tables):
        missing = [f for f in required_fields if f not in table]
        if missing:
            errors.append(f"Table {i} ({table.get('id', 'UNKNOWN')}): Missing fields: {missing}")

    if not errors:
        print("  ✓ All tables have required fields")

    # Check 2: No duplicate IDs
    print("Checking for duplicate IDs...")
    ids = [t["id"] for t in tables]
    duplicates = [id for id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate IDs found: {duplicates}")
    else:
        print("  ✓ All IDs are unique")

    # Check 3: Valid table types
    print("Checking table types...")
    invalid_types = [t["id"] for t in tables if t["table_type"] not in valid_types]
    if invalid_types:
        errors.append(f"Invalid table types: {invalid_types}")
    else:
        print("  ✓ All table types are valid")

    # Check 4: Valid filing types
    print("Checking filing types...")
    invalid_filings = [t["id"] for t in tables if t["filing_type"] not in valid_filing_types]
    if invalid_filings:
        warnings.append(f"Non-standard filing types: {invalid_filings}")
    else:
        print("  ✓ All filing types are standard")

    # Check 5: HTML is valid
    print("Checking HTML validity...")
    invalid_html = []
    for table in tables:
        try:
            soup = BeautifulSoup(table["html"], "lxml")
            table_elem = soup.find("table")
            if not table_elem:
                invalid_html.append(table["id"])
        except Exception as e:
            invalid_html.append(f"{table['id']}: {str(e)}")

    if invalid_html:
        errors.append(f"Invalid HTML in tables: {invalid_html}")
    else:
        print("  ✓ All HTML is parseable")

    # Check 6: Dimensions match HTML
    print("Checking table dimensions...")
    dimension_mismatches = []
    for table in tables:
        try:
            soup = BeautifulSoup(table["html"], "lxml")
            table_elem = soup.find("table")
            if table_elem:
                rows = table_elem.find_all("tr")
                actual_rows = len(rows)

                # Check columns (first row)
                if rows:
                    cells = rows[0].find_all(["td", "th"])
                    actual_cols = sum(int(cell.get("colspan", 1)) for cell in cells)

                    # Allow some tolerance for dimension detection
                    if abs(table["rows"] - actual_rows) > 1:
                        dimension_mismatches.append(f"{table['id']}: rows {table['rows']} vs {actual_rows}")
                    # Columns are trickier due to colspan, so just warn if wildly different
                    if abs(table["cols"] - actual_cols) > 5:
                        warnings.append(f"{table['id']}: cols {table['cols']} vs {actual_cols}")
        except Exception as e:
            errors.append(f"Error checking dimensions for {table['id']}: {e}")

    if dimension_mismatches:
        errors.append(f"Dimension mismatches: {dimension_mismatches}")
    else:
        print("  ✓ Table dimensions match HTML")

    # Check 7: Source files exist
    print("Checking source files...")
    missing_sources = []
    for table in tables:
        source_path = Path(table["source_file"])
        if not source_path.exists():
            missing_sources.append(table["id"])

    if missing_sources:
        warnings.append(f"Missing source files for: {missing_sources}")
    else:
        print("  ✓ All source files exist")

    # Check 8: Markdown conversion status
    print("Checking markdown conversions...")
    no_markdown = sum(1 for t in tables if t["markdown"].startswith("["))
    successful_markdown = len(tables) - no_markdown
    print(f"  ✓ {successful_markdown}/{len(tables)} tables converted to markdown")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if errors:
        print(f"\n❌ {len(errors)} ERRORS:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ No errors found!")

    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✓ No warnings!")

    # Statistics
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"\nTotal tables: {len(tables)}")
    print(f"Filing types: {dict(Counter(t['filing_type'] for t in tables))}")
    print(f"Table types: {dict(Counter(t['table_type'] for t in tables))}")
    print(
        f"Size range: {min(t['rows'] for t in tables)}-{max(t['rows'] for t in tables)} rows, "
        f"{min(t['cols'] for t in tables)}-{max(t['cols'] for t in tables)} cols"
    )
    print(
        f"Markdown success rate: {successful_markdown}/{len(tables)} ({successful_markdown / len(tables) * 100:.1f}%)"
    )

    print("\n" + "=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    import sys

    success = validate_corpus()
    sys.exit(0 if success else 1)
