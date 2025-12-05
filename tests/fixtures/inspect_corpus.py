#!/usr/bin/env python3
"""
Utility script to inspect and analyze the table corpus.

Usage:
    python inspect_corpus.py --stats              # Show statistics
    python inspect_corpus.py --show <id>          # Show specific table
    python inspect_corpus.py --list <filing_type> # List tables by filing type
    python inspect_corpus.py --type <table_type>  # List tables by type
    python inspect_corpus.py --sample <n>         # Show n random samples
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_corpus(path: Path = None) -> list[dict[str, Any]]:
    """Load the table corpus."""
    if path is None:
        path = Path(__file__).parent / "table_corpus.json"

    with open(path) as f:
        return json.load(f)


def show_stats(tables: list[dict[str, Any]]):
    """Display corpus statistics."""
    print(f"{'='*60}")
    print("TABLE CORPUS STATISTICS")
    print(f"{'='*60}\n")

    print(f"Total tables: {len(tables)}\n")

    # Filing types
    print("Filing Type Distribution:")
    filing_counts = Counter(t["filing_type"] for t in tables)
    for filing_type, count in sorted(filing_counts.items()):
        print(f"  {filing_type:10s}: {count:3d} ({count/len(tables)*100:5.1f}%)")

    # Table types
    print("\nTable Type Distribution:")
    type_counts = Counter(t["table_type"] for t in tables)
    for table_type, count in sorted(type_counts.items()):
        print(f"  {table_type:20s}: {count:3d} ({count/len(tables)*100:5.1f}%)")

    # Size distribution
    print("\nSize Distribution:")
    small = sum(1 for t in tables if t["rows"] <= 5)
    medium = sum(1 for t in tables if 6 <= t["rows"] <= 20)
    large = sum(1 for t in tables if t["rows"] > 20)
    print(f"  Small (2-5 rows):   {small:3d} ({small/len(tables)*100:5.1f}%)")
    print(f"  Medium (6-20 rows): {medium:3d} ({medium/len(tables)*100:5.1f}%)")
    print(f"  Large (21+ rows):   {large:3d} ({large/len(tables)*100:5.1f}%)")

    # Dimensions
    print("\nDimension Statistics:")
    print(
        f"  Rows: min={min(t['rows'] for t in tables):2d}, "
        f"max={max(t['rows'] for t in tables):2d}, "
        f"avg={sum(t['rows'] for t in tables) / len(tables):5.1f}"
    )
    print(
        f"  Cols: min={min(t['cols'] for t in tables):2d}, "
        f"max={max(t['cols'] for t in tables):2d}, "
        f"avg={sum(t['cols'] for t in tables) / len(tables):5.1f}"
    )

    # Years
    print("\nTemporal Distribution:")
    year_counts = Counter(t["filing_date"][:4] for t in tables)
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count:3d} ({count/len(tables)*100:5.1f}%)")

    print(f"\n{'='*60}")


def show_table(tables: list[dict[str, Any]], table_id: str):
    """Display a specific table."""
    table = next((t for t in tables if t["id"] == table_id), None)

    if not table:
        print(f"Table '{table_id}' not found.")
        return

    print(f"{'='*60}")
    print(f"TABLE: {table['id']}")
    print(f"{'='*60}\n")

    print(f"Filing Type:  {table['filing_type']}")
    print(f"Filing Date:  {table['filing_date']}")
    print(f"Section:      {table['section']}")
    print(f"Caption:      {table['caption']}")
    print(f"Table Type:   {table['table_type']}")
    print(f"Dimensions:   {table['rows']} rows × {table['cols']} columns")
    print(f"Source:       {Path(table['source_file']).name}")

    print(f"\n{'='*60}")
    print("MARKDOWN:")
    print(f"{'='*60}\n")

    if table["markdown"] == "[Table too large for markdown conversion]":
        print("Table too large to convert to markdown.")
    elif table["markdown"] == "[Markdown conversion failed]":
        print("Markdown conversion failed.")
    else:
        # Show first 30 lines of markdown
        lines = table["markdown"].split("\n")
        for line in lines[:30]:
            print(line)
        if len(lines) > 30:
            print(f"\n... ({len(lines) - 30} more lines)")

    print(f"\n{'='*60}")


def list_tables(tables: list[dict[str, Any]], filing_type: str = None, table_type: str = None):
    """List tables filtered by criteria."""
    filtered = tables

    if filing_type:
        filtered = [t for t in filtered if t["filing_type"] == filing_type]

    if table_type:
        filtered = [t for t in filtered if t["table_type"] == table_type]

    print(f"{'='*80}")
    print(f"TABLES (filtered: {len(filtered)} / {len(tables)} total)")
    print(f"{'='*80}\n")

    for t in filtered:
        caption_preview = t["caption"][:60]
        if len(t["caption"]) > 60:
            caption_preview += "..."

        print(
            f"[{t['id']:30s}] {t['rows']:2d}×{t['cols']:2d}  {t['filing_type']:8s}  "
            f"{t['table_type']:15s}  {caption_preview}"
        )

    print(f"\n{'='*80}")


def show_samples(tables: list[dict[str, Any]], n: int = 5):
    """Show random sample tables."""
    import random

    samples = random.sample(tables, min(n, len(tables)))

    for table in samples:
        print(f"\n{'='*60}")
        print(f"SAMPLE: {table['id']}")
        print(f"{'='*60}")
        print(f"Type: {table['table_type']}, Size: {table['rows']}×{table['cols']}")
        print(f"Caption: {table['caption'][:100]}")

        # Show first few lines of markdown
        if table["markdown"] and not table["markdown"].startswith("["):
            lines = table["markdown"].split("\n")[:5]
            print("\nPreview:")
            for line in lines:
                print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(description="Inspect table corpus")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--show", type=str, help="Show specific table by ID")
    parser.add_argument("--list", action="store_true", help="List all tables")
    parser.add_argument("--filing-type", type=str, help="Filter by filing type")
    parser.add_argument("--table-type", type=str, help="Filter by table type")
    parser.add_argument("--sample", type=int, help="Show n random samples")
    parser.add_argument("--corpus", type=Path, help="Path to corpus JSON file")

    args = parser.parse_args()

    tables = load_corpus(args.corpus)

    if args.stats:
        show_stats(tables)
    elif args.show:
        show_table(tables, args.show)
    elif args.list or args.filing_type or args.table_type:
        list_tables(tables, args.filing_type, args.table_type)
    elif args.sample:
        show_samples(tables, args.sample)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
