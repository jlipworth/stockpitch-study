"""Stock Pitch CLI - Thin wrapper for backward compatibility.

The CLI has been reorganized into submodules under src/cli/.
This file maintains backward compatibility for direct imports.

New structure:
    src/cli/__init__.py   - App setup, shared utilities
    src/cli/fetch.py      - fetch command
    src/cli/index.py      - index command + helpers
    src/cli/search.py     - search, ask commands
    src/cli/summarize.py  - summarize command + helpers
    src/cli/batch.py      - batch-* commands
    src/cli/misc.py       - notes, process, inventory, wrap commands
"""

# Re-export everything from the cli package for backward compatibility
from src.cli import (
    ANALYST_DIR,
    CONFERENCES_DIR,
    DATA_DIR,
    INDEX_DIR,
    MISC_DIR,
    NOTES_DIR,
    OUTPUT_DIR,
    PRESENTATIONS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SOURCE_DIRS,
    TRANSCRIPTS_DIR,
    SourceType,
    app,
    console,
    logger,
    parse_date,
)

__all__ = [
    "app",
    "console",
    "logger",
    "SourceType",
    "parse_date",
    "PROJECT_ROOT",
    "DATA_DIR",
    "PROCESSED_DIR",
    "INDEX_DIR",
    "OUTPUT_DIR",
    "NOTES_DIR",
    "TRANSCRIPTS_DIR",
    "ANALYST_DIR",
    "PRESENTATIONS_DIR",
    "CONFERENCES_DIR",
    "MISC_DIR",
    "SOURCE_DIRS",
]

if __name__ == "__main__":
    app()
