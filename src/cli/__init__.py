"""Stock Pitch CLI - Rapid company analysis toolkit.

This package provides the CLI interface for the stock pitch toolkit.
Commands are organized into submodules for maintainability.
"""

import logging
import os
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.logging_config import setup_logging

# Load environment variables
load_dotenv()


class SourceType(str, Enum):
    """Document source types for indexing."""

    SEC = "sec"
    TRANSCRIPTS = "transcripts"
    ANALYST = "analyst"
    PRESENTATIONS = "presentations"
    CONFERENCES = "conferences"
    MISC = "misc"
    ALL = "all"


def parse_date(value: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise typer.BadParameter(f"Invalid date format: {value}. Use YYYY-MM-DD.")


# Create the main app
app = typer.Typer(
    name="pitch",
    help="Stock pitch analysis toolkit for rapid company research.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False,
    log_file: Annotated[Path | None, typer.Option("--log-file", help="Path to log file")] = None,
) -> None:
    """Stock pitch analysis toolkit for rapid company research."""
    setup_logging(verbose=verbose, log_file=log_file)


# Get logger for CLI module
logger = logging.getLogger(__name__)

# Project paths - use current working directory for portability across forks
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
INDEX_DIR = Path(os.environ.get("PITCH_INDEX_DIR", PROJECT_ROOT / "index")).expanduser()
OUTPUT_DIR = PROJECT_ROOT / "output"
NOTES_DIR = PROJECT_ROOT / "notes"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
ANALYST_DIR = PROJECT_ROOT / "analyst"
PRESENTATIONS_DIR = PROJECT_ROOT / "presentations"
CONFERENCES_DIR = PROJECT_ROOT / "conferences"
MISC_DIR = PROJECT_ROOT / "misc"

# Source directory mapping
SOURCE_DIRS = {
    SourceType.SEC: DATA_DIR,
    SourceType.TRANSCRIPTS: TRANSCRIPTS_DIR,
    SourceType.ANALYST: ANALYST_DIR,
    SourceType.PRESENTATIONS: PRESENTATIONS_DIR,
    SourceType.CONFERENCES: CONFERENCES_DIR,
    SourceType.MISC: MISC_DIR,
}

# Import and register command modules
# These imports must come after app is defined to avoid circular imports
from src.cli import batch, fetch, index, misc, search, summarize  # noqa: E402, F401

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
    # Re-export rich utilities for submodules
    "rich_escape",
    "Progress",
    "SpinnerColumn",
    "TextColumn",
    "BarColumn",
    "MofNCompleteColumn",
    "TimeElapsedColumn",
]
