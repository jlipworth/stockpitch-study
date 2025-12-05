"""Question batch runner utilities.

Provides helper functions for the /questions-run slash command:
- Question file loading and filtering
- Source type to doc-type mapping
- File writing with locking
- Manifest updates

The actual research execution is handled by Task agents dispatched
from the /questions-run slash command.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.utils.file_locks import file_lock

from .manifest import QuestionManifest
from .parser import Question, QuestionFile, QuestionStatus, parse_question_file

logger = logging.getLogger(__name__)


# Map source types to doc-type filter values for pitch search
SOURCE_TO_DOC_TYPES = {
    "sec": ["10-K", "10-Q", "8-K", "DEF 14A"],
    "transcripts": ["transcript"],
    "analyst": ["analyst_report"],
    "presentations": ["presentation"],
    "conferences": ["conference"],
    "misc": ["misc"],
}


def get_doc_type_filter(sources: list[str]) -> str | None:
    """Convert source list to doc-type filter string for pitch search.

    Args:
        sources: List of source types (e.g., ["sec", "transcripts"])

    Returns:
        Comma-separated doc-type string for --doc-type flag, or None if all sources
    """
    if not sources:
        return None

    doc_types = []
    for src in sources:
        if src in SOURCE_TO_DOC_TYPES:
            doc_types.extend(SOURCE_TO_DOC_TYPES[src])
        else:
            logger.warning(f"Unknown source type: {src}")

    if not doc_types:
        return None

    return ",".join(doc_types)


@dataclass
class RunnerConfig:
    """Configuration for question runner."""

    concurrency: int = 3  # Max concurrent agents
    retry_errors: bool = False  # Re-run error status questions
    retry_needs_deeper: bool = False  # Re-run needs_deeper status questions
    section_filter: str | None = None  # Only run questions from this section file
    test_mode: int | None = None  # If set, run only N random questions for testing


def load_question_files(
    questions_dir: Path,
    section_filter: str | None = None,
) -> list[QuestionFile]:
    """Load all question files from directory.

    Args:
        questions_dir: Directory containing question files
        section_filter: Optional filename to filter to single section

    Returns:
        List of parsed question files
    """
    question_files = []

    for md_file in sorted(questions_dir.glob("*.md")):
        # Skip manifest and other metadata files
        if md_file.name.startswith("."):
            continue

        # Apply section filter if specified
        if section_filter and md_file.name != section_filter:
            continue

        try:
            question_file = parse_question_file(md_file)
            question_files.append(question_file)
        except Exception as e:
            logger.error(f"Failed to parse {md_file}: {e}")

    return question_files


def collect_questions_to_run(
    question_files: list[QuestionFile],
    config: RunnerConfig,
) -> list[tuple[QuestionFile, Question]]:
    """Collect questions that need to be run based on config.

    Args:
        question_files: List of question files
        config: Runner configuration

    Returns:
        List of (question_file, question) tuples to process
    """
    questions_to_run = []

    for question_file in question_files:
        for question in question_file.questions:
            # Skip already answered (unless retry mode)
            if question.status == QuestionStatus.ANSWERED:
                continue

            # Skip error status unless retry_errors enabled
            if question.status == QuestionStatus.ERROR and not config.retry_errors:
                continue

            # Skip needs_deeper unless retry_needs_deeper enabled
            if question.status == QuestionStatus.NEEDS_DEEPER and not config.retry_needs_deeper:
                continue

            questions_to_run.append((question_file, question))

    # Apply test mode filtering
    if config.test_mode:
        import random

        questions_to_run = random.sample(questions_to_run, min(config.test_mode, len(questions_to_run)))

    return questions_to_run


def write_question_file(question_file: QuestionFile) -> None:
    """Write question file back to disk with locking.

    Args:
        question_file: Question file to write
    """
    with file_lock(question_file.path):
        markdown = question_file.to_markdown()
        question_file.path.write_text(markdown)


def update_manifest(questions_dir: Path, ticker: str) -> QuestionManifest:
    """Update manifest with current stats from all question files.

    Args:
        questions_dir: Directory containing question files
        ticker: Stock ticker

    Returns:
        Updated manifest
    """
    manifest_path = questions_dir / ".manifest.json"
    manifest = QuestionManifest(manifest_path)

    if not manifest.ticker:
        manifest.ticker = ticker

    for question_file in load_question_files(questions_dir):
        manifest.update_stats_from_file(question_file)

    manifest.save()
    return manifest
