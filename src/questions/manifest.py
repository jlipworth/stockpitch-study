"""Question manifest for tracking progress and preventing duplicates."""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.file_locks import file_lock

from .parser import QuestionStatus

logger = logging.getLogger(__name__)


@dataclass
class QuestionStats:
    """Statistics for a question file."""

    filename: str
    question_count: int = 0
    pending: int = 0
    in_progress: int = 0
    answered: int = 0
    errors: int = 0
    needs_deeper: int = 0
    incomplete: int = 0


@dataclass
class DuplicateWarning:
    """Warning about potential duplicate questions."""

    q1_id: str
    q1_file: str
    q2_id: str
    q2_file: str
    similarity: float


class QuestionManifest:
    """Tracks question file progress and detects duplicates."""

    def __init__(self, manifest_path: Path):
        """Initialize manifest.

        Args:
            manifest_path: Path to .manifest.json file
        """
        self.manifest_path = manifest_path
        self.ticker = ""
        self.created = ""
        self.last_updated = ""
        self.sections: dict[str, QuestionStats] = {}
        self.question_hashes: dict[str, tuple[str, str]] = {}  # hash -> (filename, question_id)
        self.duplicates_warned: list[DuplicateWarning] = []

        if manifest_path.exists():
            self.load()
        else:
            # Initialize new manifest
            self.created = datetime.now().isoformat()
            self.last_updated = self.created

    def load(self) -> None:
        """Load manifest from disk."""
        data = json.loads(self.manifest_path.read_text())

        self.ticker = data.get("ticker", "")
        self.created = data.get("created", "")
        self.last_updated = data.get("last_updated", "")

        # Load sections
        self.sections = {}
        for filename, stats_dict in data.get("sections", {}).items():
            self.sections[filename] = QuestionStats(
                filename=filename,
                question_count=stats_dict.get("question_count", 0),
                pending=stats_dict.get("pending", 0),
                in_progress=stats_dict.get("in_progress", 0),
                answered=stats_dict.get("answered", 0),
                errors=stats_dict.get("errors", 0),
                needs_deeper=stats_dict.get("needs_deeper", 0),
                incomplete=stats_dict.get("incomplete", 0),
            )

        # Load question hashes (handle both dict and empty list formats)
        self.question_hashes = {}
        raw_hashes = data.get("question_hashes", {})
        if isinstance(raw_hashes, dict):
            for hash_val, (filename, question_id) in raw_hashes.items():
                self.question_hashes[hash_val] = (filename, question_id)
        # If it's a list (legacy format), skip - will rebuild on save

        # Load duplicate warnings
        self.duplicates_warned = []
        for dup_dict in data.get("duplicates_warned", []):
            self.duplicates_warned.append(
                DuplicateWarning(
                    q1_id=dup_dict["q1_id"],
                    q1_file=dup_dict["q1_file"],
                    q2_id=dup_dict["q2_id"],
                    q2_file=dup_dict["q2_file"],
                    similarity=dup_dict["similarity"],
                )
            )

    def save(self) -> None:
        """Save manifest to disk with locking."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "ticker": self.ticker,
            "created": self.created,
            "last_updated": datetime.now().isoformat(),
            "sections": {
                filename: {
                    "question_count": stats.question_count,
                    "pending": stats.pending,
                    "in_progress": stats.in_progress,
                    "answered": stats.answered,
                    "errors": stats.errors,
                    "needs_deeper": stats.needs_deeper,
                    "incomplete": stats.incomplete,
                }
                for filename, stats in self.sections.items()
            },
            "question_hashes": {
                hash_val: [filename, qid] for hash_val, (filename, qid) in self.question_hashes.items()
            },
            "duplicates_warned": [
                {
                    "q1_id": dup.q1_id,
                    "q1_file": dup.q1_file,
                    "q2_id": dup.q2_id,
                    "q2_file": dup.q2_file,
                    "similarity": dup.similarity,
                }
                for dup in self.duplicates_warned
            ],
        }

        with file_lock(self.manifest_path):
            self.manifest_path.write_text(json.dumps(data, indent=2))

        self.last_updated = data["last_updated"]

    def update_stats_from_file(self, question_file) -> QuestionStats:
        """Update stats for a question file.

        Args:
            question_file: QuestionFile instance

        Returns:
            Updated stats
        """
        # Count by status
        status_counts = defaultdict(int)
        for question in question_file.questions:
            status_counts[question.status] += 1

        stats = QuestionStats(
            filename=question_file.filename,
            question_count=len(question_file.questions),
            pending=status_counts[QuestionStatus.PENDING],
            in_progress=status_counts[QuestionStatus.IN_PROGRESS],
            answered=status_counts[QuestionStatus.ANSWERED],
            errors=status_counts[QuestionStatus.ERROR],
            needs_deeper=status_counts[QuestionStatus.NEEDS_DEEPER],
            incomplete=status_counts[QuestionStatus.INCOMPLETE],
        )

        self.sections[question_file.filename] = stats
        return stats

    def get_total_stats(self) -> QuestionStats:
        """Get aggregate stats across all sections.

        Returns:
            Total stats
        """
        total = QuestionStats(filename="TOTAL")

        for stats in self.sections.values():
            total.question_count += stats.question_count
            total.pending += stats.pending
            total.in_progress += stats.in_progress
            total.answered += stats.answered
            total.errors += stats.errors
            total.needs_deeper += stats.needs_deeper
            total.incomplete += stats.incomplete

        return total

    def register_question(self, filename: str, question_id: str, question_text: str) -> str | None:
        """Register a question and detect duplicates.

        Args:
            filename: Question file name
            question_id: Question ID (e.g., "Q1.1")
            question_text: Question text

        Returns:
            Hash of the question if new, None if already registered
        """
        # Normalize question text for hashing
        normalized = question_text.lower().strip()
        # Remove punctuation
        normalized = "".join(c for c in normalized if c.isalnum() or c.isspace())
        # Collapse whitespace
        normalized = " ".join(normalized.split())

        # Hash the question
        question_hash = hashlib.md5(normalized.encode()).hexdigest()

        # Check if already registered
        if question_hash in self.question_hashes:
            existing_file, existing_id = self.question_hashes[question_hash]
            if existing_file != filename or existing_id != question_id:
                # Potential duplicate in different location
                logger.warning(f"Potential duplicate: {filename}:{question_id} matches {existing_file}:{existing_id}")
            return None

        # Register new question
        self.question_hashes[question_hash] = (filename, question_id)
        return question_hash

    def format_stats_table(self) -> str:
        """Format stats as a readable table.

        Returns:
            Formatted table string
        """
        # Header
        lines = [
            "Section                  | Pending | In Progress | Answered | Errors | Needs Deeper | Incomplete",
            "-------------------------|---------|-------------|----------|--------|--------------|----------",
        ]

        # Sections
        for filename, stats in sorted(self.sections.items()):
            line = (
                f"{filename:24s} | {stats.pending:7d} | {stats.in_progress:11d} | "
                f"{stats.answered:8d} | {stats.errors:6d} | {stats.needs_deeper:12d} | {stats.incomplete:10d}"
            )
            lines.append(line)

        # Total
        total = self.get_total_stats()
        lines.append("-------------------------|---------|-------------|----------|--------|--------------|----------")
        line = (
            f"{'TOTAL':24s} | {total.pending:7d} | {total.in_progress:11d} | "
            f"{total.answered:8d} | {total.errors:6d} | {total.needs_deeper:12d} | {total.incomplete:10d}"
        )
        lines.append(line)

        return "\n".join(lines)
