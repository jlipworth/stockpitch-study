"""Question file parser for batch research workflow.

Parses structured markdown files with YAML frontmatter and question blocks.

File format:
```markdown
---
section: Section Name
sources: [analyst, transcripts]
created: 2025-12-08
---

## Q1.1
**Depth**: medium
**Status**: pending
**Question**: What is the question?

### Answer
<!-- Answer inserted here -->

### Sources
<!-- Sources inserted here -->

---
```
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class QuestionStatus(str, Enum):
    """Question lifecycle status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ANSWERED = "answered"
    ERROR = "error"
    NEEDS_DEEPER = "needs_deeper"
    INCOMPLETE = "incomplete"


class QuestionDepth(str, Enum):
    """Research depth levels."""

    FAST = "fast"
    MEDIUM = "medium"
    DEEP = "deep"


@dataclass
class Question:
    """A single research question."""

    id: str  # e.g., "Q1.1"
    question: str
    depth: QuestionDepth = QuestionDepth.MEDIUM
    status: QuestionStatus = QuestionStatus.PENDING
    sources: list[str] | None = None  # Optional override of section default
    answer: str = ""
    answer_sources: str = ""

    def to_markdown(self) -> str:
        """Convert question back to markdown format."""
        lines = [
            f"## {self.id}",
            f"**Depth**: {self.depth.value}",
        ]

        # Only include Sources if it's an override
        if self.sources is not None:
            sources_str = ", ".join(self.sources)
            lines.append(f"**Sources**: [{sources_str}]")

        lines.append(f"**Status**: {self.status.value}")
        lines.append(f"**Question**: {self.question}")
        lines.append("")
        lines.append("### Answer")
        if self.answer:
            lines.append(self.answer)
        lines.append("")
        lines.append("### Sources")
        if self.answer_sources:
            lines.append(self.answer_sources)
        lines.append("")

        return "\n".join(lines)


@dataclass
class QuestionFile:
    """Parsed question file with metadata."""

    path: Path
    section: str
    sources: list[str]  # Default sources for this section
    created: str
    questions: list[Question] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Get filename only."""
        return self.path.name

    def to_markdown(self) -> str:
        """Convert entire file back to markdown."""
        # Frontmatter
        frontmatter = {
            "section": self.section,
            "sources": self.sources,
            "created": self.created,
        }
        lines = [
            "---",
            yaml.dump(frontmatter, default_flow_style=False).strip(),
            "---",
            "",
        ]

        # Questions
        for i, question in enumerate(self.questions):
            lines.append(question.to_markdown())
            # Add separator between questions (but not after last one)
            if i < len(self.questions) - 1:
                lines.append("---")
                lines.append("")

        return "\n".join(lines)


def parse_question_file(path: Path) -> QuestionFile:
    """Parse a question markdown file.

    Args:
        path: Path to question file

    Returns:
        Parsed question file

    Raises:
        ValueError: If file format is invalid
    """
    content = path.read_text()

    # Extract frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"No YAML frontmatter found in {path}")

    frontmatter_text = frontmatter_match.group(1)
    frontmatter = yaml.safe_load(frontmatter_text)

    # Validate required frontmatter fields
    if "section" not in frontmatter:
        raise ValueError(f"Missing 'section' in frontmatter: {path}")
    if "sources" not in frontmatter:
        raise ValueError(f"Missing 'sources' in frontmatter: {path}")

    # Extract body (everything after frontmatter)
    body_start = frontmatter_match.end()
    body = content[body_start:]

    # Parse questions
    questions = _parse_questions(body, path)

    return QuestionFile(
        path=path,
        section=frontmatter["section"],
        sources=frontmatter["sources"],
        created=frontmatter.get("created", ""),
        questions=questions,
    )


def _parse_questions(body: str, path: Path) -> list[Question]:
    """Parse question blocks from file body.

    Args:
        body: File content after frontmatter
        path: Path to file (for error messages)

    Returns:
        List of parsed questions
    """
    questions = []

    # Split by question headers (## Q1.1, ## Q1.2, etc.)
    question_pattern = r"^## (Q\d+\.\d+)$"
    question_blocks = re.split(f"({question_pattern})", body, flags=re.MULTILINE)

    # First element is content before first question (should be empty or whitespace)
    question_blocks = question_blocks[1:]

    # Process pairs: (header_match, question_id, block_content)
    for i in range(0, len(question_blocks), 3):
        if i + 2 >= len(question_blocks):
            break

        question_id = question_blocks[i + 1]
        block_content = question_blocks[i + 2]

        try:
            question = _parse_question_block(question_id, block_content)
            questions.append(question)
        except ValueError as e:
            logger.warning(f"Skipping invalid question {question_id} in {path}: {e}")

    return questions


def _parse_question_block(question_id: str, content: str) -> Question:
    """Parse a single question block.

    Args:
        question_id: Question ID (e.g., "Q1.1")
        content: Block content after question header

    Returns:
        Parsed question

    Raises:
        ValueError: If required fields are missing
    """
    # Extract fields
    depth_match = re.search(r"\*\*Depth\*\*:\s*(\w+)", content, re.IGNORECASE)
    status_match = re.search(r"\*\*Status\*\*:\s*(\w+)", content, re.IGNORECASE)
    sources_match = re.search(r"\*\*Sources\*\*:\s*\[(.*?)\]", content, re.IGNORECASE)
    question_match = re.search(r"\*\*Question\*\*:\s*(.+?)(?=\n###|\n---|\Z)", content, re.IGNORECASE | re.DOTALL)

    # Extract answer and sources sections
    answer_match = re.search(r"### Answer\n(.*?)(?=### Sources|\Z)", content, re.DOTALL)
    sources_section_match = re.search(r"### Sources\n(.*?)(?=\n---|\Z)", content, re.DOTALL)

    # Validate required fields
    if not question_match:
        raise ValueError("Missing 'Question' field")

    question_text = question_match.group(1).strip()

    # Parse depth (default to medium)
    depth = QuestionDepth.MEDIUM
    if depth_match:
        try:
            depth = QuestionDepth(depth_match.group(1).lower())
        except ValueError:
            logger.warning(f"Invalid depth '{depth_match.group(1)}' for {question_id}, using medium")

    # Parse status (default to pending)
    status = QuestionStatus.PENDING
    if status_match:
        try:
            status = QuestionStatus(status_match.group(1).lower())
        except ValueError:
            logger.warning(f"Invalid status '{status_match.group(1)}' for {question_id}, using pending")

    # Parse sources override (if specified)
    sources = None
    if sources_match:
        sources = [s.strip() for s in sources_match.group(1).split(",")]

    # Extract answer and sources
    answer = ""
    if answer_match:
        answer = answer_match.group(1).strip()
        # Remove HTML comments
        answer = re.sub(r"<!--.*?-->", "", answer, flags=re.DOTALL).strip()

    answer_sources = ""
    if sources_section_match:
        answer_sources = sources_section_match.group(1).strip()
        # Remove HTML comments
        answer_sources = re.sub(r"<!--.*?-->", "", answer_sources, flags=re.DOTALL).strip()

    return Question(
        id=question_id,
        question=question_text,
        depth=depth,
        status=status,
        sources=sources,
        answer=answer,
        answer_sources=answer_sources,
    )
