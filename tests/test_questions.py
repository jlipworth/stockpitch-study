"""Tests for question batch processing module."""

import tempfile
from pathlib import Path

import pytest

from src.questions import (
    Question,
    QuestionDepth,
    QuestionFile,
    QuestionManifest,
    QuestionStats,
    QuestionStatus,
    RunnerConfig,
    parse_question_file,
)


@pytest.fixture
def sample_question_file():
    """Create a sample question file for testing."""
    content = """---
section: Test Section
sources: [analyst, transcripts]
created: 2025-12-08
---

## Q1.1
**Depth**: medium
**Status**: pending
**Question**: What is the revenue growth rate?

### Answer

### Sources

---

## Q1.2
**Depth**: deep
**Sources**: [sec]
**Status**: answered
**Question**: What are the main risk factors?

### Answer
The main risk factors include market competition and regulatory changes.

### Sources
10-K 2024-03-31 Item 1A: "Risk factors include..."

---
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        return Path(f.name)


def test_parse_question_file(sample_question_file):
    """Test parsing a question file."""
    question_file = parse_question_file(sample_question_file)

    assert question_file.section == "Test Section"
    assert question_file.sources == ["analyst", "transcripts"]
    # YAML parses dates as datetime.date objects
    assert str(question_file.created) == "2025-12-08"
    assert len(question_file.questions) == 2

    # Check first question
    q1 = question_file.questions[0]
    assert q1.id == "Q1.1"
    assert q1.depth == QuestionDepth.MEDIUM
    assert q1.status == QuestionStatus.PENDING
    assert q1.sources is None  # No override
    assert "revenue growth" in q1.question.lower()
    assert q1.answer == ""

    # Check second question
    q2 = question_file.questions[1]
    assert q2.id == "Q1.2"
    assert q2.depth == QuestionDepth.DEEP
    assert q2.status == QuestionStatus.ANSWERED
    assert q2.sources == ["sec"]  # Override
    assert "risk factors" in q2.question.lower()
    assert "market competition" in q2.answer


def test_question_to_markdown():
    """Test converting question back to markdown."""
    question = Question(
        id="Q1.1",
        question="What is the revenue?",
        depth=QuestionDepth.MEDIUM,
        status=QuestionStatus.PENDING,
        sources=None,
        answer="",
        answer_sources="",
    )

    markdown = question.to_markdown()

    assert "## Q1.1" in markdown
    assert "**Depth**: medium" in markdown
    assert "**Status**: pending" in markdown
    assert "**Question**: What is the revenue?" in markdown
    assert "### Answer" in markdown
    assert "### Sources" in markdown
    # Should NOT include Sources field since it's None
    assert "**Sources**:" not in markdown


def test_question_file_to_markdown(sample_question_file):
    """Test converting entire file back to markdown."""
    question_file = parse_question_file(sample_question_file)

    # Modify a question
    question_file.questions[0].status = QuestionStatus.ANSWERED
    question_file.questions[0].answer = "Revenue grew 15% YoY"

    markdown = question_file.to_markdown()

    # Check frontmatter
    assert "section: Test Section" in markdown
    assert "sources:" in markdown
    assert "created: 2025-12-08" in markdown

    # Check updated question
    assert "## Q1.1" in markdown
    assert "**Status**: answered" in markdown
    assert "Revenue grew 15% YoY" in markdown


def test_question_manifest():
    """Test manifest tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / ".manifest.json"
        manifest = QuestionManifest(manifest_path)

        manifest.ticker = "TEST"

        # Create a mock question file
        question_file = QuestionFile(
            path=Path("test.md"),
            section="Test",
            sources=["analyst"],
            created="2025-12-08",
            questions=[
                Question(id="Q1.1", question="Question 1", status=QuestionStatus.PENDING),
                Question(id="Q1.2", question="Question 2", status=QuestionStatus.ANSWERED),
                Question(id="Q1.3", question="Question 3", status=QuestionStatus.ERROR),
            ],
        )

        # Update stats
        stats = manifest.update_stats_from_file(question_file)

        assert stats.question_count == 3
        assert stats.pending == 1
        assert stats.answered == 1
        assert stats.errors == 1

        # Save and reload
        manifest.save()
        manifest2 = QuestionManifest(manifest_path)

        assert manifest2.ticker == "TEST"
        assert "test.md" in manifest2.sections
        assert manifest2.sections["test.md"].question_count == 3


def test_manifest_stats_table():
    """Test manifest stats table formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / ".manifest.json"
        manifest = QuestionManifest(manifest_path)

        # Add some stats
        manifest.sections["section1.md"] = QuestionStats(
            filename="section1.md",
            question_count=10,
            pending=3,
            answered=6,
            errors=1,
        )
        manifest.sections["section2.md"] = QuestionStats(
            filename="section2.md",
            question_count=5,
            pending=2,
            answered=3,
        )

        table = manifest.format_stats_table()

        # Check table contents
        assert "Section" in table
        assert "section1.md" in table
        assert "section2.md" in table
        assert "TOTAL" in table

        # Check totals
        total = manifest.get_total_stats()
        assert total.question_count == 15
        assert total.pending == 5
        assert total.answered == 9
        assert total.errors == 1


def test_runner_config():
    """Test runner configuration."""
    config = RunnerConfig()
    assert config.concurrency == 3
    assert config.retry_errors is False
    assert config.section_filter is None

    config2 = RunnerConfig(concurrency=5, retry_errors=True)
    assert config2.concurrency == 5
    assert config2.retry_errors is True


def test_duplicate_detection():
    """Test question duplicate detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / ".manifest.json"
        manifest = QuestionManifest(manifest_path)

        # Register first question
        hash1 = manifest.register_question("file1.md", "Q1.1", "What is the revenue growth?")
        assert hash1 is not None

        # Register same question (exact match)
        hash2 = manifest.register_question("file1.md", "Q1.1", "What is the revenue growth?")
        assert hash2 is None  # Duplicate

        # Register similar question (normalized match)
        hash3 = manifest.register_question("file2.md", "Q2.1", "what is the revenue growth")
        assert hash3 is None  # Duplicate (case/punctuation normalized)

        # Register different question
        hash4 = manifest.register_question("file1.md", "Q1.2", "What are the risk factors?")
        assert hash4 is not None  # New question


def test_question_depth_enum():
    """Test question depth enumeration."""
    assert QuestionDepth.FAST.value == "fast"
    assert QuestionDepth.MEDIUM.value == "medium"
    assert QuestionDepth.DEEP.value == "deep"


def test_question_status_enum():
    """Test question status enumeration."""
    assert QuestionStatus.PENDING.value == "pending"
    assert QuestionStatus.IN_PROGRESS.value == "in_progress"
    assert QuestionStatus.ANSWERED.value == "answered"
    assert QuestionStatus.ERROR.value == "error"
    assert QuestionStatus.NEEDS_DEEPER.value == "needs_deeper"
    assert QuestionStatus.INCOMPLETE.value == "incomplete"
