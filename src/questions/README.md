# Question Batch Runner Module

Batch research workflow for processing structured research questions with concurrent execution, file locking, and progress tracking.

> **TODO**: The `QuestionRunner` class and `run_questions()` convenience wrapper
> described in the original design have not been implemented. The runner module
> provides helper functions instead, with actual research execution handled by
> Task agents dispatched from the `/questions-run` slash command. See
> `codebase_improvements.md` for technical debt tracking.

## Overview

This module provides the core functionality for the `/questions` workflow:

1. **Parser** (`parser.py`) - Parse markdown files with YAML frontmatter and question blocks
1. **Manifest** (`manifest.py`) - Track progress, statistics, and detect duplicates
1. **Runner** (`runner.py`) - Execute batch research with concurrency control and file locking

## Architecture

### File Format

Questions are organized in markdown files with this structure:

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

### Components

#### Parser (`parser.py`)

**Classes:**

- `QuestionStatus` - Enum for question lifecycle (pending, in_progress, answered, error, needs_deeper, incomplete)
- `QuestionDepth` - Enum for research depth (fast, medium, deep)
- `Question` - Single research question with metadata
- `QuestionFile` - Parsed file with frontmatter and questions

**Functions:**

- `parse_question_file(path)` - Parse markdown file into structured format
- `Question.to_markdown()` - Convert question back to markdown
- `QuestionFile.to_markdown()` - Convert entire file back to markdown

**Features:**

- YAML frontmatter parsing with validation
- Regex-based question block extraction
- Status and depth enum validation
- Per-question source overrides
- Answer/sources section extraction
- Round-trip markdown conversion

#### Manifest (`manifest.py`)

**Classes:**

- `QuestionStats` - Statistics for a question file
- `DuplicateWarning` - Warning about potential duplicates
- `QuestionManifest` - Progress tracker with duplicate detection

**Functions:**

- `QuestionManifest.load()` - Load manifest from disk
- `QuestionManifest.save()` - Save manifest to disk with file locking
- `QuestionManifest.update_stats_from_file()` - Update stats from question file
- `QuestionManifest.register_question()` - Register question, detect duplicates
- `QuestionManifest.format_stats_table()` - Format stats as ASCII table

**Features:**

- File-locked JSON persistence
- Per-section statistics tracking
- Aggregate totals calculation
- Duplicate detection via normalized hashing
- Formatted stats table output

#### Runner (`runner.py`)

**Classes:**

- `RunnerConfig` - Configuration for batch execution (concurrency, retry modes, section filtering, test mode)

**Functions:**

- `get_doc_type_filter(sources)` - Convert source list to doc-type filter string for `pitch search`
- `load_question_files(questions_dir, section_filter)` - Load all question files from directory
- `collect_questions_to_run(question_files, config)` - Filter questions based on status and config
- `write_question_file(question_file)` - Write question file back to disk with locking
- `update_manifest(questions_dir, ticker)` - Update manifest with current stats from all question files

**Features:**

- Source type to doc-type mapping (SEC, transcripts, analyst, etc.)
- Question file loading with section filtering
- Status-based filtering (skip answered, optionally retry errors)
- Test mode support (run N random questions)
- File locking for safe concurrent writes
- Manifest updates after processing

> **Note**: The `QuestionRunner` class and `run_questions()` wrapper were planned
> but not implemented. Actual research execution is handled by Task agents dispatched
> from the `/questions-run` slash command.

## Usage

### Parsing Question Files

```python
from pathlib import Path
from src.questions import parse_question_file

# Parse a question file
question_file = parse_question_file(Path("questions/{TICKER}/analyst_opinions.md"))

print(f"Section: {question_file.section}")
print(f"Sources: {question_file.sources}")
print(f"Questions: {len(question_file.questions)}")

# Access individual questions
for question in question_file.questions:
    print(f"{question.id}: {question.question}")
    print(f"  Status: {question.status.value}")
    print(f"  Depth: {question.depth.value}")
```

### Tracking Progress with Manifest

```python
from pathlib import Path
from src.questions import QuestionManifest

# Load or create manifest
manifest = QuestionManifest(Path("questions/{TICKER}/.manifest.json"))
manifest.ticker = "{TICKER}"

# Update stats from a question file
stats = manifest.update_stats_from_file(question_file)
print(f"Pending: {stats.pending}, Answered: {stats.answered}")

# Save manifest
manifest.save()

# Print stats table
print(manifest.format_stats_table())
```

### Using Runner Utilities

The runner module provides helper functions for loading and filtering questions.
Actual batch execution is handled by the `/questions-run` slash command which
dispatches Task agents.

```python
from pathlib import Path
from src.questions import (
    RunnerConfig,
    load_question_files,
    collect_questions_to_run,
    write_question_file,
    update_manifest,
    get_doc_type_filter,
)

# Load all question files from a directory
questions_dir = Path("questions/{TICKER}")
question_files = load_question_files(questions_dir)

# Load with section filter (single file only)
question_files = load_question_files(questions_dir, section_filter="analyst_opinions.md")

# Configure what questions to run
config = RunnerConfig(
    concurrency=3,
    retry_errors=True,         # Re-run questions with ERROR status
    retry_needs_deeper=False,  # Skip needs_deeper questions
    section_filter=None,       # All sections
    test_mode=5,               # Only run 5 random questions (for testing)
)

# Collect questions that need processing
questions_to_run = collect_questions_to_run(question_files, config)
print(f"Questions to process: {len(questions_to_run)}")

# Get doc-type filter string for pitch search
sources = ["sec", "transcripts"]
doc_type_filter = get_doc_type_filter(sources)
# Returns: "10-K,10-Q,8-K,DEF 14A,transcript"

# After updating a question, write it back
for question_file, question in questions_to_run:
    question.answer = "..."
    question.status = QuestionStatus.ANSWERED
    write_question_file(question_file)

# Update manifest with current stats
manifest = update_manifest(questions_dir, ticker="{TICKER}")
print(manifest.format_stats_table())
```

> **Note**: The `QuestionRunner` class and `run_questions()` wrapper shown in
> the original design were not implemented. Use the `/questions-run` slash
> command for batch execution instead.

## Configuration

### RunnerConfig Options

| Option               | Type      | Default | Description                               |
| -------------------- | --------- | ------- | ----------------------------------------- |
| `concurrency`        | int       | 3       | Max concurrent searches (GPU rate limit)  |
| `retry_errors`       | bool      | False   | Re-run questions with error status        |
| `retry_needs_deeper` | bool      | False   | Re-run questions with needs_deeper status |
| `section_filter`     | str\|None | None    | Only run questions from this section file |
| `test_mode`          | int\|None | None    | Run only N random questions for testing   |

### Question Depth Levels

| Depth    | Max Searches | File Snippets | Model  |
| -------- | ------------ | ------------- | ------ |
| `fast`   | 2            | 0             | Haiku  |
| `medium` | 4            | 1             | Sonnet |
| `deep`   | 8            | 3             | Sonnet |

### Question Status Lifecycle

```
pending → in_progress → answered
                     ↓
                   error
                     ↓
               (manual review)
                     ↓
            needs_deeper / incomplete
```

## File Locking

The runner uses `src/utils/file_locks.py` for safe concurrent writes:

- **Platform-aware**: `fcntl.flock()` on Unix/WSL, `msvcrt.locking()` on Windows
- **File-level locking**: Entire section file locked during write
- **Lock files**: Creates `.lock` files alongside data files
- **Automatic cleanup**: Lock released in finally block

## Integration with Research Agent

Research execution is handled by the `/questions-run` slash command, which:

1. Loads question files using `load_question_files()`
1. Collects pending questions using `collect_questions_to_run()`
1. Dispatches Task agents to answer questions using `/research`
1. Writes results back using `write_question_file()`
1. Updates the manifest using `update_manifest()`

The runner module provides the utilities, while the slash command orchestrates
the workflow and manages Task agent concurrency.

## Error Handling

The module handles errors at multiple levels:

1. **Parse errors**: Logged as warnings, invalid question files skipped
1. **Invalid status/depth**: Falls back to defaults (pending, medium)
1. **File write errors**: Logged, file remains in previous state
1. **Manifest errors**: Safe JSON persistence with file locking

## Testing

Run tests with:

```bash
pytest tests/test_questions.py -v
```

**Test coverage:**

- Question file parsing
- Markdown round-trip conversion
- Manifest tracking and persistence
- Stats calculation and formatting
- Duplicate detection
- Runner utilities (load, filter, write)
- Enum validation

## Design Decisions

### Why File-Level Locking?

- **Simplicity**: Easier to implement than question-level locking
- **Performance**: Questions answered out of order anyway (concurrency)
- **Atomicity**: Entire file update is atomic (write → release)
- **Low contention**: Only 3 concurrent workers, files are separate

### Why Immediate Write-Back?

- **Crash recovery**: Progress saved immediately, minimal data loss
- **Real-time monitoring**: Can tail/watch files during batch run
- **Simplicity**: No complex batching/buffering logic

### Why Synchronous Functions?

- **Simplicity**: Standard file I/O without async complexity
- **CLI compatibility**: Works directly with Typer CLI
- **Task agent orchestration**: Async handled at slash command level

### Why Markdown Format?

- **Human-readable**: Easy to review and edit manually
- **Version control friendly**: Git diffs show meaningful changes
- **Extensible**: Can add new fields without breaking parser
- **Editor support**: Syntax highlighting, folding, search

## Future Enhancements

### Phase 1: Source Filtering & Recency

- [ ] Implement source filtering (pre-search filter on doc types)
- [ ] Implement recency weighting (post-search reranking boost)
- [ ] Add age decay formulas (SEC 7yr, Analyst 3yr, Transcript 5yr)

### Phase 2: Pre-Run Checks

- [ ] Duplicate detection across sections
- [ ] Syntax validation
- [ ] Source filter validation
- [ ] Ready-to-run confirmation

### Phase 3: CLI Integration

- [ ] `/questions parse` - Convert PDF to structured markdown
- [ ] `/questions check` - Pre-run validation
- [ ] `/questions run` - Batch execution
- [ ] `/questions status` - Progress display

## Directory Structure

```
questions/{ticker}/
├── .manifest.json              # Progress tracking
├── analyst_opinions.md         # Section file
├── mgmt_company_board.md       # Section file
├── product_services.md         # Section file
└── ...
```

## Dependencies

- `pyyaml` - YAML frontmatter parsing
- `src.utils.file_locks` - Cross-platform file locking

## License

Part of the Stock Pitch Case Template project.
