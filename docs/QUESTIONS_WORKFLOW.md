# Bulk Questions Workflow

A workflow for processing handwritten research questions into structured searches with batch execution.

## Overview

1. Upload handwritten notes (PDF) with research questions
1. Convert to structured markdown with section/question detection
1. Review and clean questions (edit depth, sources, wording)
1. Run batch research with rate limiting and progress tracking
1. Review answers, flag incomplete ones for deeper search
1. Re-run flagged questions as needed

## Directory Structure

```
questions/{ticker}/
├── .manifest.json              # Progress tracking, duplicate detection
├── analyst_opinions.md         # Section file with Q&As
├── mgmt_company_board.md       # Section file
├── product_services.md         # Section file
└── ...
```

## Question File Format

Each section file uses this structure:

```markdown
---
section: Analyst Opinions
sources: [analyst, transcripts]
created: 2025-12-08
---

## Q1.1
**Depth**: medium
**Status**: pending
**Question**: What is the consensus view on {COMPANY}'s competitive positioning vs {COMPETITOR}?

### Answer
<!-- Research results inserted here -->

### Sources
<!-- Citations inserted here -->

---

## Q1.2
**Depth**: deep
**Sources**: [transcripts, conferences]  <!-- overrides section default -->
**Status**: pending
**Question**: How has management's tone on competitive dynamics changed over the past year?

### Answer

### Sources

---
```

### Field Definitions

**Frontmatter (section-level):**

- `section`: Human-readable section name
- `sources`: Default source filter for all questions in section
- `created`: Date file was created

**Per-question fields:**

- `Depth`: `fast` | `medium` | `deep` - controls research agent thoroughness
- `Sources`: Optional override of section default (omit to use section default)
- `Status`: Question lifecycle state
- `Question`: The actual research question

### Status Values

| Status         | Meaning                                     |
| -------------- | ------------------------------------------- |
| `pending`      | Not yet searched                            |
| `in_progress`  | Currently being searched                    |
| `answered`     | Search complete, answer inserted            |
| `error`        | Search failed after retry                   |
| `needs_deeper` | Flagged for re-run with deeper search       |
| `incomplete`   | Needs more info (possibly external sources) |

### Source Categories

| Source          | Contents                       |
| --------------- | ------------------------------ |
| `sec`           | 10-K, 10-Q, 8-K, proxy filings |
| `transcripts`   | Earnings call transcripts      |
| `analyst`       | Bank research reports          |
| `presentations` | Investor day decks             |
| `conferences`   | GS, MS, JPM fireside chats     |

Default: all sources enabled unless specified.

## Recency Weighting

Documents are weighted by age to prioritize recent information:

### SEC Filings

- Linear decay over 7 years
- Floor weight: 0.3 (30% relevance for 7+ year old docs)
- Formula: `weight = max(0.3, 1.0 - (age_years / 7) * 0.7)`

### Analyst Reports

- Linear decay over 3 years
- Floor weight: 0.2 (20% relevance for 3+ year old docs)
- Formula: `weight = max(0.2, 1.0 - (age_years / 3) * 0.8)`

### Transcripts & Conferences

- Linear decay over 5 years
- Floor weight: 0.25
- Formula: `weight = max(0.25, 1.0 - (age_years / 5) * 0.75)`

Weighting is applied as a reranking boost (multiplied with relevance score), not a hard filter.

## Slash Commands

### `/questions parse <pdf_path>`

Converts handwritten PDF to structured question files.

1. Uses Claude vision to transcribe handwriting
1. Detects sections (page breaks + underlined titles)
1. Detects questions (red highlighter marks)
1. Outputs draft .md files in `questions/{ticker}/`
1. Marks unclear text with `[unclear: ???]`

**Output:** Draft section files ready for review.

### `/questions review`

Interactive review of parsed questions.

1. Shows detected sections and question counts
1. Highlights questions with `[unclear: ???]`
1. Lists auto-classified depth levels for review
1. Warns about potential duplicates across sections

### `/questions check`

Pre-run verification before batch execution.

1. Validates question syntax and format
1. Checks for duplicate questions across sections
1. Verifies source filters are valid
1. Shows summary: N questions across M sections
1. Confirms ready to run

### `/questions run [options]`

Executes batch research.

**Options:**

- `--test N`: Run only N random questions (for testing workflow)
- `--section X`: Run only questions from section X
- `--retry-errors`: Re-run questions with `error` status
- `--deeper`: Re-run questions with `needs_deeper` status

**Behavior:**

- Runs 3 concurrent searches (GPU rate limit)
- Uses file locking for safe concurrent writes
- Writes answers immediately on completion
- Tracks progress in `.manifest.json`
- Skips questions with `answered` status (unless `needs_deeper`)

### `/questions status`

Shows progress across all sections.

```
Section                  | Pending | In Progress | Answered | Errors | Needs Deeper
-------------------------|---------|-------------|----------|--------|-------------
analyst_opinions.md      |    5    |      1      |    12    |   0    |      2
mgmt_company_board.md    |    8    |      0      |     7    |   1    |      0
product_services.md      |   15    |      2      |     3    |   0    |      0
-------------------------|---------|-------------|----------|--------|-------------
TOTAL                    |   28    |      3      |    22    |   1    |      2
```

## Workflow Stages

### Stage 1: Parse Handwriting

```bash
/questions parse imports/{ticker}-questions.pdf
```

Claude vision converts PDF, detecting:

- Section breaks (page breaks + underlined titles)
- Question boundaries (red highlighter marks)
- Multi-line questions grouped as single units

Output: Draft .md files in `questions/{TICKER}/`

### Stage 2: Review & Edit (Manual)

1. Open each section file in editor
1. Fix `[unclear: ???]` placeholders
1. Adjust question wording for search effectiveness
1. Set appropriate depth per question
1. Configure section-level source filters
1. Add per-question source overrides where needed
1. Split or merge question groups as appropriate

### Stage 3: Pre-Run Check

```bash
/questions check
```

Validates:

- All questions have valid syntax
- No duplicate questions
- Source filters are valid
- Depth levels are set

### Stage 4: Test Batch

```bash
/questions run --test 5
```

Runs 5 random questions across sections to validate workflow.

Review results, adjust if needed.

### Stage 5: Full Batch

```bash
/questions run
```

Runs all pending questions with progress tracking.

### Stage 6: Review & Flag (Manual)

1. Review answers in section files
1. Change status to `needs_deeper` for questions needing more thorough search
1. Change status to `incomplete` for questions needing external info

### Stage 7: Re-Run Flagged

```bash
/questions run --deeper
```

Re-runs `needs_deeper` questions with increased depth.

## Manifest File Format

`.manifest.json` tracks progress and prevents duplicates:

```json
{
  "ticker": "{TICKER}",
  "created": "2025-12-08T16:00:00Z",
  "last_updated": "2025-12-08T18:30:00Z",
  "sections": {
    "analyst_opinions.md": {
      "question_count": 18,
      "pending": 5,
      "answered": 12,
      "errors": 0,
      "needs_deeper": 1
    }
  },
  "question_hashes": [
    "a1b2c3...",
    "d4e5f6..."
  ],
  "duplicates_warned": [
    {"q1": "Q1.3", "q2": "Q2.7", "similarity": 0.92}
  ]
}
```

## Implementation Notes

### File Locking

- Lock entire section file during writes (not per-question)
- Use `fcntl.flock()` on Linux/WSL
- Release lock immediately after write completes
- Questions may be answered out of order within file

### Research Integration

- Uses same hybrid search logic as `/research` command
- Adds source filtering (pre-search filter on document type)
- Adds recency weighting (post-search reranking boost)
- Logs feedback to `research_feedback.jsonl` for strategy improvement

### Error Handling

- Retry failed searches once
- Mark as `error` status after second failure
- Continue with remaining questions
- Errors logged to manifest for review

### Duplicate Detection

- Hash question text (normalized, lowercase, no punctuation)
- Warn if similarity > 0.85 across sections
- Don't auto-remove; let user decide
