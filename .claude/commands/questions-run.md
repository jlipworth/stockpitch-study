______________________________________________________________________

## description: Run batch research on question files with agent dispatch argument-hint: [options] allowed-tools: Task, Read, Write, Bash, Glob

# Questions Batch Runner

Run research on pending questions by dispatching Task agents.

## Arguments

Parse from `$ARGUMENTS`:

- `--section X` - Only process questions from section file X (e.g., `02_mgmt_company_board.md`)
- `--test N` - Only process N questions (for testing)
- `--retry-errors` - Re-run questions with ERROR status
- `--deeper` - Re-run questions with NEEDS_DEEPER status

**NOTE**: Concurrency is DISABLED. Process questions SEQUENTIALLY to avoid resource contention.

## Step 1: Load Questions

Read question files from `questions/{TICKER}/`:

```bash
ls questions/*/[0-9]*.md 2>/dev/null || echo "No question files found"
```

For each .md file matching section filter (or all if no filter):

1. Read the file
1. Parse questions with status = `pending` (or `error`/`needs_deeper` based on flags)
1. Collect (filename, question_id, question_text, depth, sources) tuples

## Step 2: Read Company Context

Read CLAUDE.md's "Research Context" section for:

- Ticker
- Fiscal calendar
- Key metrics

## Step 3: Dispatch Agents

For each question to process (respecting --test limit):

1. Map depth to search limits:

   - `fast` → max 2 searches, haiku model
   - `medium` → max 4 searches, default model
   - `deep` → max 8 searches, default model

1. Map sources to doc-type filter:

   ```
   sec → "10-K,10-Q,8-K,DEF 14A"
   transcripts → "transcript"
   analyst → "analyst_report"
   presentations → "presentation"
   conferences → "conference"
   ```

1. Run the search command directly (NO background agents):

   ```bash
   uv run pitch search {TICKER} "query" --doc-type {filter} -k 5
   ```

1. Process results and update file immediately before moving to next question

**SEQUENTIAL ONLY**: Process one question at a time. Complete each question fully (search → answer → update file) before starting the next. This avoids resource contention from concurrent searches.

## Step 4: Update File After Each Question

After completing each question's research:

- Set status = `answered` (or `error` if failed)
- Write answer to `### Answer` section
- Write sources to `### Sources` section

Use file locking when writing:

```python
# The question file parser handles this
from src.questions.parser import parse_question_file
from src.questions.manifest import QuestionManifest
```

## Research Protocol (for Task Agent)

Include this in each agent prompt:

______________________________________________________________________

**Company**: {TICKER}
**Fiscal Calendar**: {from CLAUDE.md}

**Question**: {question_text}
**Sources Filter**: {mapped doc-types or "all" if empty}
**Depth**: {fast|medium|deep}

## Search Command

```bash
# With source filter
uv run pitch search {TICKER} "query" --doc-type {filter} -k 5

# Without filter (if sources is empty/all)
uv run pitch search {TICKER} "query" -k 5
```

## Limits

| Depth  | Max Searches | File Reads        |
| ------ | ------------ | ----------------- |
| fast   | 4            | 0                 |
| medium | 6            | 2 (50 lines max)  |
| deep   | 10           | 4 (50 lines each) |

## Rules

1. Use search results directly as context
1. Only read source files (data/, transcripts/, presentations/, conferences/)
1. NEVER read processed/ folder (AI summaries)
1. NEVER read full files - always use offset/limit
1. Stop when you have the answer
1. Cite sources with quotes

## Output Format

```
**Answer**: [Direct answer with facts]

**Sources**:
- [doc_type] [specific date] [section]: "relevant quote"

**Confidence**: high/medium/low
```

**Citation Examples (BE SPECIFIC):**

- `10-K FY2025 (filed 2025-05-22) Item 2 Properties: "quote"`
- `Q2 FY2026 Earnings Call, Nov 05, 2025, CFO remarks: "quote"`
- `DEF 14A 2025-07-08 Executive Compensation: "quote"`
- `Goldman Sachs Conference, Dec 02, 2025: "quote"`

**BAD citations (too vague):**

- `Transcript Q&A: "quote"` ← Missing which transcript!
- `10-K: "quote"` ← Missing which year!
- `Conference: "quote"` ← Missing which conference!

______________________________________________________________________

## Step 5: Update Manifest

After all questions processed, update `.manifest.json`:

```bash
uv run python -c "
from pathlib import Path
from src.questions.manifest import QuestionManifest
from src.questions.parser import parse_question_file

questions_dir = Path('questions/{TICKER}')
manifest = QuestionManifest(questions_dir / '.manifest.json')

for md_file in questions_dir.glob('[0-9]*.md'):
    qf = parse_question_file(md_file)
    manifest.update_stats_from_file(qf)

manifest.save()
print(manifest.stats_table())
"
```

## Progress Reporting

After each batch of agents completes, report:

- Questions answered: N
- Questions with errors: N
- Remaining: N

______________________________________________________________________

**Running**: $ARGUMENTS
