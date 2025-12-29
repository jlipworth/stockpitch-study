# Stock Pitch Case Template

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Toolkit for rapid company analysis and stock pitch recommendations. Combines SEC filing retrieval, document processing, RAG-based search, and AI-assisted Q&A.

## Getting Started

### For New Users (Fork Workflow)

1. **Clone the template** to create a company-specific repo:

   ```bash
   git clone "/path/to/Case Template" /path/to/AAPL-Case
   cd /path/to/AAPL-Case
   ```

1. **Install dependencies**:

   ```bash
   uv sync
   ```

1. **Run interactive setup** (recommended):

   ```bash
   python scripts/setup_company.py
   ```

   The setup script will prompt for:

   - Ticker symbol and company name
   - Position (Long/Short)
   - SEC user agent email
   - Fiscal year end
   - Years of filings to fetch
   - Key thesis points

   It automatically creates `.env`, `COMPANY.md`, folder structure, and fetches SEC filings.

1. **Add source materials** to appropriate folders:

   - `transcripts/{TICKER}/` - Earnings call transcripts (PDF)
   - `analyst/{TICKER}/` - Bank analyst reports (PDF)
   - `conferences/{TICKER}/` - Conference presentations (PDF)
   - `presentations/{TICKER}/` - Investor day materials (PDF)
   - `imports/` - Drop ZIP files here for Claude Code to sort

1. **Build index and start research**:

   ```bash
   uv run pitch index {TICKER}
   uv run pitch inventory
   uv run pitch search {TICKER} "revenue growth"
   ```

### For Contributors

See [Contributing](#contributing) for PR guidelines. Company-specific data is gitignored - only code changes will be included in PRs.

## Features

- **SEC Filing Fetcher** - Download 10-K, 10-Q, 8-K, DEF 14A, and Form 4 filings
- **Form 4 Parser** - Structured parsing of insider trading XML (buys, sells, grants, exercises)
- **Document Parser** - Section-aware chunking with fiscal period extraction (Q1-Q4/FY)
- **Vector Search** - Hybrid search (semantic + BM25) powered by LanceDB
- **RAG Q&A** - Ask questions about filings with Claude-powered answers
- **Document Summarizer** - Hierarchical summarization with parallel API calls and section-specific guidance
- **Markdown Formatter** - Wrap long lines for readability on tablets/narrow displays
- **Materials Inventory** - Quick reference log of all project materials
- **Handwriting Converter** - Convert handwritten PDF notes to markdown with Claude vision

## Setup

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Run interactive setup (creates .env, COMPANY.md, fetches filings)
uv run python scripts/setup_company.py
```

<details>
<summary><strong>Manual Setup (without script)</strong></summary>

```bash
cp .env.template .env
# Edit .env to add:
#   SEC_USER_AGENT="Your Name your@email.com"
#   ANTHROPIC_API_KEY="your-api-key"
```

</details>

### Troubleshooting

<details>
<summary><strong>CUDA/GPU Issues</strong></summary>

```bash
# Check if PyTorch sees your GPU
uv run python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

</details>

<details>
<summary><strong>Environment Issues</strong></summary>

```bash
# Delete .venv and recreate from scratch
rm -rf .venv
uv sync

# If Python version is wrong, pin it
uv python pin 3.12
uv sync
```

</details>

<details>
<summary><strong>BGE-M3 Model Download</strong></summary>

The embedding model (~2.2GB) downloads automatically on first use. If you encounter timeouts:

```bash
# Pre-download the model
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

</details>

<details>
<summary><strong>SEC API Rate Limits</strong></summary>

SEC EDGAR requires a user agent. Ensure `.env` contains:

```
SEC_USER_AGENT="Your Name your@email.com"
```

If you hit rate limits, the fetcher automatically retries with backoff.

</details>

## Quick Start

```bash
# One-command pipeline: fetch, index, and summarize
uv run pitch process AAPL

# Or step by step:
uv run pitch fetch AAPL -t "10-K" -y 2     # 1. Fetch SEC filings
uv run pitch index AAPL                     # 2. Build vector index
uv run pitch summarize AAPL --latest        # 3. Generate summaries
uv run pitch search AAPL "revenue growth"   # 4. Search documents
uv run pitch ask AAPL "What are the main risk factors?"  # 5. Ask questions
```

## CLI Commands

### Setup New Company

```bash
# Interactive setup (prompts for all inputs)
python scripts/setup_company.py

# Pre-specify ticker
python scripts/setup_company.py AAPL

# Skip SEC filing fetch
python scripts/setup_company.py --skip-fetch

# Include competitor research instructions
python scripts/setup_company.py --research
```

The setup script:

- Creates `.env` with SEC_USER_AGENT
- Generates `COMPANY.md` with company context
- Creates folder structure (`transcripts/`, `analyst/`, etc.)
- Fetches SEC filings (10-K, 10-Q, 8-K, DEF 14A)
- Prints next steps

### Full Pipeline (Recommended)

```bash
# Run complete pipeline: fetch → index → summarize
uv run pitch process AAPL

# Customize filing types and years
uv run pitch process MSFT -t "10-K" -y 3

# Skip summarization (faster, no API costs)
uv run pitch process NVDA --skip-summarize
```

### Fetch SEC Filings

```bash
# Fetch 10-K and 10-Q for last 3 years
uv run pitch fetch AAPL -t "10-K,10-Q" -y 3

# Fetch all filing types with date range
uv run pitch fetch AAPL --all --start-date 2022-01-01 --end-date 2024-12-31

# Fetch specific filing types
uv run pitch fetch AAPL -t "10-K,8-K,DEF 14A" -y 2

# Incremental fetch - only new filings since last fetch
uv run pitch fetch AAPL --since
```

### Build Vector Index

```bash
# Build index for all sources (default: SEC + transcripts + analyst + presentations + conferences + misc)
uv run pitch index AAPL

# Index only specific source types
uv run pitch index AAPL --source sec           # SEC filings only
uv run pitch index AAPL --source transcripts   # Earnings call PDFs only
uv run pitch index AAPL --source analyst       # Bank research reports only
uv run pitch index AAPL --source presentations # Investor day materials only
uv run pitch index AAPL --source conferences   # Conference transcripts only
uv run pitch index AAPL --source misc          # Other materials only

# Force rebuild entire index
uv run pitch index AAPL --rebuild

# Override embedding batch size (auto-calculated based on GPU memory)
uv run pitch index AAPL --batch-size 64
```

**Source directories:** Place files in the appropriate folder before indexing:

- `data/{ticker}/` - SEC filings (auto-populated by `pitch fetch`)
- `transcripts/{ticker}/` - Earnings call PDFs
- `analyst/{ticker}/` - Bank research PDFs
- `presentations/{ticker}/` - Investor day PDFs
- `conferences/{ticker}/` - Conference transcript PDFs (GS, MS, JPM, etc.)
- `misc/{ticker}/` - Other materials

### Search Documents

```bash
# Hybrid search (default - combines semantic + keyword)
uv run pitch search AAPL "revenue growth"

# Full-text search only
uv run pitch search AAPL "iPhone sales" --mode fts

# Vector similarity search
uv run pitch search AAPL "competitive landscape" --mode vector

# Filter by filing type
uv run pitch search AAPL "risk factors" --doc-type 10-K

# Filter by section
uv run pitch search AAPL "competition" --section "Item 1A"

# Limit results (--limit, --top-k, or -k all work)
uv run pitch search AAPL "revenue" --limit 5

# Rerank results with cross-encoder for better precision (fetches 50, reranks to top-k)
uv run pitch search AAPL "net interest margin" --rerank
```

### Ask Questions (RAG)

```bash
# Ask a question (uses Claude for answer generation)
uv run pitch ask AAPL "What are the main risk factors?"

# Filter to specific filing type
uv run pitch ask AAPL "How did services revenue perform?" --doc-type 10-K

# Hide source citations
uv run pitch ask AAPL "What is the company's strategy?" --no-sources

# Rerank context for higher precision retrieval
uv run pitch ask AAPL "What is the company's NIM trend?" --rerank
```

### Summarize Documents

```bash
# Summarize SEC filings (default)
uv run pitch summarize AAPL --latest

# Summarize specific source types
uv run pitch summarize AAPL --source transcripts      # Earnings calls
uv run pitch summarize AAPL --source analyst          # Bank research
uv run pitch summarize AAPL --source presentations    # Investor day
uv run pitch summarize AAPL --source conferences      # Conference transcripts
uv run pitch summarize AAPL --source all              # Everything

# Filter by document type within SEC filings
uv run pitch summarize AAPL --doc-type 10-K --latest

# Summarize a specific file directly
uv run pitch summarize AAPL --file transcripts/AAPL/q3_call.pdf
```

**Performance:** Summarization runs 5 concurrent API calls by default for faster processing. Very long sections (>80k chars) are automatically split into chunks and combined.

#### Expected Summary Output Sizes

Each section has configured token limits. Approximate output sizes:

| Document Type             | Max Tokens | ~Words    | ~Pages | Notes                                    |
| ------------------------- | ---------- | --------- | ------ | ---------------------------------------- |
| **10-K**                  | 18,500     | 14,000    | 28     | All 16+ sections with deep MD&A analysis |
| **10-Q**                  | 8,700      | 6,500     | 13     | Condensed quarterly update               |
| **DEF 14A (Proxy)**       | 8,300      | 6,200     | 12     | Executive comp is largest section        |
| **8-K**                   | 800-2,000  | 600-1,500 | 1-3    | Varies by items filed (typically 1-3)    |
| **Earnings Transcript**   | 5,400      | 4,000     | 8      | Prepared remarks + Q&A                   |
| **Analyst Report**        | 6,100      | 4,600     | 9      | Thesis, valuation, estimates, risks      |
| **Investor Presentation** | 8,100      | 6,100     | 12     | Strategy, financials, segments, Q&A      |

**High-priority sections** (Business, MD&A, Executive Comp, Investment Thesis) get 2,000-3,000 tokens each for thorough analysis. **Low-priority sections** (Mine Safety, Exhibits, generic risks) get 300-400 tokens for brief coverage.

Token estimates: 1 token ≈ 0.75 words ≈ 4 characters. Pages assume ~500 words/page.

### Batch Summarization (50% Cost Savings)

For large jobs, use batch processing which runs asynchronously with 50% cost savings:

```bash
# Submit SEC filings for batch summarization
uv run pitch batch-submit AAPL -d 10-K
uv run pitch batch-submit AAPL -d 10-Q --date 2025-08-06

# Submit transcripts, presentations, conferences via --file
uv run pitch batch-submit AAPL -d transcript --file transcripts/AAPL/q4_call.pdf
uv run pitch batch-submit AAPL -d presentation --file presentations/AAPL/investor_day.pdf
uv run pitch batch-submit AAPL -d conference --file conferences/AAPL/gs_conference.pdf

# Check status of all batch jobs
uv run pitch batch-status

# Check status of specific job
uv run pitch batch-status msgbatch_xxx

# Poll until complete
uv run pitch batch-status msgbatch_xxx --poll

# Retrieve results when complete
uv run pitch batch-results msgbatch_xxx

# Resubmit truncated sections with higher token limits
uv run pitch batch-resubmit msgbatch_xxx           # 1.5x tokens (default)
uv run pitch batch-resubmit msgbatch_xxx -m 2.0    # 2x tokens
```

**Supported batch document types:** `10-K`, `10-Q`, `8-K`, `DEF 14A`, `transcript`, `presentation`, `conference`, `analyst`

**How it works:**

- **Automatic chunking**: Long sections (>80k chars for high-priority sections like MD&A, >50k for others) are automatically split by paragraphs into multiple batch requests. Results are combined seamlessly during retrieval.

- **Smart token limits**: Output tokens are set to 60% of input tokens (estimated at 4 chars/token), with no artificial caps. This trusts the prompts to keep output concise while allowing sufficient space for detailed analysis.

- **Resubmit workflow**: If a section gets truncated due to complex content, use `batch-resubmit` to reprocess just that section with higher token limits. The new content automatically merges into the existing summary file.

### Convert Handwritten Notes

```bash
# Convert PDF of handwritten notes to markdown
uv run pitch notes my_notes.pdf

# Specify output path
uv run pitch notes my_notes.pdf --output transcribed.md

# Higher quality (slower)
uv run pitch notes my_notes.pdf --dpi 300
```

### Format Markdown (Line Wrapping)

```bash
# Wrap long lines in a markdown file for readability (100 char default)
uv run pitch wrap processed/AAPL/10-K_2025-05-22_summary.md

# Preview changes without writing
uv run pitch wrap summary.md --preview

# Custom line width
uv run pitch wrap summary.md --width 80

# Output to different file
uv run pitch wrap summary.md --output summary_wrapped.md
```

Preserves headers, tables, code blocks, and bullet point indentation. Useful for reading summaries on narrow displays or tablets.

### Materials Inventory

Generate a quick reference of all materials in the project:

```bash
# For template repo (specify ticker)
uv run pitch inventory AAPL

# For forked company repo (auto-detects)
uv run pitch inventory

# Custom output path
uv run pitch inventory -o docs/materials.md
```

Creates a `MATERIALS.md` with tables listing:

- SEC filings (type, date, accession number)
- Transcripts, analyst reports, presentations, misc
- Summary counts by category

### Slash Commands (Claude Code)

Within Claude Code, use `/notes` for interactive handwriting conversion:

```
/notes /path/to/handwritten.pdf
```

## Project Structure

```
├── data/{ticker}/           # Downloaded SEC filings
├── index/{ticker}/          # Vector indexes (LanceDB)
├── processed/{ticker}/      # Generated summaries
├── src/
│   ├── cli.py              # CLI entry point
│   ├── filings/            # SEC fetcher module
│   ├── rag/                # Parser, embeddings, search, query
│   ├── summarizer/         # Document summarization (with section guidance)
│   └── notes/              # Handwriting conversion
├── .claude/commands/       # Slash commands for Claude Code
└── tests/                  # Test suite
```

### Company-Specific Forks

For dedicated analysis, clone to a company-specific repo:

```
{Ticker}-Case/
├── filings/                # SEC filings
├── transcripts/            # Earnings calls
├── analyst/                # Bank research reports
├── presentations/          # Investor day materials
├── conferences/            # Conference transcripts
├── misc/                   # Other materials
└── src/summarizer/         # ← Customize prompts for industry
```

See [CLAUDE.md](CLAUDE.md) for fork workflow and industry customization guidance.

## Documentation

| Document                                                 | Description                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------------ |
| [CLAUDE.md](CLAUDE.md)                                   | Project context, fork workflow, industry customization             |
| [COMPANY.md](COMPANY.md)                                 | Template for company-specific context (fill this in for your fork) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)             | System architecture and design decisions                           |
| [docs/RAG_IMPROVEMENTS.md](docs/RAG_IMPROVEMENTS.md)     | RAG system roadmap and implementation status                       |
| [docs/QUESTIONS_WORKFLOW.md](docs/QUESTIONS_WORKFLOW.md) | Structured research questions workflow                             |

## Contributing

1. **Fork** the template repository
1. **Create a branch** for your changes
1. **Make changes** - company-specific data is gitignored automatically
1. **Run pre-commit**: `uv run pre-commit run --all-files`
1. **Run tests**: `pytest -v`
1. **Submit PR** - only code changes will be included

### What's Gitignored (Safe for PRs)

All company-specific data is automatically excluded:

- `data/`, `index/`, `processed/`, `output/` - SEC filings and generated content
- `transcripts/`, `analyst/`, `presentations/`, `conferences/`, `misc/` - Research materials
- `research/`, `questions/`, `model/` - Analysis notes and Excel models
- `MATERIALS.md`, `*.jsonl` - Generated inventory and logs

## Running Tests

```bash
pytest -v                    # Run all tests
pytest tests/test_store.py   # Run specific test file
```

## Tech Stack

- **Python 3.12** with uv for dependencies
- **LanceDB** for vector storage with hybrid search
- **BGE-M3** for embeddings (1024 dimensions, 8192 token context)
- **Claude Sonnet 4.5** for RAG answers, summarization, and vision
- **PyMuPDF** for PDF processing
- **edgartools** for SEC EDGAR access

## Development Notes

- **Update README incrementally**: As new features are implemented, add usage examples
- **Track open questions**: Use `TODOS.md` for active development tasks
- **Fiscal period context**: Parser extracts Q1-Q4/FY period from iXBRL metadata to prevent quarter confusion in summaries
- **Summary naming**: Outputs use clean names parsed from source filenames:
  - SEC filings: `10-K_2025-05-22_summary.md`
  - Earnings: `earnings_Q1_FY25_2024-08-07.md`
  - Conferences: `conference_Goldman_Sachs_2025-09-10.md`
  - Presentations: `presentation_Investor_Day_2025-12-06.md`
  - Analyst: `analyst_Morgan_Stanley_2025-09-15.md`

______________________________________________________________________

## License

[![GPLv3 License](https://www.gnu.org/graphics/gplv3-127x51.png)](https://www.gnu.org/licenses/gpl-3.0.en.html)

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
