# Stock Pitch Case Template

## Purpose

A toolkit for rapidly analyzing company case studies and producing stock pitch recommendations (long/short) within a 24-hour turnaround. Combines document processing, RAG-based search, and AI-assisted summarization.

> **IMPORTANT: Keep this file template-generic.**
>
> - NO company-specific content (tickers, company names, paths)
> - Use `{TICKER}`, `{COMPANY}`, `/path/to/` placeholders in examples
> - Company-specific context goes in `COMPANY.md`
> - This file must be pushable to the Case Template repo without modification

## Tech Stack

- **Python**: 3.12
- **Vector Store**: LanceDB (local, hybrid search support)
- **Embeddings**: BGE-M3 (1024 dimensions, 8192 token context, runs on RTX 4090)
- **LLM**: Claude Sonnet 4.5 via API (RAG answers, summarization, vision)
- **PDF Processing**: PyMuPDF for page extraction
- **Config**: `.env` file for API keys and secrets
- **Interface**: CLI with Rich progress display

## Development Environment

### Prerequisites

- **uv**: Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **Python 3.12**: Required (uv will install if needed)

### Setup

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running the CLI

```bash
# Use uv run to ensure you're using the current fork's code:
uv run pitch --help

# Or activate the venv and run directly:
source .venv/bin/activate
pitch --help
```

### Embedding Daemon

For faster searches, start the embedding daemon to keep models loaded in GPU memory:

```bash
# Start daemon (foreground)
uv run pitch serve

# Start daemon (background, auto-stops after 15 min idle)
uv run pitch serve --background

# Check status
uv run pitch daemon-status

# Stop daemon
uv run pitch daemon-stop
```

When running, `pitch search` auto-detects the daemon and uses it for embedding/reranking.
Use `--no-daemon` to force local model loading.

**Architecture:**

- Unix socket IPC (`.pitch-daemon.sock` in project root)
- Sequential request queue (prevents GPU OOM)
- Memory-based batch sizing (adapts to GPU)
- 15-min idle auto-shutdown (configurable)

## Core Components

### 1. SEC Filings Processor

- **Source**: SEC EDGAR (free, self-hosted parsing)
- **Filing types**: 10-K, 10-Q, 8-K, DEF 14A (proxy), Form 4 (insider)
- **Format**: HTML preferred for table structure
- **Library**: `edgartools` or `sec-edgar-downloader`
- **Fiscal period**: Extracts Q1-Q4/FY from iXBRL metadata to prevent quarter confusion

### 2. Document Summarizer

- **Approach**: Hierarchical, section-aware chunking
- **LLM**: Claude Sonnet (quality over cost)
- **Output format**: Structured markdown (headers + bullet points)
- **Strategy**: Preserve document structure, concise but thoughtful summaries
- **Anti-hallucination**: Prompts enforce ticker/company name, fiscal period, data grounding

### 3. RAG/Search System

- **Embedding model**: Local (BGE-M3 on RTX 4090, 8192 token context)
- **Vector store**: LanceDB
- **Search type**: Hybrid (semantic + BM25 keyword search)
- **Chunking**: Document-aware, keeps tables intact
- **Incremental**: Hash-based change detection for efficient updates
- **Index scope**: Per-company (isolated indexes)

### 4. Handwritten Notes Converter

- **Input**: High-res PDF scans
- **Output**: Markdown
- **LLM**: Claude Sonnet with vision (neat cursive, mostly text)
- **Fallback**: Try Haiku first for cost, escalate to Sonnet if needed

### 5. Transcript/Analyst Report Processor

- Earnings call transcripts
- Investor day presentations
- Bank analyst reports
- Same summarization pipeline as documents

## Technical Resources

- NVIDIA RTX 4090 for embeddings and local inference
- Claude API for summarization and vision tasks
- Data is narrow; search/retrieval is the hard problem

## Directory Structure

### Template Repository (Multi-Company)

```
Case Template/
├── CLAUDE.md              # Project context for AI assistant
├── ROADMAP.md             # Future improvements and planned features
├── .env                   # API keys (gitignored)
├── src/                   # Python modules
│   ├── filings/           # SEC fetcher, parsers
│   ├── summarizer/        # Document summarization
│   ├── rag/               # Embeddings, vector store, search
│   ├── notes/             # Handwriting conversion
│   └── cli.py             # CLI entry point
├── data/
│   └── {ticker}/          # Per-company SEC filings (auto-fetched)
├── index/
│   └── {ticker}/          # Per-company vector indexes
├── processed/
│   └── {ticker}/          # Per-company summaries
└── output/                # Final pitch materials
```

### Company-Specific Repository (Forked)

For company-specific analysis, fork the template and use this structure:

```
{Ticker}-Case/
├── CLAUDE.md              # Project context (synced from template)
├── COMPANY.md             # Company-specific context (ticker, fiscal calendar, metrics)
├── MATERIALS.md           # Auto-generated inventory of indexed materials
│
├── research/              # ** HUMAN-CURATED RESEARCH NOTES **
│   ├── competitive_landscape.md  # Market analysis, competitor comparisons, market share
│   ├── customers.md              # Customer segmentation, notable wins, churn analysis
│   ├── executives.md             # Leadership bios, track record, compensation
│   ├── website_summary.md        # Company overview from IR website
│   ├── market_sentiment.md       # G2/Gartner reviews, analyst ratings, Reddit sentiment
│   └── competitors/              # Deep-dive profiles on each major competitor
│
├── questions/{ticker}/    # ** STRUCTURED RESEARCH QUESTIONS **
│   ├── 01_products_competitive.md  # Product/competitive positioning questions
│   ├── 02_mgmt_company_board.md    # Management, governance questions
│   └── 03_analysts.md              # Analyst-focused questions
│
├── model/                 # ** EXCEL MODEL INTEGRATION **
│   ├── *.xlsx             # Financial model files
│   ├── MODEL_STRUCTURE.md # Column/row mappings for programmatic access
│   └── MODEL_TODOS.md     # Model-specific tasks and validation status
│
├── data/{ticker}/         # SEC filings (auto-fetched via `pitch fetch`)
├── transcripts/{ticker}/  # Earnings call transcripts (PDF, manual upload)
├── conferences/{ticker}/  # Conference/fireside chat transcripts (PDF)
├── analyst/{ticker}/      # Bank analyst reports (PDF)
├── presentations/{ticker}/ # Investor day presentations (PDF)
├── misc/                  # Other materials that don't fit elsewhere
│
├── index/{ticker}/        # Vector index (auto-generated via `pitch index`)
├── processed/{ticker}/    # AI-generated summaries (via `pitch summarize`)
├── notes/                 # Converted handwritten notes (via `/notes` command)
├── output/                # Final pitch materials (memos, tear sheets)
│
├── docs/                  # ** DOCUMENTATION **
│   ├── GETTING_STARTED.md        # Step-by-step new company guide
│   ├── bulk_questions_workflow.md # Batch research workflow
│   ├── excel_model_workflow.md    # Model integration patterns
│   └── RAG_IMPROVEMENTS.md        # Technical roadmap
│
└── src/                   # Python modules (synced from template)
```

#### Key Folders for Research Context

When starting a new session, these folders contain the most useful context:

| Folder                | Purpose                          | When to Use                                               |
| --------------------- | -------------------------------- | --------------------------------------------------------- |
| `research/`           | Human-curated analysis and notes | Understanding competitive dynamics, customers, management |
| `questions/{ticker}/` | Structured Q&A with answers      | Finding specific facts already researched                 |
| `processed/{ticker}/` | AI summaries of source docs      | Quick overview of filings, transcripts                    |
| `model/`              | Financial model and structure    | Connecting research to valuation                          |

#### Research Standards: Citations Required

**All numerical claims and facts must include citations.** This prevents unverified figures from propagating through research notes.

**For `questions/{ticker}/*.md` files:**

- Each answer MUST have a `### Sources` section listing specific sources
- Each answer MUST have a `**Confidence**` level (High/Medium/Low)
- Format citations as: `[source type] [date]: [specific detail]`
  - Example: `conference SEP 04, 2024: CFO on DPS transition`
  - Example: `10-K FY2025 Item 7: Revenue recognition policy`

**For `research/*.md` files:**

- Numerical claims should include inline citations or footnotes
- Mark unverified claims with `[UNVERIFIED]` or `[NEEDS CITATION]`
- External sources (Gartner, G2, analyst reports) should be explicitly noted

**For `materials/*.md` (pitch documents):**

- All figures should be traceable to `questions/` or `processed/` files
- When in doubt, verify against primary sources before including

**Verification workflow:**

1. Before adding a figure, search the index: `uv run pitch search {TICKER} "query"`
1. If not found, mark as `[UNVERIFIED]` or remove
1. Periodically audit research notes for unverified claims

### Creating a Company Fork

**Automated (recommended):**

```bash
# From the Case Template directory
./scripts/setup_company.sh AAPL                     # Creates ../Long/AAPL
./scripts/setup_company.sh AAPL /custom/path/AAPL   # Custom destination
```

The script handles: cloning, git setup, directory creation, .env copying, and COMPANY.md placeholders.

**Manual setup:**

```bash
# Clone template to company-specific directory
git clone "/path/to/Case Template" /path/to/AAPL-Case
cd /path/to/AAPL-Case

# Create company branch (keeps template as upstream)
git remote rename origin template
git checkout -b aapl-analysis

# Create the folder structure
mkdir -p filings transcripts analyst presentations conferences misc model

# Customize COMPANY.md with company-specific context (see below)
# Customize prompts for the industry (edit SECTION_WEIGHTS in summarizer.py)
# Then commit your customizations
```

### Setting Up COMPANY.md

The `COMPANY.md` file contains all company-specific research context. This keeps `CLAUDE.md` identical across forks for clean merges.

1. **Copy the template**: `COMPANY.md` already exists with placeholders
1. **Fill in company details**:
   - Ticker and company name
   - Fiscal calendar (fiscal year end, quarter mapping)
   - Key metrics for the industry (SaaS, financials, industrials, etc.)
   - Current investment narrative/thesis points
1. **Update after indexing**: Run `pitch inventory` and update the "Available Documents" section
1. **Commit**: This file is company-specific and won't conflict on merges

### Git Workflow for Template ↔ Fork Sync

- **Push to template:** `Template:` prefix commits → cherry-pick to template
- **Fork-only:** `{TICKER}:` prefix commits → stays in fork
- **Pull from template:** `git fetch template && git merge template/master`
- **CRITICAL:** Always separate template-generic and fork-specific changes into different commits

## Industry-Specific Customization

The `SECTION_WEIGHTS` dictionary in `src/summarizer/summarizer.py` contains all prompt guidance. Customize for your industry:

### Tech Companies

Focus on: ARR/MRR, NRR, customer acquisition costs, R&D capitalization, stock-based comp, TAM claims

### Financial Services

Focus on: NIM, credit quality, loan loss reserves, capital ratios, AUM flows, fee income mix

### Industrials/Manufacturing

Focus on: Capacity utilization, backlog, input costs, working capital, CapEx cycles

### Healthcare/Pharma

Focus on: Pipeline stages, FDA timelines, patent cliffs, pricing/reimbursement, clinical trial data

### Retail/Consumer

Focus on: Same-store sales, inventory turns, e-commerce mix, promotional activity, consumer trends

To customize, edit the `guidance` field in relevant sections. Example for a bank:

```python
"Item 7": {
    "weight": "high",
    "max_tokens": 3000,
    "guidance": """MD&A for a BANK - focus on:
    - Net interest margin (NIM) and spread trends
    - Loan growth by category (C&I, CRE, consumer)
    - Credit quality: NPLs, charge-offs, reserve coverage
    - Fee income breakdown and trends
    - Efficiency ratio and expense management
    - Capital ratios vs. regulatory minimums
    - Interest rate sensitivity (asset-sensitive vs liability-sensitive)
    ..."""
}
```

## Claude Code Slash Commands

Custom commands in `.claude/commands/`:

- **`/notes <pdf_path>`** - Convert handwritten PDF notes to markdown

- **`/fact-check [ticker]`** - Verification agent for fact-checking summaries

  - Checks date consistency (filename vs content)
  - Validates fiscal period accuracy (company-specific fiscal year)
  - Cross-references metrics across documents
  - Flags potential hallucinations

- **`/research [question]`** - Super-hybrid RAG research agent

  - Classifies query type (metric lookup, trend, comparative, etc.)
  - Selects optimal search strategy (FTS, vector, hybrid, filtered)
  - Tries multiple approaches if first attempt insufficient
  - Logs feedback to `research_feedback.jsonl` for strategy improvement
  - Designed to evolve into MCP server once patterns are understood

## Research Context

<!-- Company-specific context is in COMPANY.md. This keeps CLAUDE.md mergeable with template. -->

<!-- The /research agent and other tools should read COMPANY.md for ticker, fiscal calendar, metrics, etc. -->

**See:** [COMPANY.md](COMPANY.md) for company-specific research context including:

- Ticker and fiscal calendar
- Key metrics and KPIs
- Investment narrative
- Available documents index

## Key Design Decisions

1. **Per-company isolation**: Each case study gets its own index/data dirs
1. **HTML filings**: Preserve table structure for better chunking
1. **Hybrid search**: Semantic for concepts, BM25 for exact terms (tickers, names)
1. **Local embeddings**: RTX 4090 for speed and cost savings
1. **Claude for quality tasks**: Summarization and handwriting need quality
1. **Incremental processing**: Hash files to avoid reprocessing unchanged docs
1. **Tables for search**: Don't extract separately; chunk them for embeddings
