# RAG Pipeline Improvements

## Status Summary (Dec 21, 2025)

| Phase                           | Status      | Completed Items                                                                 |
| ------------------------------- | ----------- | ------------------------------------------------------------------------------- |
| Phase 1: Context Expansion      | ✅ COMPLETE | Chunk linking, position tracking, context expansion, **parent-child sections**  |
| Phase 2: Table Intelligence     | ✅ COMPLETE | Table dataclass, classification, table-aware chunking, search filters           |
| Phase 3: Query Enhancement      | ✅ COMPLETE | Query classification, **query expansion (120+ acronyms, 50+ synonym clusters)** |
| Phase 4: Advanced Chunking      | PARTIAL     | ✅ Content-type-specific chunk sizes                                            |
| Phase 5: Continuous Improvement | PARTIAL     | ✅ Feedback logging, ✅ Position tracking                                       |
| Phase 6: Advanced Features      | NOT STARTED | Multi-hop, NER metadata                                                         |

**Quick Wins: 4/4 COMPLETE** ✅

- Content-type-specific chunk sizes (`parser.py`)
- Chunk position tracking (`parser.py`)
- Query classification for hybrid weights (`query_classifier.py`)
- Feedback logging (`feedback.py`)

**Phase 1 Implementation** ✅

- Chunk dataclass extended with `prev_chunk_id`, `next_chunk_id`, `position_in_section`, `is_section_start`, `is_section_end`
- `get_chunk_with_context()` method in `search.py` for fetching adjacent chunks
- `expand_results_with_context()` method for expanding search results with surrounding context
- **Parent-child section retrieval**: Chunks have `section_id`, sections stored in `sections.json`
- `get_parent_context()` method returns full section when under size limit, falls back to window mode for large sections
- `expand_results_with_parent()` method for batch section retrieval with deduplication
- All metadata stored in LanceDB and queryable

**Phase 3 Query Expansion** ✅ (Dec 12, 2025)

- `src/rag/query_expansion.py`: QueryExpander class with financial domain knowledge
- 120+ acronym expansions (ARR, NRR, NIM, EBITDA, etc.)
- 50+ synonym clusters (revenue/sales/top line, profit/earnings/net income, etc.)
- Integrated into Searcher with `expand_query` parameter
- Multi-query search with result deduplication

**CLI Integration** ✅ (Dec 13, 2025)

- `pitch ask`: Parent-child retrieval enabled by default in `_build_context()`

  - Uses `expand_results_with_parent()` to get full section context
  - Deduplicates results from same section
  - Falls back to window mode for large sections (>8000 tokens)
  - Labels output as `[full section]` or `[excerpt]`

- `pitch search`: New `--context` option for expanded results

  - `--context none` (default): Chunk text only (500 chars)
  - `--context section`: Full parent section (deduplicates by section_id)
  - `--context window`: Chunk + 2 neighbors
  - `--max-chars N`: Control output length (0 = unlimited)

- Query expansion: Already active by default (`SearchConfig.expand_query=True`)

**Next Recommended Steps:**

1. **MEDIUM**: Phase 4 - Semantic chunking (beyond content-type-specific sizes)
1. **MEDIUM**: Phase 6 - Multi-hop retrieval for complex analytical questions

**Recent Fixes (Dec 11, 2025):**

- GPU memory management for reranking: Added `clear_gpu_cache()` before/after reranking, reduced reranker batch size (16→4), reduced `rerank_top_n` (50→30) to prevent OOM when both embedding model and reranker are loaded simultaneously with long SEC filing texts.

______________________________________________________________________

## Current State

The RAG pipeline is a well-architected system for financial document retrieval and question-answering:

### Architecture Overview

1. **Parser (`src/rag/parser.py`)**: Extracts text from SEC filings with section detection and table-aware chunking

   - Supports 10-K, 10-Q, 8-K, DEF 14A filing types
   - HTML-based section detection with fallback to text-based regex
   - Converts tables to markdown format
   - Chunks sections into 1000-token pieces with 200-token overlap
   - Extracts fiscal metadata from iXBRL tags

1. **Embeddings (`src/rag/embeddings.py`)**: Generates and manages embeddings using BGE-M3

   - 1024-dimension embeddings with 8192 token context window
   - GPU-accelerated with CUDA/MPS support
   - Cross-encoder reranking with BGE Reranker v2 M3
   - Automatic memory management and batch size tuning

1. **Vector Store (`src/rag/store.py`)**: LanceDB-based storage with per-company isolation

   - Hybrid search (vector similarity + BM25 full-text search)
   - Metadata filtering on ticker, filing_type, filing_date, section
   - Incremental indexing with hash-based change detection
   - FTS (full-text search) index for keyword matching

1. **Search (`src/rag/search.py`)**: High-level search interface

   - Three search modes: vector, FTS, hybrid
   - Optional cross-encoder reranking
   - Context window generation for LLM queries
   - Similar document search

1. **Query (`src/rag/query.py`)**: Claude-powered RAG query engine

   - Integrates search with Claude Sonnet 4.5
   - Configurable context window (default 4000 tokens)
   - Source citation tracking

### Strengths

- **Clean separation of concerns**: Each module has a clear responsibility
- **Production-ready features**: Incremental indexing, file locking, error handling
- **Flexible search**: Multiple search modes with filtering and reranking
- **GPU optimization**: Memory management for long indexing runs
- **Domain-specific**: Table handling and fiscal metadata extraction for SEC filings

## Identified Limitations

### 1. Chunk Boundary Context Loss (HIGH PRIORITY) ✅ SOLVED

**Problem**: Fixed-size chunking with paragraph boundaries can split important context across chunks.

**Examples**:

- Table header in one chunk, data rows in another
- Multi-paragraph explanations split across chunks
- Section introduction separated from key details
- Cross-references that span chunks

**Impact**:

- Search may return incomplete information
- LLM receives fragmented context
- User questions requiring connected information may fail

**Current Mitigation**:

- 200-token overlap helps but is insufficient for long tables or multi-paragraph discussions
- Tables converted to markdown may still be split if they exceed chunk size

### 2. No Surrounding Context Retrieval (HIGH PRIORITY) ✅ SOLVED

**Problem**: When a chunk is retrieved, there's no mechanism to fetch adjacent chunks for fuller context.

**Current behavior**:

- Search returns individual chunks
- No chunk position tracking relative to siblings
- No parent-child document hierarchy
- Cannot expand context window with adjacent chunks

**Impact**:

- Retrieved chunks may lack necessary surrounding context
- Questions requiring broader context get incomplete answers
- Cannot validate or expand findings by looking at adjacent content

### 3. Limited Chunk Metadata (MEDIUM PRIORITY) ✅ SOLVED (position tracking)

**Problem**: Chunks only track basic metadata, missing valuable context information.

**Missing metadata**:

- Chunk position within section (first/middle/last)
- Previous/next chunk IDs for navigation
- Paragraph vs table vs list type classification
- Named entity mentions (companies, metrics, dates)
- Semantic role (definition, example, data, analysis)

**Impact**:

- Cannot implement intelligent context expansion
- Difficult to assess chunk relevance beyond similarity score
- Cannot filter by content type (e.g., "only return tables")

### 4. Suboptimal Table Handling (MEDIUM PRIORITY) ✅ SOLVED

**Problem**: Tables are converted to markdown and chunked like text, but they have special properties.

**Issues**:

- Large tables split across chunks lose header context
- Table captions may be in a different chunk from data
- Column headers repeated in every chunk (inefficient)
- Markdown table format may not embed well
- No table-specific search (e.g., "find tables with revenue data")

**Impact**:

- Financial tables (key data for analysis) are harder to retrieve
- Multi-year comparison tables fragmented
- Cannot execute table-specific queries

### 5. No Query Expansion or Reformulation (MEDIUM PRIORITY) ✅ SOLVED

**Problem**: Single query embedding may miss relevant documents using different terminology.

**Missing features**:

- No synonym expansion (e.g., "revenue" vs "sales" vs "top line")
- No query reformulation for failed searches
- No multi-query generation for complex questions
- No acronym expansion (NIM, NRR, ARR, etc.)

**Impact**:

- Terminology mismatches reduce recall
- Domain-specific jargon not handled
- Users must know exact terminology from filings

### 6. Chunk Size Tradeoffs (LOW-MEDIUM PRIORITY) PARTIAL

**Problem**: Fixed 1000-token chunks are a one-size-fits-all approach.

**Tradeoffs**:

- **Small chunks**: Better precision, worse context, more chunks to search
- **Large chunks**: Better context, worse precision, fewer relevant results
- **Current (1000)**: Middle ground but not optimal for all content types

**Content-specific needs**:

- Executive summaries: Larger chunks (preserve narrative flow)
- Risk factors: Medium chunks (one risk per chunk)
- Financial tables: Special handling (keep table intact)
- MD&A sections: Larger chunks (preserve analytical context)

**Impact**:

- Suboptimal retrieval for certain content types
- More GPU memory usage than necessary
- Longer embedding times

### 7. Limited Hybrid Search Tuning (LOW PRIORITY) ✅ SOLVED

**Problem**: Hybrid search uses fixed 0.7 vector / 0.3 FTS weighting.

**Issues**:

- Optimal weights vary by query type
- No query classification for weight selection
- No A/B testing or feedback loop
- No per-document-type weighting

**Query type examples**:

- Exact metric lookups: Higher FTS weight (e.g., "Q2 2024 revenue")
- Conceptual questions: Higher vector weight (e.g., "What are the main risks?")
- Named entity queries: Higher FTS weight (e.g., "{COMPANY} AG")

**Impact**:

- Suboptimal ranking for some query types
- Cannot tune per use case

### 8. No Multi-Hop Reasoning (LOW PRIORITY)

**Problem**: Complex questions requiring multiple retrieval steps are answered in a single pass.

**Example multi-hop queries**:

- "How has gross margin changed over the last 3 years and what were the drivers?"

  - Step 1: Find gross margin data across periods
  - Step 2: Find MD&A discussions of margin changes
  - Step 3: Synthesize trends and causes

- "Compare capital allocation priorities mentioned in 2023 vs 2024"

  - Step 1: Find 2023 capital allocation discussion
  - Step 2: Find 2024 capital allocation discussion
  - Step 3: Extract and compare priorities

**Impact**:

- Complex analytical questions get incomplete answers
- User must break down questions manually
- Cannot follow reasoning chains

### 9. No Semantic Chunking (MEDIUM PRIORITY)

**Problem**: Chunks are created by token count, not semantic boundaries.

**Current approach**:

- Split on paragraphs until reaching ~1000 tokens
- Overlap of 200 tokens
- No understanding of semantic completeness

**Better approach**:

- Identify logical units (complete thoughts, arguments, tables)
- Keep related content together
- Split only at major semantic boundaries
- Variable-size chunks based on content

**Impact**:

- Artificially split coherent content
- Search returns incomplete thoughts
- Lower quality embeddings for split content

### 10. Missing Negative Examples for Reranking (LOW PRIORITY)

**Problem**: Reranker only sees retrieved candidates, no hard negatives.

**Current reranking**:

- Fetch top 50 from hybrid search
- Rerank to top 10
- No training or fine-tuning on domain data

**Potential improvements**:

- Collect user feedback on result quality
- Generate hard negatives (high similarity but wrong answer)
- Fine-tune reranker on financial document pairs
- Use contrastive learning

**Impact**:

- Suboptimal reranking for domain-specific queries
- Cannot improve over time with usage

## Detailed Design Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation including:

- Component architecture (Parser, Embeddings, Vector Store, Search, Query)
- Design decisions for parent-child hierarchy, chunk tracking, table handling
- Query expansion and classification designs
- Data flow diagrams

## Implementation Roadmap

### Phase 1: Context Expansion ✅ COMPLETE (Dec 2025)

**Priority**: HIGH - Addresses biggest limitation

1. ~~Implement parent-child document hierarchy~~ ✅ Chunk linking with prev/next IDs
1. ~~Add chunk position tracking and context expansion~~ ✅ Position tracking in parser.py
1. ~~Update search interface to use expanded context~~ ✅ `get_chunk_with_context()`, `expand_results_with_context()`
1. ~~Write tests and documentation~~ ✅ Tests in test_parser.py, test_search.py
1. ~~Parent-child section retrieval~~ ✅ (Dec 13) Full section retrieval with size-aware fallback

**Implementation Details (Dec 13)**:

- `Section` dataclass with `section_id`, `approx_tokens` property
- `Chunk` extended with `section_id` field linking to parent section
- `sections.json` stores full section content (indexed during `pitch index`)
- `VectorStore.store_section()` and `get_section_for_chunk()` methods
- `SearchConfig.max_section_tokens` (default 8000) controls section vs. window mode
- `Searcher.get_parent_context()` returns full section when small, falls back to window for large sections
- `Searcher.expand_results_with_parent()` for batch section retrieval with deduplication
- Tests in `tests/test_parent_child.py` (20 tests)

**Impact**: Dramatically improves answer quality by providing fuller context

### Phase 2: Table Intelligence - ✅ COMPLETE (Dec 21, 2025)

**Priority**: MEDIUM-HIGH - Critical for financial analysis

**Test Corpus**: 50 diverse tables in `tests/fixtures/table_corpus.json`

- Filing types: 12 10-K, 12 10-Q, 8 8-K, 18 DEF 14A
- Table types: comparison (19), other (22), compensation (4), governance (4), metrics (1)
- Sizes: small 2-5 rows (19), medium 6-20 rows (18), large 21+ rows (13)

**Problems Solved**:

1. ~~Tables split across chunks~~ → Tables under 2000 tokens kept intact; larger tables split with headers repeated
1. ~~No table-specific search~~ → `content_type="table"`, `table_type="compensation"` filters
1. ~~No table classification~~ → `classify_table()` with 6 types via regex patterns

**Design Decisions (Dec 21)**:

1. **Caption extraction**: Don't rely on caption-only embeddings. Embed full table markdown, prepend section header as context.
1. **Dual storage**: Store both raw HTML and markdown. HTML for future rendering, markdown for search/embedding.
1. **Structured extraction**: DEFERRED. Too fragile, low ROI. Focus on making tables searchable as text first.

**Implementation (All Complete)**:

1. **Table Detection & Extraction** ✅

   - `Table` dataclass with `table_id`, `html`, `markdown`, `caption`, `table_type`, `approx_tokens`
   - `Table.from_html()` class method for creation
   - `convert_tables_to_readable()` returns `Table` objects with classification
   - Files: `parser.py`

1. **Table-Aware Chunking** ✅

   - Tables under 2000 tokens kept intact as single chunks
   - Larger tables split by rows with headers repeated
   - `_split_large_table()` and `_is_table_paragraph()` helpers
   - `_split_into_paragraphs()` preserves table newline structure
   - Files: `parser.py`

1. **Table Classification** ✅

   - `classify_table()` with pattern matching on caption + content
   - Types: `financial_statement`, `compensation`, `governance`, `comparison`, `schedule`, `metrics`, `other`
   - `TABLE_TYPE_PATTERNS` dict with regex patterns for each type
   - ~44% accuracy on test corpus (acceptable for simple regex approach)
   - Files: `parser.py`

1. **Search Filters** ✅

   - `content_type`, `table_type`, `is_table_continuation` added to chunk metadata
   - `FILTERABLE_COLUMNS` updated in `store.py`
   - `Searcher.search()` accepts `content_type` and `table_type` parameters
   - Files: `store.py`, `search.py`

1. **Testing & Validation** ✅

   - `tests/test_table_intelligence.py` with 25 tests
   - Tests cover: classification, dataclass, chunking, metadata propagation, corpus validation
   - All tests passing

**Deferred for Future**:

- Caption-only embeddings (fragile caption extraction)
- Structured data extraction (Revenue: $1.2B → queryable fields)
- Complex merged cell handling (store HTML, deal with later)

**Impact**: Better handling of critical financial data

### Phase 3: Query Enhancement ✅ COMPLETE (Dec 12, 2025)

**Priority**: MEDIUM - Improves recall

1. ~~Build financial domain synonym/acronym dictionary~~ ✅ 120+ acronyms, 50+ synonym clusters
1. ~~Implement query expansion~~ ✅ `src/rag/query_expansion.py`
1. ~~Query classification for dynamic weighting~~ ✅ `src/rag/query_classifier.py`
1. ~~Test and tune~~ ✅ Tests in `tests/test_query_expansion.py`

**Implementation Details**:

- `QueryExpander` class with `ACRONYMS` dict (ARR, NRR, NIM, EBITDA, SBC, etc.)
- `SYNONYMS` dict with clusters (revenue/sales/top line, profit/earnings/net income, etc.)
- `expand()` method generates query variations
- Integrated into `Searcher.search()` with `expand_query=True` parameter
- Multi-query search with automatic result deduplication

**Impact**: Reduces terminology mismatches, better ranking

### Phase 4: Advanced Chunking (2 weeks) - PARTIALLY COMPLETE

**Priority**: MEDIUM - Quality improvement

1. Implement semantic chunking (5 days)
1. ~~Content-type-specific chunk sizes (2 days)~~ ✅ DONE
1. Validate and tune chunk quality (3 days)

**Impact**: More coherent chunks, better embeddings

#### 4.1 Semantic Chunking (NOT STARTED)

**Goal**: Create chunks at semantic boundaries rather than arbitrary token counts.

**Implementation Approach**:

Create `src/rag/semantic_chunker.py`:

```python
"""Semantic chunking based on document structure and topic coherence."""

from dataclasses import dataclass

# Optional: Use spaCy for sentence segmentation
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


@dataclass
class SemanticChunk:
    """A semantically coherent chunk."""
    text: str
    chunk_type: str  # "paragraph", "list", "heading", "table"
    confidence: float  # How confident we are this is a complete unit


class SemanticChunker:
    """
    Create chunks at semantic boundaries.

    Strategies:
    1. Identify paragraph boundaries
    2. Keep lists together
    3. Detect topic shifts via sentence similarity
    4. Keep tables intact (via table markers)
    5. Respect heading boundaries
    """

    def __init__(
        self,
        min_chunk_tokens: int = 200,
        max_chunk_tokens: int = 2000,
        target_chunk_tokens: int = 800,
    ):
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.target_chunk_tokens = target_chunk_tokens

    def chunk(self, text: str) -> list[SemanticChunk]:
        """Create semantic chunks from text."""
        # Step 1: Split into structural units (paragraphs, lists, etc.)
        units = self._split_into_units(text)
        # Step 2: Merge small units, split large ones
        chunks = self._balance_chunks(units)
        return chunks
```

**Key Methods**:

- `_split_into_units()`: Split on double newlines, classify as paragraph/list/heading/table
- `_balance_chunks()`: Merge small units and split large ones to reach target size
- `_merge_units()`: Combine multiple units into a single chunk
- `_split_large_unit()`: Split at sentence boundaries using spaCy or regex

**External Dependencies** (optional):

```bash
poetry add spacy
python -m spacy download en_core_web_sm
```

**Integration with Parser**:

Modify `SECFilingParser.chunk_section()` to optionally use semantic chunking:

```python
def chunk_section(
    self,
    section: Section,
    use_semantic_chunking: bool = False,  # New parameter
) -> list[Chunk]:
    if use_semantic_chunking:
        from .semantic_chunker import SemanticChunker
        chunker = SemanticChunker(...)
        semantic_chunks = chunker.chunk(section.content)
        # Convert to Chunk objects with metadata
```

**Index Rebuild Required**: YES

### Phase 5: Continuous Improvement (1 week) - PARTIALLY COMPLETE

**Priority**: LOW-MEDIUM - Enables long-term optimization

1. ~~Add feedback logging (1 day)~~ ✅ DONE
1. Build analysis tools (2 days) - `analyze_feedback()` exists, CLI command pending
1. Implement reranker fine-tuning pipeline (2 days)

**Impact**: Enables data-driven improvements over time

#### 5.1 Analysis CLI Command (NOT STARTED)

**Goal**: CLI command to analyze feedback logs and identify improvement opportunities.

**Implementation** (`src/cli.py`):

```python
@app.command()
def analyze_feedback(
    log_path: Path = typer.Option(Path("logs/search_feedback.jsonl"), help="Path to feedback log"),
    days: int = typer.Option(7, help="Analyze last N days"),
):
    """Analyze search feedback logs."""
    from collections import Counter
    from datetime import datetime, timedelta
    from src.rag.feedback import FeedbackLogger

    logger = FeedbackLogger(log_path)
    entries = logger.read_all()

    # Filter by date
    cutoff = datetime.now() - timedelta(days=days)
    recent = [e for e in entries if datetime.fromisoformat(e.timestamp) > cutoff]

    # Output analysis:
    # - Query type distribution
    # - Average scores by query type
    # - Low-performing queries (top score < 0.5)
    # - Most retrieved sections
```

**Analysis Outputs**:

- Query type distribution (what kinds of queries are most common)
- Average top-3 score by query type (which query types perform poorly)
- Low-scoring queries (candidates for synonym/acronym additions)
- Most retrieved sections (where is the most relevant content)

#### 5.2 Reranker Fine-Tuning (NOT STARTED)

**Goal**: Fine-tune the cross-encoder reranker on domain-specific data.

**Approach**:

1. Collect positive pairs from high-scoring search results
1. Generate hard negatives (high similarity but wrong answer)
1. Fine-tune BGE Reranker on financial document pairs
1. Evaluate on held-out test set

**Data Collection**:

- Use feedback logs to identify queries with high user engagement
- Positive: (query, high-scoring relevant chunk)
- Negative: (query, high-similarity but irrelevant chunk)

**Index Rebuild Required**: NO (reranker is query-time only)

### Phase 6: Advanced Features (3 weeks) - NOT STARTED

**Priority**: LOW - Nice-to-have enhancements

1. Multi-hop retrieval (6 days)
1. Enhanced chunk metadata with NER (4 days)
1. Performance optimization and caching (3 days)

**Impact**: Handles sophisticated analytical queries

#### 6.1 Multi-Hop Retrieval (NOT STARTED)

**Goal**: Decompose complex questions and perform iterative retrieval.

**Use Cases**:

- "How has gross margin changed over the last 3 years and what were the drivers?"
- "Compare capital allocation priorities in 2023 vs 2024"
- "What is the company's strategy evolution from FY23 to FY24?"

**Implementation** (`src/rag/multihop.py`):

```python
"""Multi-hop retrieval for complex questions."""

from dataclasses import dataclass
from anthropic import Anthropic


@dataclass
class HopResult:
    """Result from a single retrieval hop."""
    sub_question: str
    answer: str
    sources: list
    reasoning: str


@dataclass
class MultiHopResponse:
    """Response from multi-hop retrieval."""
    original_question: str
    sub_questions: list[str]
    hop_results: list[HopResult]
    final_answer: str
    reasoning_chain: str


class MultiHopRAG:
    """Multi-step retrieval for complex questions."""

    DECOMPOSE_PROMPT = """Break down this complex financial research question
    into 2-4 simpler sub-questions that can be answered independently.

    Original question: {question}

    Return ONLY a JSON array of sub-questions."""

    SYNTHESIZE_PROMPT = """Synthesize information from multiple sources to answer:

    Original question: {question}

    Sub-question answers:
    {sub_answers}

    Provide a comprehensive answer citing relevant details."""

    def query_multihop(self, question: str, max_hops: int = 4) -> MultiHopResponse:
        # Step 1: Decompose question using LLM
        sub_questions = self._decompose_question(question, max_hops)
        # Step 2: Answer each sub-question with RAG
        hop_results = [self._answer_sub_question(sq) for sq in sub_questions]
        # Step 3: Synthesize final answer from hop results
        final_answer, reasoning = self._synthesize_answer(question, hop_results)
        return MultiHopResponse(...)
```

**Index Rebuild Required**: NO (query-time only)

#### 6.2 Enhanced Chunk Metadata with NER (NOT STARTED)

**Goal**: Enrich chunks with named entities, metrics, and semantic metadata.

**Implementation** (`src/rag/enrichment.py`):

```python
"""Chunk enrichment with NER and semantic metadata."""

from dataclasses import dataclass, field

# Optional: spaCy for NER
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


@dataclass
class EnrichedMetadata:
    """Semantic metadata extracted from chunk."""
    entities: list[str] = field(default_factory=list)  # Named entities
    metrics: list[str] = field(default_factory=list)   # Financial metrics
    time_periods: list[str] = field(default_factory=list)  # Dates/periods
    has_numbers: bool = False
    has_percentages: bool = False
    has_currency: bool = False
    sentiment: str = "neutral"  # positive/negative/neutral
    content_indicators: list[str] = field(default_factory=list)


class ChunkEnricher:
    """Extract semantic metadata from chunks."""

    METRIC_PATTERNS = [
        r'\b(revenue|sales|income|profit|EBITDA|margin|growth|return)\b',
        r'\b(ROE|ROA|ROI|ROIC|EPS|P/E|NIM|ARR|MRR|NRR)\b',
    ]

    TIME_PATTERNS = [
        r'\b(Q[1-4])\s*(\d{4})?\b',
        r'\b(FY|fiscal\s+year)\s*(\d{4})?\b',
        r'\b(20\d{2})\b',
    ]

    def enrich(self, text: str) -> EnrichedMetadata:
        # Extract named entities (using spaCy if available)
        # Extract financial metrics via regex
        # Extract time periods via regex
        # Detect numerical content
        # Simple sentiment indicators
        ...
```

**Search Enhancements**:

- Filter searches: "Find chunks mentioning EBITDA"
- Boost results: Prefer chunks with numbers for metric queries
- Better prompts: Tell Claude what type of content it's seeing

**External Dependencies** (optional):

```bash
poetry add spacy
python -m spacy download en_core_web_sm
```

**Index Rebuild Required**: YES

#### 6.3 Performance Optimization and Caching (NOT STARTED)

**Goal**: Improve search latency and reduce redundant computation.

**Potential Optimizations**:

1. **Query result caching**: Cache frequently repeated queries
1. **Embedding caching**: Cache query embeddings to avoid re-computation
1. **Batch reranking**: Optimize reranker batch sizes for GPU memory
1. **Index partitioning**: Partition large indexes by filing type or date

**Index Rebuild Required**: NO

## Quick Wins ✅ IMPLEMENTED (Dec 10, 2025)

All quick wins have been implemented and tested (297 tests passing):

1. **Content-type-specific chunk sizes** ✅

   - `src/rag/parser.py`: Added `CHUNK_SIZE_BY_SECTION` config
   - Section-specific sizes: Item 1 (1200), Item 1A/7 (1500), Item 8 (800), default (1000)
   - `_get_chunk_size_for_section()` method routes to correct size

1. **Chunk position tracking** ✅

   - `src/rag/parser.py`: Extended `Chunk` dataclass with position fields
   - Tracks: `position_in_section`, `total_chunks_in_section`, `is_section_start`, `is_section_end`
   - Enables future context expansion features

1. **Query classification for hybrid weights** ✅

   - `src/rag/query_classifier.py`: New module with pattern-based classification
   - Query types: exact_metric, conceptual, entity, trend, comparison, default
   - Dynamic weights per type (e.g., exact_metric uses 0.3v/0.7f for FTS emphasis)
   - Integrated into `search.py` with `auto_weight` config option (default: True)

1. **Feedback logging** ✅

   - `src/rag/feedback.py`: New module for JSONL-based search logging
   - Logs: query, mode, weights, classification, top results preview
   - Output: `index/{ticker}/search_feedback.jsonl`
   - `analyze_feedback()` function for pattern analysis
   - Integrated into `search.py` with `log_feedback` config option (default: True)

## Conclusion

The current RAG implementation is solid and production-ready, but has room for significant improvements. The highest-priority enhancements focus on **context expansion** and **specialized table handling**, which address the core limitations of chunk-based retrieval for financial documents.

The recommended roadmap prioritizes high-impact, medium-complexity improvements first, followed by longer-term enhancements that enable continuous optimization. With the Phase 1 and Phase 2 improvements alone, the system would handle complex financial analysis queries substantially better.
