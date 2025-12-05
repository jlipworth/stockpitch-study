"""RAG module for document parsing, embedding, and search."""

from .analyst_parser import (
    AnalystReportMetadata,
    AnalystReportParser,
    ParsedAnalystReport,
    ReportSection,
    parse_analyst_report,
)
from .base_parser import BaseParser, ParsedDocument
from .constants import (
    AGENT_DEFAULT_TOP_K,
    AGENT_MAX_TOKENS_ANSWER,
    AGENT_MAX_TOKENS_CLASSIFY,
    AGENT_MAX_TOKENS_CONFIDENCE,
    AGENT_MAX_TOKENS_DECOMPOSE,
    CONTEXT_SNIPPET_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONTEXT_MAX_TOKENS,
    DEFAULT_RAG_MAX_TOKENS,
    DEFAULT_TOP_K,
    MIN_CHUNK_SIZE,
    PREVIEW_MAX_CHARS,
    TABLE_MAX_TOKENS,
)
from .embeddings import (
    BATCH_SIZE_BY_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_RERANKER,
    EmbeddedChunk,
    EmbeddingModel,
    GPUMemoryInfo,
    Reranker,
    RerankResult,
    batch_cosine_similarity,
    calculate_batch_size,
    clear_gpu_cache,
    cosine_similarity,
    detect_device,
    get_default_batch_size,
    get_gpu_memory,
)
from .parser import (
    Chunk,
    ParsedFiling,
    SECFilingParser,
    Section,
)
from .query import (
    DEFAULT_RAG_MODEL,
    RAGQueryEngine,
    RAGResponse,
    create_rag_engine,
)
from .query_expansion import (
    ExpandedQuery,
    QueryExpander,
    expand_query,
    get_expander,
)
from .search import (
    QueryClassifierFn,
    SearchConfig,
    Searcher,
    SearchMode,
    create_searcher,
)
from .section_patterns import (
    CHUNK_SIZE_BY_SECTION,
    SECTION_PATTERNS,
    SECTION_PATTERNS_8K,
    SECTION_PATTERNS_10K,
    SECTION_PATTERNS_10Q,
    SECTION_PATTERNS_DEF14A,
    get_section_patterns,
    get_section_regex,
)
from .store import (
    IndexManifest,
    SearchResult,
    VectorStore,
)
from .table_utils import (
    Table,
    classify_table,
    convert_tables_to_markdown,
    convert_tables_to_readable,
    extract_tables,
    render_table_as_markdown,
)
from .transcript_parser import (
    ParsedTranscript,
    Speaker,
    SpeakerBlock,
    TranscriptMetadata,
    TranscriptParser,
    parse_transcript,
)

__all__ = [
    # Constants
    "AGENT_DEFAULT_TOP_K",
    "AGENT_MAX_TOKENS_ANSWER",
    "AGENT_MAX_TOKENS_CLASSIFY",
    "AGENT_MAX_TOKENS_CONFIDENCE",
    "AGENT_MAX_TOKENS_DECOMPOSE",
    "CONTEXT_SNIPPET_MAX_CHARS",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CONTEXT_MAX_TOKENS",
    "DEFAULT_RAG_MAX_TOKENS",
    "DEFAULT_TOP_K",
    "MIN_CHUNK_SIZE",
    "PREVIEW_MAX_CHARS",
    "TABLE_MAX_TOKENS",
    # Base Parser ABC
    "BaseParser",
    "ParsedDocument",
    # SEC Parser
    "SECFilingParser",
    "ParsedFiling",
    "Section",
    "Chunk",
    "SECTION_PATTERNS",
    "SECTION_PATTERNS_10K",
    "SECTION_PATTERNS_10Q",
    "SECTION_PATTERNS_8K",
    "SECTION_PATTERNS_DEF14A",
    "CHUNK_SIZE_BY_SECTION",
    "get_section_patterns",
    "get_section_regex",
    # Table Utils
    "Table",
    "classify_table",
    "convert_tables_to_markdown",
    "convert_tables_to_readable",
    "extract_tables",
    "render_table_as_markdown",
    # Transcript Parser
    "TranscriptParser",
    "ParsedTranscript",
    "TranscriptMetadata",
    "Speaker",
    "SpeakerBlock",
    "parse_transcript",
    # Analyst Report Parser
    "AnalystReportParser",
    "ParsedAnalystReport",
    "AnalystReportMetadata",
    "ReportSection",
    "parse_analyst_report",
    # Embeddings
    "EmbeddingModel",
    "EmbeddedChunk",
    "detect_device",
    "get_default_batch_size",
    "get_gpu_memory",
    "calculate_batch_size",
    "clear_gpu_cache",
    "GPUMemoryInfo",
    "cosine_similarity",
    "batch_cosine_similarity",
    "DEFAULT_MODEL",
    "BATCH_SIZE_BY_DEVICE",
    # Reranking
    "Reranker",
    "RerankResult",
    "DEFAULT_RERANKER",
    # Store
    "VectorStore",
    "SearchResult",
    "IndexManifest",
    # Search
    "Searcher",
    "SearchConfig",
    "SearchMode",
    "QueryClassifierFn",
    "create_searcher",
    # RAG Query
    "RAGQueryEngine",
    "RAGResponse",
    "create_rag_engine",
    "DEFAULT_RAG_MODEL",
    # Query Expansion
    "QueryExpander",
    "ExpandedQuery",
    "expand_query",
    "get_expander",
]
