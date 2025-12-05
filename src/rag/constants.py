"""Constants and configuration defaults for RAG module.

Centralizes magic numbers and default values for better maintainability
and discoverability. Import from here rather than hardcoding values.
"""

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# Default chunk size in tokens for text splitting
DEFAULT_CHUNK_SIZE = 1000

# Overlap between chunks to maintain context
DEFAULT_CHUNK_OVERLAP = 200

# Minimum chunk size - chunks smaller than this are merged
MIN_CHUNK_SIZE = 100

# Maximum tokens for a single table chunk before splitting
TABLE_MAX_TOKENS = 2000

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

# Default number of results to return from search
DEFAULT_TOP_K = 10

# Maximum context tokens for RAG queries
DEFAULT_CONTEXT_MAX_TOKENS = 4000

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Default max tokens for RAG response generation
DEFAULT_RAG_MAX_TOKENS = 2048

# Agent-specific max tokens for different response types
AGENT_MAX_TOKENS_ANSWER = 2000  # Full answer generation
AGENT_MAX_TOKENS_DECOMPOSE = 800  # Query decomposition
AGENT_MAX_TOKENS_CLASSIFY = 500  # Query classification
AGENT_MAX_TOKENS_CONFIDENCE = 300  # Confidence assessment

# Default top_k for agent searches (smaller for precision)
AGENT_DEFAULT_TOP_K = 5

# =============================================================================
# PREVIEW/DISPLAY LIMITS
# =============================================================================

# Maximum characters for text previews in logs/UI
PREVIEW_MAX_CHARS = 500

# Maximum characters for context snippets
CONTEXT_SNIPPET_MAX_CHARS = 2000
