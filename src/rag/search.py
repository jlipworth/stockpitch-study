"""High-level search interface combining embeddings and vector store."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .constants import DEFAULT_CONTEXT_MAX_TOKENS, DEFAULT_TOP_K
from .embeddings import DEFAULT_MODEL, EmbeddingModel, Reranker, clear_gpu_cache
from .feedback import FeedbackLogger
from .query_classifier import QueryClassification, classify_query
from .query_expansion import ExpandedQuery, expand_query
from .store import SearchResult, VectorStore

# Type alias for classifier function
QueryClassifierFn = Callable[[str], QueryClassification]

logger = logging.getLogger(__name__)

SearchMode = Literal["vector", "fts", "hybrid"]


@dataclass
class SearchConfig:
    """Configuration for search behavior."""

    mode: SearchMode = "hybrid"
    top_k: int = DEFAULT_TOP_K
    vector_weight: float = 0.7  # For hybrid search (default, may be overridden by auto_weight)
    rerank: bool = False  # Whether to rerank results
    rerank_top_n: int = 30  # Candidates to fetch before reranking (reduced from 50 for GPU memory)
    auto_weight: bool = True  # Quick Win #3: Auto-adjust weights by query type
    log_feedback: bool = True  # Quick Win #4: Log search queries for analysis
    expand_query: bool = True  # Expand queries with acronyms/synonyms for better recall
    # Parent-child retrieval settings
    max_section_tokens: int = 8000  # Max tokens for full section retrieval (~32K chars)
    context_window: int = 2  # Fallback: chunks before/after when section too large


class Searcher:
    """High-level search interface for querying company documents."""

    def __init__(
        self,
        index_dir: Path,
        ticker: str,
        embedding_model: EmbeddingModel | None = None,
        reranker: Reranker | None = None,
        config: SearchConfig | None = None,
        classifier: QueryClassifierFn | None = None,
    ):
        """
        Initialize searcher for a company.

        Args:
            index_dir: Base directory for indexes
            ticker: Company ticker
            embedding_model: Optional pre-loaded embedding model
            reranker: Optional reranker for improved precision
            config: Search configuration
            classifier: Optional query classifier function. Defaults to classify_query.
                        Enables testing with mock classifiers or swapping strategies.
        """
        self.ticker = ticker.upper()
        self.config = config or SearchConfig()

        # Lazy load embedding model
        self._embedding_model = embedding_model
        self._model_name = DEFAULT_MODEL

        # Lazy load reranker (only if config.rerank is True)
        self._reranker = reranker

        # Query classifier (dependency injection for testability)
        self._classifier = classifier or classify_query

        # Initialize vector store
        self.store = VectorStore(
            index_dir=index_dir,
            ticker=self.ticker,
        )

        # Quick Win #4: Initialize feedback logger
        self._feedback_logger: FeedbackLogger | None = None
        self._index_dir = index_dir

        if self.store.count() == 0:
            raise ValueError(f"No index found for {self.ticker}. Run 'pitch index {self.ticker}' first.")

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel(model_name=self._model_name)
        return self._embedding_model

    @property
    def reranker(self) -> Reranker:
        """Lazy load reranker."""
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def feedback_logger(self) -> FeedbackLogger:
        """Lazy load feedback logger."""
        if self._feedback_logger is None:
            self._feedback_logger = FeedbackLogger(self._index_dir, self.ticker)
        return self._feedback_logger

    def search(
        self,
        query: str,
        mode: SearchMode | None = None,
        top_k: int | None = None,
        filing_type: str | list[str] | None = None,
        section: str | None = None,
        rerank: bool | None = None,
        auto_weight: bool | None = None,
        expand: bool | None = None,
        # Phase 2: Table Intelligence filters
        content_type: str | None = None,
        table_type: str | None = None,
        # Fiscal period filters
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
    ) -> list[SearchResult]:
        """
        Search indexed documents.

        Args:
            query: Search query text
            mode: Search mode (vector, fts, hybrid). Uses config default if None.
            top_k: Number of results. Uses config default if None.
            filing_type: Filter by filing type (e.g., "10-K") or list of filing types
            section: Filter by section (e.g., "Item 1A")
            rerank: Whether to rerank results with cross-encoder. Uses config default if None.
            auto_weight: Whether to auto-adjust weights by query type (Quick Win #3).
            expand: Whether to expand query with acronyms/synonyms. Uses config default if None.
            content_type: Filter by content type ("text" or "table"). Phase 2.
            table_type: Filter by table type (e.g., "financial_statement", "compensation"). Phase 2.
            fiscal_year: Filter by fiscal year (e.g., 2025).
            fiscal_quarter: Filter by fiscal quarter (1-4).

        Returns:
            List of SearchResult objects sorted by relevance
        """
        mode = mode or self.config.mode
        top_k = top_k or self.config.top_k
        should_rerank = rerank if rerank is not None else self.config.rerank
        use_auto_weight = auto_weight if auto_weight is not None else self.config.auto_weight
        should_expand = expand if expand is not None else self.config.expand_query

        # Query expansion for better recall
        expanded: ExpandedQuery | None = None
        fts_query = query  # Default to original query for FTS
        if should_expand:
            expanded = expand_query(query)
            if expanded.expansions:
                fts_query = expanded.fts_query
                logger.debug(
                    f"Query expanded: {len(expanded.expansions)} expansions, "
                    f"acronyms={expanded.acronyms_found}, synonyms={expanded.synonyms_found}"
                )

        # Quick Win #3: Classify query and adjust weights
        query_classification: QueryClassification | None = None
        if mode == "hybrid" and use_auto_weight:
            query_classification = self._classifier(query)
            vector_weight = query_classification.vector_weight
            logger.debug(
                f"Query classified as '{query_classification.query_type}' "
                f"(confidence={query_classification.confidence:.2f}), "
                f"weights: vector={vector_weight:.1f}, fts={query_classification.fts_weight:.1f}"
            )
        else:
            vector_weight = self.config.vector_weight

        logger.debug(
            f"Search for {self.ticker}: mode={mode}, top_k={top_k}, "
            f"rerank={should_rerank}, filing_type={filing_type}, section={section}"
        )

        # If reranking, fetch more candidates first
        fetch_k = self.config.rerank_top_n if should_rerank else top_k

        # Build filter dict
        filter_dict = {}
        if filing_type:
            filter_dict["filing_type"] = filing_type
        if section:
            filter_dict["section"] = section
        # Phase 2: Table Intelligence filters
        if content_type:
            filter_dict["content_type"] = content_type
        if table_type:
            filter_dict["table_type"] = table_type
        # Fiscal period filters
        if fiscal_year:
            filter_dict["fiscal_year"] = fiscal_year
        if fiscal_quarter:
            filter_dict["fiscal_quarter"] = fiscal_quarter

        # Initial retrieval
        if mode == "vector":
            query_embedding = self.embedding_model.encode_query(query)
            results = self.store.search_vector(
                query_embedding,
                top_k=fetch_k,
                filter_dict=filter_dict if filter_dict else None,
            )

        elif mode == "fts":
            results = self.store.search_fts(
                fts_query,  # Use expanded query for FTS
                top_k=fetch_k,
                filter_dict=filter_dict if filter_dict else None,
            )

        else:  # hybrid
            query_embedding = self.embedding_model.encode_query(query)  # Original query for embedding
            results = self.store.search_hybrid(
                query_embedding,
                fts_query,  # Use expanded query for FTS portion
                top_k=fetch_k,
                vector_weight=vector_weight,  # Uses auto-classified weight if enabled
                filter_dict=filter_dict if filter_dict else None,
            )

        # Rerank if enabled
        if should_rerank and results:
            logger.debug(f"Reranking {len(results)} results")
            # Clear GPU cache before loading reranker to prevent OOM
            clear_gpu_cache()
            results = self._rerank_results(query, results, top_k)
            # Clear GPU cache after reranking to free memory
            clear_gpu_cache()

        final_results = results[:top_k]

        # Quick Win #4: Log search feedback
        if self.config.log_feedback:
            try:
                self.feedback_logger.log_search(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    vector_weight=vector_weight,
                    query_type=query_classification.query_type if query_classification else "default",
                    query_confidence=query_classification.confidence if query_classification else 0.5,
                    results=final_results,
                    reranked=should_rerank,
                    filing_type_filter=str(filing_type) if filing_type else None,
                    section_filter=section,
                    expanded_query=fts_query if expanded and expanded.expansions else None,
                    expansion_count=len(expanded.expansions) if expanded else 0,
                )
            except Exception as e:
                logger.debug(f"Failed to log search feedback: {e}")

        logger.info(f"Search completed: returned {len(final_results)} results")
        return final_results

    def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return

        Returns:
            Reranked list of SearchResult
        """
        texts = [r.text for r in results]
        metadata = [r.metadata for r in results]

        reranked = self.reranker.rerank(
            query=query,
            texts=texts,
            top_k=top_k,
            metadata=metadata,
        )

        # Convert back to SearchResult with rerank scores
        return [
            SearchResult(
                text=r.text,
                score=r.score,  # Use rerank score
                metadata=r.metadata,
            )
            for r in reranked
        ]

    # Phase 1.1: Context expansion methods
    def get_chunk_with_context(
        self,
        chunk_id: str,
        before: int = 1,
        after: int = 1,
    ) -> list[SearchResult]:
        """
        Fetch a chunk with surrounding context chunks.

        Phase 1.1: Enables richer context by traversing chunk links.

        Args:
            chunk_id: ID of the target chunk
            before: Number of chunks to fetch before
            after: Number of chunks to fetch after

        Returns:
            List of SearchResult in document order (before + target + after)
        """
        # Get the target chunk
        target = self.store.get_chunk_by_id(chunk_id)
        if target is None:
            return []

        # Traverse backwards
        current_id = target.prev_chunk_id
        prev_chunks: list[SearchResult] = []
        for _ in range(before):
            if not current_id:
                break
            chunk = self.store.get_chunk_by_id(current_id)
            if chunk:
                prev_chunks.insert(0, chunk)
                current_id = chunk.prev_chunk_id
            else:
                break

        # Traverse forwards
        current_id = target.next_chunk_id
        next_chunks: list[SearchResult] = []
        for _ in range(after):
            if not current_id:
                break
            chunk = self.store.get_chunk_by_id(current_id)
            if chunk:
                next_chunks.append(chunk)
                current_id = chunk.next_chunk_id
            else:
                break

        return prev_chunks + [target] + next_chunks

    def expand_results_with_context(
        self,
        results: list[SearchResult],
        context_window: int = 1,
        deduplicate: bool = True,
    ) -> list[SearchResult]:
        """
        Expand search results with adjacent chunks.

        Phase 1.1: Provides richer context for each search result.

        Args:
            results: Initial search results
            context_window: Number of chunks before/after to include
            deduplicate: Remove duplicate chunks from expanded results

        Returns:
            Expanded list of SearchResult
        """
        expanded: list[SearchResult] = []
        seen_ids: set[str] = set()

        for result in results:
            chunk_id = result.chunk_id
            if not chunk_id:
                # No chunk_id, just include the result as-is
                expanded.append(result)
                continue

            context_chunks = self.get_chunk_with_context(chunk_id, before=context_window, after=context_window)

            for chunk in context_chunks:
                if deduplicate and chunk.chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk.chunk_id)
                expanded.append(chunk)

        return expanded

    # =========================================================================
    # Parent-child retrieval: Full section or fallback to chunk + neighbors
    # =========================================================================

    def get_parent_context(
        self,
        result: SearchResult,
        max_tokens: int | None = None,
        fallback_window: int | None = None,
    ) -> tuple[str, str]:
        """
        Get parent section content for a search result, with size-aware fallback.

        If the parent section is under max_tokens, returns the full section.
        Otherwise, falls back to chunk + neighbors.

        Args:
            result: Search result to expand
            max_tokens: Max tokens for full section (uses config default if None)
            fallback_window: Chunks before/after for fallback (uses config default if None)

        Returns:
            Tuple of (content, mode) where mode is "section" or "window"
        """
        max_tokens = max_tokens or self.config.max_section_tokens
        fallback_window = fallback_window or self.config.context_window

        # If max_tokens is 0, always use window mode
        if max_tokens == 0:
            return self._get_window_context(result, fallback_window), "window"

        # Try to get parent section
        section = self.store.get_section_for_chunk(result.metadata)

        if section is None:
            # No section stored, fall back to window mode
            logger.debug(f"No section found for chunk {result.chunk_id}, using window mode")
            return self._get_window_context(result, fallback_window), "window"

        # Check section size
        approx_tokens = section.get("approx_tokens", 0)
        if approx_tokens > max_tokens:
            logger.debug(
                f"Section {result.section_id} too large ({approx_tokens} tokens > {max_tokens}), " "using window mode"
            )
            return self._get_window_context(result, fallback_window), "window"

        # Return full section
        logger.debug(f"Returning full section {result.section_id} ({approx_tokens} tokens)")
        return section.get("content", result.text), "section"

    def _get_window_context(self, result: SearchResult, window: int) -> str:
        """Get chunk with surrounding context using window mode."""
        if not result.chunk_id:
            return result.text

        chunks = self.get_chunk_with_context(result.chunk_id, before=window, after=window)
        if not chunks:
            return result.text

        return "\n\n".join(chunk.text for chunk in chunks)

    def expand_results_with_parent(
        self,
        results: list[SearchResult],
        max_tokens: int | None = None,
        fallback_window: int | None = None,
        deduplicate_sections: bool = True,
    ) -> list[tuple[SearchResult, str, str]]:
        """
        Expand search results with parent context.

        Args:
            results: Search results to expand
            max_tokens: Max tokens per section (uses config default if None)
            fallback_window: Window size for fallback (uses config default if None)
            deduplicate_sections: If True, only include each section once

        Returns:
            List of (original_result, expanded_content, mode) tuples
            mode is "section" or "window"
        """
        expanded = []
        seen_sections: set[str] = set()

        for result in results:
            section_id = result.section_id

            # Deduplicate by section if enabled
            if deduplicate_sections and section_id and section_id in seen_sections:
                continue

            content, mode = self.get_parent_context(result, max_tokens, fallback_window)
            expanded.append((result, content, mode))

            if section_id:
                seen_sections.add(section_id)

        return expanded

    def search_similar(
        self,
        text: str,
        top_k: int | None = None,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        """
        Find documents similar to a given text.

        Args:
            text: Text to find similar documents for
            top_k: Number of results
            exclude_self: If True, excludes exact matches

        Returns:
            List of similar documents
        """
        top_k = top_k or self.config.top_k

        # Encode as document (not query) for similarity search
        embedding = self.embedding_model.encode([text])[0]

        # Get extra results if excluding self
        fetch_k = top_k + 1 if exclude_self else top_k
        results = self.store.search_vector(embedding, top_k=fetch_k)

        if exclude_self and results:
            # Filter out exact text matches
            results = [r for r in results if r.text.strip() != text.strip()][:top_k]

        return results

    def get_context_window(
        self,
        query: str,
        max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
        mode: SearchMode | None = None,
        filing_type: str | list[str] | None = None,
        rerank: bool | None = None,
    ) -> tuple[list[SearchResult], str]:
        """
        Get search results formatted as a context window for LLM.

        Args:
            query: Search query
            max_tokens: Approximate max tokens for context
            mode: Search mode
            filing_type: Filter by filing type (string or list of strings)
            rerank: Whether to rerank results

        Returns:
            Tuple of (results, formatted_context_string)
        """
        # Estimate tokens per result (rough: 1 token ≈ 4 chars)
        avg_chunk_tokens = 300  # Rough estimate
        estimated_results = max(1, max_tokens // avg_chunk_tokens)

        results = self.search(
            query,
            mode=mode,
            top_k=estimated_results * 2,  # Fetch extra, trim later
            filing_type=filing_type,
            rerank=rerank,
        )

        # Build context string, respecting token limit
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough token-to-char ratio
        included_results = []

        for result in results:
            # Format this result
            header = f"[{result.filing_type} {result.filing_date} - {result.section}]"
            part = f"{header}\n{result.text}\n"
            part_chars = len(part)

            if total_chars + part_chars > max_chars:
                break

            context_parts.append(part)
            total_chars += part_chars
            included_results.append(result)

        context = "\n---\n".join(context_parts)
        return included_results, context


def create_searcher(
    index_dir: Path,
    ticker: str,
    mode: SearchMode = "hybrid",
    top_k: int = DEFAULT_TOP_K,
) -> Searcher:
    """
    Factory function to create a searcher with common defaults.

    Args:
        index_dir: Base directory for indexes
        ticker: Company ticker
        mode: Default search mode
        top_k: Default number of results

    Returns:
        Configured Searcher instance
    """
    config = SearchConfig(mode=mode, top_k=top_k)
    return Searcher(index_dir=index_dir, ticker=ticker, config=config)
