"""RAG query module for answering questions with Claude."""

import os
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic

from .search import Searcher, SearchMode
from .store import SearchResult

# Default model for RAG queries
DEFAULT_RAG_MODEL = "claude-sonnet-4-5-20250929"

# System prompt for RAG queries
RAG_SYSTEM_PROMPT = """You are a financial analyst assistant helping with stock research.
You have access to SEC filings (10-K, 10-Q, 8-K, etc.) for the company being analyzed.

When answering questions:
1. Base your answers ONLY on the provided context from SEC filings
2. Cite specific sections and dates when referencing information
3. If the context doesn't contain enough information, say so clearly
4. Be precise with numbers and financial terms
5. Distinguish between facts from filings and your analysis/interpretation

Format your response with clear structure when appropriate."""


@dataclass
class RAGResponse:
    """Response from a RAG query."""

    answer: str
    sources: list[SearchResult]
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def source_citations(self) -> str:
        """Format sources as citations."""
        if not self.sources:
            return "No sources used."

        citations = []
        for i, src in enumerate(self.sources, 1):
            citations.append(f"[{i}] {src.filing_type} ({src.filing_date}) - {src.section}")
        return "\n".join(citations)


class RAGQueryEngine:
    """Query engine that combines search with Claude for answers."""

    def __init__(
        self,
        searcher: Searcher,
        api_key: str | None = None,
        model: str = DEFAULT_RAG_MODEL,
        max_context_tokens: int = 4000,
    ):
        """
        Initialize RAG query engine.

        Args:
            searcher: Configured Searcher instance
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use
            max_context_tokens: Max tokens for context window
        """
        self.searcher = searcher
        self.model = model
        self.max_context_tokens = max_context_tokens

        # Initialize Anthropic client
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or pass api_key parameter.")
        self.client = Anthropic(api_key=api_key)

    def query(
        self,
        question: str,
        search_mode: SearchMode = "hybrid",
        filing_type: str | list[str] | None = None,
        system_prompt: str | None = None,
        rerank: bool | None = None,
    ) -> RAGResponse:
        """
        Answer a question using RAG.

        Args:
            question: The question to answer
            search_mode: How to search for context
            filing_type: Filter to specific filing type (string or list of strings)
            system_prompt: Override default system prompt
            rerank: Whether to rerank search results for better precision

        Returns:
            RAGResponse with answer and sources
        """
        # Get relevant context
        sources, context = self.searcher.get_context_window(
            query=question,
            max_tokens=self.max_context_tokens,
            mode=search_mode,
            filing_type=filing_type,
            rerank=rerank,
        )

        if not context:
            return RAGResponse(
                answer="I couldn't find any relevant information in the indexed documents to answer this question.",
                sources=[],
                model=self.model,
                input_tokens=0,
                output_tokens=0,
            )

        # Build the prompt
        user_message = self._build_user_message(question, context)

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt or RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract text from response (first content block is always text for our prompts)
        answer: str = getattr(response.content[0], "text", "")

        return RAGResponse(
            answer=answer,
            sources=sources,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _build_user_message(self, question: str, context: str) -> str:
        """Build the user message with context and question."""
        return f"""Based on the following excerpts from SEC filings for {self.searcher.ticker}, please answer the question.

## Context from SEC Filings

{context}

## Question

{question}

Please provide a clear, well-structured answer based on the context above. Cite specific filings and sections when referencing information."""


def create_rag_engine(
    index_dir: Path,
    ticker: str,
    api_key: str | None = None,
    model: str = DEFAULT_RAG_MODEL,
    rerank: bool = False,
) -> RAGQueryEngine:
    """
    Factory function to create a RAG query engine.

    Args:
        index_dir: Base directory for indexes
        ticker: Company ticker
        api_key: Optional Anthropic API key
        model: Claude model to use
        rerank: Whether to enable reranking by default

    Returns:
        Configured RAGQueryEngine
    """
    from .search import SearchConfig, Searcher

    config = SearchConfig(rerank=rerank)
    searcher = Searcher(index_dir=index_dir, ticker=ticker, config=config)
    return RAGQueryEngine(searcher=searcher, api_key=api_key, model=model)
