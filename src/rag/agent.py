"""Agentic RAG with query decomposition and multi-hop retrieval."""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from anthropic import Anthropic

from .search import SearchConfig, Searcher
from .store import SearchResult

logger = logging.getLogger(__name__)


class Depth(str, Enum):
    """Search depth levels."""

    FAST = "fast"
    MEDIUM = "medium"
    DEEP = "deep"


@dataclass
class StructuredQuery:
    """A decomposed sub-query with optional filters."""

    query: str
    year: int | None = None
    quarter: str | None = None  # Q1, Q2, Q3, Q4
    filing_type: str | None = None
    rationale: str = ""

    def get_date_filter(self) -> tuple[str | None, str | None]:
        """
        Get date range filter based on year/quarter.

        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format, or (None, None)
        """
        if not self.year:
            return None, None

        if self.quarter:
            # Map quarters to date ranges (calendar year)
            quarter_ranges = {
                "Q1": (f"{self.year}-01-01", f"{self.year}-03-31"),
                "Q2": (f"{self.year}-04-01", f"{self.year}-06-30"),
                "Q3": (f"{self.year}-07-01", f"{self.year}-09-30"),
                "Q4": (f"{self.year}-10-01", f"{self.year}-12-31"),
            }
            return quarter_ranges.get(self.quarter.upper(), (None, None))

        # Full year
        return f"{self.year}-01-01", f"{self.year}-12-31"


# Depth configurations
DEPTH_CONFIG = {
    Depth.FAST: {
        "sub_queries": 0,  # Direct search, no decomposition
        "max_hops": 1,
        "max_searches": 2,
        "file_reads": 0,
        "file_read_lines": 0,
    },
    Depth.MEDIUM: {
        "sub_queries": 3,
        "max_hops": 2,
        "max_searches": 6,
        "file_reads": 2,
        "file_read_lines": 50,
    },
    Depth.DEEP: {
        "sub_queries": 5,
        "max_hops": 3,
        "max_searches": 12,
        "file_reads": 4,
        "file_read_lines": 75,
    },
}


@dataclass
class HopResult:
    """Result from a single search hop."""

    query: str
    results: list[SearchResult]
    file_contents: list[dict] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from the RAG agent."""

    answer: str
    sources: list[SearchResult]
    confidence: str  # high, medium, low
    hops: list[HopResult]
    sub_queries: list[str]
    model: str
    total_searches: int
    total_file_reads: int
    input_tokens: int
    output_tokens: int

    @property
    def source_citations(self) -> str:
        """Format sources as citations."""
        if not self.sources:
            return "No sources used."

        seen = set()
        citations = []
        for src in self.sources:
            key = (src.filing_type, src.filing_date, src.section)
            if key not in seen:
                seen.add(key)
                citations.append(f"- {src.filing_type} ({src.filing_date}) {src.section}")
        return "\n".join(citations)


# Patterns for detecting temporal references
YEAR_PATTERN = re.compile(r"\b(20[1-2][0-9])\b")
QUARTER_PATTERN = re.compile(r"\b(Q[1-4])\s*(20[1-2][0-9])?\b", re.IGNORECASE)
FISCAL_YEAR_PATTERN = re.compile(r"\bFY\s*'?(20)?([1-2][0-9])\b", re.IGNORECASE)

# Patterns for detecting comparative queries
COMPARATIVE_PATTERNS = [
    re.compile(r"\b(?:vs\.?|versus|compared?\s+to|comparison)\b", re.IGNORECASE),
    re.compile(r"\b(?:change[ds]?|evolution|evolved?|trend|differ(?:ence)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:from|between)\s+\d{4}\s+(?:to|and)\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b(?:year[- ]over[- ]year|yoy|quarter[- ]over[- ]quarter|qoq)\b", re.IGNORECASE),
]


def detect_temporal_refs(question: str) -> list[dict]:
    """
    Detect year and quarter references in a question.

    Returns:
        List of dicts with 'year' and optionally 'quarter' keys
    """
    refs = []

    # Find fiscal year references (FY24, FY2024, FY'24)
    for match in FISCAL_YEAR_PATTERN.finditer(question):
        year_suffix = match.group(2)
        year = int(f"20{year_suffix}")
        refs.append({"year": year, "type": "fiscal"})

    # Find explicit years (2023, 2024)
    for match in YEAR_PATTERN.finditer(question):
        year = int(match.group(1))
        # Skip if already captured as fiscal year
        if not any(r["year"] == year for r in refs):
            refs.append({"year": year, "type": "calendar"})

    # Find quarter references (Q1 2024, Q2)
    for match in QUARTER_PATTERN.finditer(question):
        quarter = match.group(1).upper()
        year_str = match.group(2)
        if year_str:
            year = int(year_str)
            refs.append({"year": year, "quarter": quarter, "type": "quarter"})
        elif refs:
            # Attach to most recent year reference
            refs[-1]["quarter"] = quarter

    return refs


def is_comparative_query(question: str) -> bool:
    """Check if the question is asking for a comparison."""
    return any(pattern.search(question) for pattern in COMPARATIVE_PATTERNS)


def extract_comparison_periods(question: str) -> list[dict]:
    """
    Extract periods being compared in a comparative query.

    Returns:
        List of period dicts, typically 2 for "X vs Y" comparisons
    """
    refs = detect_temporal_refs(question)

    if len(refs) >= 2:
        # Multiple explicit periods - use them
        return refs

    if len(refs) == 1:
        # Single period - infer the other (usually previous year)
        ref = refs[0]
        if ref["type"] in ("fiscal", "calendar"):
            return [
                {"year": ref["year"] - 1, "type": ref["type"]},
                ref,
            ]

    return refs


DECOMPOSITION_PROMPT = """You are a financial research assistant. Given a user's question about a company, generate search queries that will help find relevant information in SEC filings, earnings transcripts, and analyst reports.

**Company**: {ticker}
**Fiscal Calendar**: {fiscal_info}

**User Question**: {question}

Generate {num_queries} search queries that would help answer this question. Consider:
1. Different phrasings of the core question
2. Related sub-questions that provide context
3. Specific terms/metrics that might appear in filings

Return ONLY a JSON array of strings, no explanation:
["query 1", "query 2", "query 3"]"""


# Enhanced prompt for temporal/comparative queries
STRUCTURED_DECOMPOSITION_PROMPT = """You are a financial research assistant. Given a user's question, generate structured search queries with appropriate filters.

**Company**: {ticker}
**Fiscal Calendar**: {fiscal_info}
**Today's Date**: {today}

**User Question**: {question}

**Detected Context**:
- Temporal references: {temporal_refs}
- Is comparative query: {is_comparative}

Generate {num_queries} search queries. For each, specify:
- query: The search terms (concise, focused)
- year: Filter to specific year if relevant (null if not)
- quarter: Filter to Q1/Q2/Q3/Q4 if relevant (null if not)
- filing_type: "10-K", "10-Q", "transcript", "analyst" if specific type needed (null for all)
- rationale: Brief explanation of what this search targets

**Important**:
- For comparative queries, create separate searches for each time period
- Use year filters to ensure results from the correct period
- Keep queries focused - don't combine multiple years in one query

Return ONLY a JSON array, no explanation:
[
  {{"query": "search terms", "year": 2024, "quarter": null, "filing_type": null, "rationale": "why"}},
  {{"query": "search terms", "year": 2023, "quarter": null, "filing_type": "10-K", "rationale": "why"}}
]"""


ANSWER_PROMPT = """You are a financial analyst researching {ticker}.

**Fiscal Calendar**: {fiscal_info}

**Question**: {question}

**Retrieved Context (USE THIS DIRECTLY)**:
{context}

## Critical Rules

1. **Answer ONLY from the provided context** - Do not use outside knowledge
2. **Quote directly from snippets** - Use exact text from the context above
3. **Be specific with citations** - Include doc type, date/fiscal period, and section
4. **Say "not found" if missing** - Don't fabricate or guess
5. **Prioritize recent documents** - FY2025/FY2026 over older filings

## Citation Format (BE SPECIFIC)

Good citations:
- 10-K FY2025 (filed 2025-05-22) Item 7: "exact quote from context"
- Q2 FY2026 Earnings Call, Nov 05, 2025, CEO remarks: "exact quote"
- Goldman Sachs Conference, Dec 02, 2025: "exact quote"
- DEF 14A 2025-07-08 Executive Compensation: "exact quote"

Bad citations (too vague - DO NOT USE):
- 10-K: "quote" ← Missing which year!
- Transcript: "quote" ← Missing which transcript!
- Conference: "quote" ← Missing which conference!

## Response Format

### Answer

[Direct answer synthesizing the context. Include key facts, metrics, and quotes.]

### Sources

- [doc_type] [date/fiscal period] [section]: "relevant quote"
- [doc_type] [date/fiscal period] [section]: "relevant quote"

### Confidence

[high/medium/low] - [brief reason]

### Missing Information

[What's not covered in the context, if anything. Say "None" if answer is complete.]"""


FOLLOWUP_PROMPT = """Based on your previous answer attempt, you indicated low confidence or missing information.

**Original Question**: {question}
**What you found**: {summary}
**What's missing**: {missing}

Generate {num_queries} follow-up search queries to find the missing information.

Return ONLY a JSON array of strings:
["query 1", "query 2"]"""


class RAGAgent:
    """Agentic RAG with query decomposition and multi-hop retrieval."""

    def __init__(
        self,
        searcher: Searcher,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        fiscal_info: str = "",
    ):
        """
        Initialize RAG agent.

        Args:
            searcher: Configured Searcher instance
            api_key: Anthropic API key
            model: Claude model to use
            fiscal_info: Company fiscal calendar info (e.g., "FY ends March 31")
        """
        self.searcher = searcher
        self.model = model
        self.fiscal_info = fiscal_info or "Standard calendar year"

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self.client = Anthropic(api_key=api_key)

        # Track token usage
        self._input_tokens = 0
        self._output_tokens = 0

    def ask(
        self,
        question: str,
        depth: Depth = Depth.FAST,
        doc_type: str | list[str] | None = None,
        rerank: bool = True,
    ) -> AgentResponse:
        """
        Answer a question using agentic RAG.

        Args:
            question: The question to answer
            depth: Search depth (fast/medium/deep)
            doc_type: Filter to specific document types
            rerank: Whether to rerank search results

        Returns:
            AgentResponse with answer, sources, and metadata
        """
        config = DEPTH_CONFIG[depth]
        self._input_tokens = 0
        self._output_tokens = 0

        all_results: list[SearchResult] = []
        all_hops: list[HopResult] = []
        sub_queries: list[str] = []
        total_file_reads = 0

        # Step 1: Query decomposition (skip for fast)
        structured_queries: list[StructuredQuery] = []
        if config["sub_queries"] > 0:
            decomposed = self._decompose_query(question, config["sub_queries"])

            # Handle both structured and simple decomposition
            if decomposed and isinstance(decomposed[0], StructuredQuery):
                structured_queries = decomposed  # type: ignore[assignment]
                sub_queries = [sq.query for sq in structured_queries]
                # For structured queries, don't add the original question
                # (it's already decomposed into filtered sub-queries)
            else:
                sub_queries = decomposed  # type: ignore[assignment]
                queries_to_run_simple = [question] + sub_queries
        else:
            queries_to_run_simple = [question]

        # Step 2: Initial search hop
        total_searches = self._execute_initial_searches(
            structured_queries=structured_queries,
            queries_to_run_simple=queries_to_run_simple if not structured_queries else [],
            doc_type=doc_type,
            rerank=rerank,
            max_searches=config["max_searches"],
            all_results=all_results,
            all_hops=all_hops,
        )

        # Dedupe results by content hash (source + section + chunk_index)
        seen_chunks: set[str] = set()
        unique_results = self._dedupe_results(all_results, seen_chunks)

        # Step 3: Optional file reads (medium/deep)
        file_contents = []
        if config["file_reads"] > 0 and unique_results:
            file_contents = self._read_source_files(
                unique_results[: config["file_reads"]],
                max_lines=config["file_read_lines"],
            )
            total_file_reads = len(file_contents)

        # Step 4: Generate answer
        context = self._build_context(unique_results, file_contents)
        answer, confidence, missing = self._generate_answer(question, context)

        # Step 5: Multi-hop if needed (medium/deep only)
        hop_count = 1
        while (
            confidence == "low"
            and missing
            and hop_count < config["max_hops"]
            and total_searches < config["max_searches"]
        ):
            hop_count += 1

            # Generate follow-up queries
            followup_queries = self._generate_followup(
                question,
                answer,
                missing,
                min(2, config["max_searches"] - total_searches),
            )

            # Search with follow-ups
            for query in followup_queries:
                if total_searches >= config["max_searches"]:
                    break

                results = self.searcher.search(
                    query=query,
                    top_k=5,
                    mode="hybrid",
                    filing_type=doc_type,
                    rerank=rerank,
                )
                total_searches += 1

                # Add new results (dedupe against seen chunks)
                new_results = self._dedupe_results(results, seen_chunks)
                unique_results.extend(new_results)

                hop = HopResult(query=query, results=new_results)
                all_hops.append(hop)

            # Re-generate answer with expanded context
            context = self._build_context(unique_results, file_contents)
            answer, confidence, missing = self._generate_answer(question, context)

        return AgentResponse(
            answer=answer,
            sources=unique_results,
            confidence=confidence,
            hops=all_hops,
            sub_queries=sub_queries,
            model=self.model,
            total_searches=total_searches,
            total_file_reads=total_file_reads,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    def _dedupe_results(
        self,
        results: list[SearchResult],
        seen_chunks: set[str],
    ) -> list[SearchResult]:
        """Dedupe search results by content hash.

        Uses source + section + chunk_index as unique key. Updates seen_chunks
        in place and returns only new (unseen) results.
        """
        unique = []
        for r in results:
            chunk_key = f"{r.metadata.get('source', '')}:{r.section}:{r.metadata.get('chunk_index', 0)}"
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique.append(r)
        return unique

    def _execute_initial_searches(
        self,
        structured_queries: list[StructuredQuery],
        queries_to_run_simple: list[str],
        doc_type: str | list[str] | None,
        rerank: bool,
        max_searches: int,
        all_results: list[SearchResult],
        all_hops: list[HopResult],
    ) -> int:
        """Execute initial search queries (structured or simple).

        Mutates all_results and all_hops in place.
        Returns total number of searches performed.
        """
        total_searches = 0

        if structured_queries:
            # Structured decomposition with filters
            for sq in structured_queries[:max_searches]:
                start_date, end_date = sq.get_date_filter()
                effective_filing_type = sq.filing_type or doc_type

                results = self._search_with_date_filter(
                    query=sq.query,
                    start_date=start_date,
                    end_date=end_date,
                    filing_type=effective_filing_type,
                    rerank=rerank,
                )
                total_searches += 1
                all_results.extend(results)

                # Log the structured query details
                query_desc = sq.query
                if sq.year:
                    query_desc += f" [year={sq.year}]"
                if sq.quarter:
                    query_desc += f" [quarter={sq.quarter}]"

                all_hops.append(HopResult(query=query_desc, results=results))

                if total_searches >= max_searches:
                    break
        else:
            # Simple decomposition
            for query in queries_to_run_simple[:max_searches]:
                results = self.searcher.search(
                    query=query,
                    top_k=5,
                    mode="hybrid",
                    filing_type=doc_type,
                    rerank=rerank,
                )
                total_searches += 1
                all_results.extend(results)
                all_hops.append(HopResult(query=query, results=results))

                if total_searches >= max_searches:
                    break

        return total_searches

    def _search_with_date_filter(
        self,
        query: str,
        start_date: str | None,
        end_date: str | None,
        filing_type: str | list[str] | None = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """
        Search with date range filter applied.

        For now, we filter results post-hoc since LanceDB filtering by date
        requires the filing_date to be in a sortable format.

        Args:
            query: Search query
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
            filing_type: Filter by document type
            rerank: Whether to rerank results

        Returns:
            Filtered search results
        """
        # Fetch more results to allow for filtering
        fetch_k = 15 if (start_date or end_date) else 5

        results = self.searcher.search(
            query=query,
            top_k=fetch_k,
            mode="hybrid",
            filing_type=filing_type,
            rerank=rerank,
        )

        if not (start_date or end_date):
            return results[:5]

        # Filter by date range
        filtered = []
        for r in results:
            filing_date = r.filing_date
            if not filing_date:
                continue

            # Normalize date format (handle both YYYY-MM-DD and other formats)
            try:
                # Try to extract year from filing_date
                if len(filing_date) >= 4:
                    filing_year = int(filing_date[:4])

                    # Quick year check
                    if start_date and filing_year < int(start_date[:4]):
                        continue
                    if end_date and filing_year > int(end_date[:4]):
                        continue

                    # If years match, do full date comparison if possible
                    if len(filing_date) >= 10:
                        if start_date and filing_date < start_date:
                            continue
                        if end_date and filing_date > end_date:
                            continue

                filtered.append(r)
            except (ValueError, TypeError):
                # If date parsing fails, include the result
                filtered.append(r)

            if len(filtered) >= 5:
                break

        return filtered

    def _decompose_query(self, question: str, num_queries: int) -> list[str] | list[StructuredQuery]:
        """
        Use LLM to decompose question into sub-queries.

        For temporal/comparative queries, returns StructuredQuery objects with filters.
        For simple queries, returns plain strings (backward compatible).
        """
        # Detect if this needs structured decomposition
        temporal_refs = detect_temporal_refs(question)
        comparative = is_comparative_query(question)

        if temporal_refs or comparative:
            return self._decompose_structured(question, num_queries, temporal_refs, comparative)

        # Simple decomposition (original behavior)
        prompt = DECOMPOSITION_PROMPT.format(
            ticker=self.searcher.ticker,
            fiscal_info=self.fiscal_info,
            question=question,
            num_queries=num_queries,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens

        # Parse JSON array from response
        text = getattr(response.content[0], "text", "").strip()
        try:
            import json

            # Handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            # Fallback: return empty list
            return []

    def _decompose_structured(
        self,
        question: str,
        num_queries: int,
        temporal_refs: list[dict],
        is_comparative: bool,
    ) -> list[StructuredQuery]:
        """
        Decompose with structured output for temporal/comparative queries.

        Returns StructuredQuery objects with year/quarter/filing_type filters.
        """
        import json
        from datetime import date

        prompt = STRUCTURED_DECOMPOSITION_PROMPT.format(
            ticker=self.searcher.ticker,
            fiscal_info=self.fiscal_info,
            today=date.today().isoformat(),
            question=question,
            temporal_refs=json.dumps(temporal_refs),
            is_comparative=is_comparative,
            num_queries=num_queries,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens

        text = getattr(response.content[0], "text", "").strip()
        try:
            # Handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)

            # Convert to StructuredQuery objects
            queries = []
            for item in parsed:
                if isinstance(item, dict):
                    queries.append(
                        StructuredQuery(
                            query=item.get("query", ""),
                            year=item.get("year"),
                            quarter=item.get("quarter"),
                            filing_type=item.get("filing_type"),
                            rationale=item.get("rationale", ""),
                        )
                    )
                elif isinstance(item, str):
                    # Fallback for simple string
                    queries.append(StructuredQuery(query=item))

            return queries

        except (json.JSONDecodeError, IndexError):
            # Fallback: create simple queries from temporal refs
            if is_comparative and temporal_refs:
                periods = extract_comparison_periods(question)
                base_query = re.sub(r"\b(20[1-2][0-9]|FY\s*'?\d{2})\b", "", question).strip()
                return [
                    StructuredQuery(
                        query=base_query,
                        year=p.get("year"),
                        quarter=p.get("quarter"),
                        rationale=f"Period {p.get('year')}",
                    )
                    for p in periods
                ]
            return []

    def _read_source_files(
        self,
        results: list[SearchResult],
        max_lines: int,
    ) -> list[dict]:
        """Read source files for additional context."""
        file_contents = []
        seen_files = set()

        for result in results:
            # source_file may not be in metadata for all indexes
            source_file = result.metadata.get("source_file", "")
            if not source_file or source_file in seen_files:
                continue

            seen_files.add(source_file)
            file_path = Path(source_file)

            if not file_path.exists():
                continue

            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                # Try to find the relevant section around the chunk
                chunk_text = result.text[:100]  # First 100 chars for matching
                start_line = 0
                for i, line in enumerate(lines):
                    if chunk_text[:50] in line:
                        start_line = max(0, i - 5)
                        break

                excerpt = "".join(lines[start_line : start_line + max_lines])
                file_contents.append(
                    {
                        "file": str(file_path),
                        "filing_type": result.filing_type,
                        "filing_date": result.filing_date,
                        "excerpt": excerpt[:5000],  # Cap at 5k chars
                    }
                )
            except Exception as e:
                logger.warning("Failed to read file %s: %s", file_path, e)
                continue

        return file_contents

    def _build_context(
        self,
        results: list[SearchResult],
        file_contents: list[dict],
        use_parent_context: bool = True,
    ) -> str:
        """Build context string from search results and file contents.

        Args:
            results: Search results to include
            file_contents: Additional file excerpts
            use_parent_context: If True, expand results with parent section context
        """
        parts = []

        if use_parent_context and results:
            # Use parent-child retrieval for fuller context
            # This deduplicates by section, so multiple chunks from same section
            # only appear once with full section text
            expanded = self.searcher.expand_results_with_parent(
                results[:15],  # Cap at 15 results
                deduplicate_sections=True,
            )

            for i, (r, content, mode) in enumerate(expanded, 1):
                mode_label = "[full section]" if mode == "section" else "[excerpt]"
                # Limit content to ~4000 chars to avoid context explosion
                truncated = content[:4000] + "..." if len(content) > 4000 else content
                parts.append(f"[{i}] {r.filing_type} ({r.filing_date}) - {r.section} {mode_label}\n{truncated}")
        else:
            # Fallback to chunk-only mode
            for i, r in enumerate(results[:15], 1):
                parts.append(f"[{i}] {r.filing_type} ({r.filing_date}) - {r.section}\n{r.text[:1500]}")

        # Add file excerpts
        for fc in file_contents:
            parts.append(f"\n--- File: {fc['filing_type']} ({fc['filing_date']}) ---\n{fc['excerpt']}")

        return "\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> tuple[str, str, str | None]:
        """Generate answer from context, return (answer, confidence, missing_info)."""
        if not context.strip():
            return (
                "No relevant information found in the indexed documents.",
                "low",
                "No search results returned. Try different search terms.",
            )

        prompt = ANSWER_PROMPT.format(
            ticker=self.searcher.ticker,
            fiscal_info=self.fiscal_info,
            question=question,
            context=context,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens

        text = getattr(response.content[0], "text", "")

        # Parse confidence (new format: ### Confidence\n\nhigh/medium/low - reason)
        confidence = "medium"
        text_lower = text.lower()
        if "### confidence" in text_lower:
            conf_section = text.split("### Confidence")[-1].split("###")[0]
            if "high" in conf_section.lower()[:50]:
                confidence = "high"
            elif "low" in conf_section.lower()[:50]:
                confidence = "low"
        # Fallback to old format
        elif "**Confidence**: high" in text or "**Confidence**: High" in text:
            confidence = "high"
        elif "**Confidence**: low" in text or "**Confidence**: Low" in text:
            confidence = "low"

        # Parse missing information (new format: ### Missing Information)
        missing = None
        if "### Missing Information" in text:
            parts = text.split("### Missing Information")
            if len(parts) > 1:
                missing_text = parts[1].split("###")[0].strip()
                if missing_text and "none" not in missing_text.lower()[:30]:
                    missing = missing_text[:500]
        # Fallback to old format
        elif "**Missing Information**" in text:
            parts = text.split("**Missing Information**")
            if len(parts) > 1:
                missing_text = parts[1].strip()
                if missing_text and "none" not in missing_text.lower()[:20]:
                    missing = missing_text[:500]

        return text, confidence, missing

    def _generate_followup(
        self,
        question: str,
        summary: str,
        missing: str,
        num_queries: int,
    ) -> list[str]:
        """Generate follow-up search queries."""
        prompt = FOLLOWUP_PROMPT.format(
            question=question,
            summary=summary[:500],
            missing=missing,
            num_queries=num_queries,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens

        text = getattr(response.content[0], "text", "").strip()
        try:
            import json

            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return []


def create_agent(
    index_dir: Path,
    ticker: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    fiscal_info: str = "",
    rerank: bool = True,
) -> RAGAgent:
    """
    Factory function to create a RAG agent.

    Args:
        index_dir: Base directory for indexes
        ticker: Company ticker
        api_key: Optional Anthropic API key
        model: Claude model to use
        fiscal_info: Fiscal calendar info
        rerank: Whether to enable reranking

    Returns:
        Configured RAGAgent
    """
    config = SearchConfig(rerank=rerank)
    searcher = Searcher(index_dir=index_dir, ticker=ticker, config=config)
    return RAGAgent(
        searcher=searcher,
        api_key=api_key,
        model=model,
        fiscal_info=fiscal_info,
    )
