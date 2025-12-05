"""Search feedback logging for continuous improvement.

Quick Win #4: Log search queries, results, and context for analysis.

Logs are written to JSONL format for easy analysis:
- Enables identifying common failing queries
- Helps tune hybrid search weights
- Finds missing synonyms/acronyms
- Provides training data for future improvements
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default feedback log location (relative to index directory)
DEFAULT_FEEDBACK_FILE = "search_feedback.jsonl"


@dataclass
class SearchFeedbackEntry:
    """A single search feedback entry."""

    timestamp: str
    ticker: str
    query: str
    mode: str
    top_k: int
    vector_weight: float
    query_type: str  # From query classifier
    query_confidence: float
    num_results: int
    reranked: bool
    filing_type_filter: str | None
    section_filter: str | None
    # Top result summaries (for analysis without storing full text)
    top_results: list[dict[str, Any]]  # [{score, filing_type, section, preview}]
    # Query expansion fields
    expanded_query: str | None = None  # The expanded FTS query if expansion was used
    expansion_count: int = 0  # Number of expansions added
    # Optional user feedback
    user_rating: int | None = None  # 1-5 scale
    user_selected_indices: list[int] | None = None  # Which results were useful


class FeedbackLogger:
    """Logger for search feedback data."""

    def __init__(self, index_dir: Path, ticker: str):
        """
        Initialize feedback logger.

        Args:
            index_dir: Base directory for indexes
            ticker: Company ticker
        """
        self.ticker = ticker.upper()
        self.log_path = index_dir / ticker.upper() / DEFAULT_FEEDBACK_FILE

    def log_search(
        self,
        query: str,
        mode: str,
        top_k: int,
        vector_weight: float,
        query_type: str,
        query_confidence: float,
        results: list[Any],  # list[SearchResult]
        reranked: bool = False,
        filing_type_filter: str | None = None,
        section_filter: str | None = None,
        expanded_query: str | None = None,
        expansion_count: int = 0,
    ) -> None:
        """
        Log a search query and its results.

        Args:
            query: The search query
            mode: Search mode (vector, fts, hybrid)
            top_k: Number of results requested
            vector_weight: Weight used for hybrid search
            query_type: Classified query type
            query_confidence: Classification confidence
            results: List of SearchResult objects
            reranked: Whether results were reranked
            filing_type_filter: Filing type filter used
            section_filter: Section filter used
            expanded_query: The expanded FTS query if expansion was used
            expansion_count: Number of expansions added
        """
        # Create summaries of top results (don't log full text)
        top_results = []
        for r in results[:10]:  # Log top 10 max
            preview = r.text[:200] + "..." if len(r.text) > 200 else r.text
            top_results.append(
                {
                    "score": round(r.score, 4),
                    "filing_type": r.metadata.get("filing_type", ""),
                    "filing_date": r.metadata.get("filing_date", ""),
                    "section": r.metadata.get("section", ""),
                    "preview": preview,
                }
            )

        entry = SearchFeedbackEntry(
            timestamp=datetime.now().isoformat(),
            ticker=self.ticker,
            query=query,
            mode=mode,
            top_k=top_k,
            vector_weight=vector_weight,
            query_type=query_type,
            query_confidence=query_confidence,
            num_results=len(results),
            reranked=reranked,
            filing_type_filter=filing_type_filter,
            section_filter=section_filter,
            top_results=top_results,
            expanded_query=expanded_query,
            expansion_count=expansion_count,
        )

        self._write_entry(entry)

    def _write_entry(self, entry: SearchFeedbackEntry) -> None:
        """Append entry to JSONL log file."""
        try:
            # Ensure directory exists
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")

        except Exception as e:
            logger.warning(f"Failed to write search feedback: {e}")

    def log_user_feedback(
        self,
        query: str,
        rating: int,
        selected_indices: list[int] | None = None,
    ) -> None:
        """
        Log user feedback for a query (to be called after search).

        Args:
            query: The original query
            rating: User rating (1-5)
            selected_indices: Which result indices were useful
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "ticker": self.ticker,
            "type": "user_feedback",
            "query": query,
            "user_rating": rating,
            "user_selected_indices": selected_indices,
        }

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write user feedback: {e}")


def analyze_feedback(log_path: Path) -> dict[str, Any]:
    """
    Analyze feedback log to identify patterns.

    Args:
        log_path: Path to feedback JSONL file

    Returns:
        Dict with analysis results
    """
    if not log_path.exists():
        return {"error": "No feedback log found"}

    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Filter to search entries (not user feedback)
    search_entries = [e for e in entries if "mode" in e]

    if not search_entries:
        return {"error": "No search entries found"}

    # Analysis
    query_types = {}
    modes = {}
    low_result_queries = []  # Queries with few or no results
    avg_results_by_type = {}

    for entry in search_entries:
        # Count query types
        qt = entry.get("query_type", "unknown")
        query_types[qt] = query_types.get(qt, 0) + 1

        # Count modes
        mode = entry.get("mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1

        # Track low-result queries
        if entry.get("num_results", 0) < 3:
            low_result_queries.append(
                {
                    "query": entry.get("query"),
                    "num_results": entry.get("num_results"),
                    "query_type": qt,
                }
            )

        # Average results by query type
        if qt not in avg_results_by_type:
            avg_results_by_type[qt] = []
        avg_results_by_type[qt].append(entry.get("num_results", 0))

    # Calculate averages
    for qt, counts in avg_results_by_type.items():
        avg_results_by_type[qt] = round(sum(counts) / len(counts), 1)

    return {
        "total_searches": len(search_entries),
        "query_type_distribution": query_types,
        "mode_distribution": modes,
        "avg_results_by_query_type": avg_results_by_type,
        "low_result_queries": low_result_queries[:20],  # Top 20
    }
