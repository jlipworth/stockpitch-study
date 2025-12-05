"""Query classification for dynamic hybrid search weighting.

Quick Win #3: Classifies queries and returns optimal vector/FTS weights.

Query types and their optimal weights:
- exact_metric: High FTS weight (0.3v/0.7f) - "Q2 2024 revenue", "net income FY25"
- conceptual: High vector weight (0.8v/0.2f) - "What are the main risks?"
- entity: High FTS weight (0.2v/0.8f) - "Deutsche Telekom", "Rick McConnell"
- trend: Balanced (0.6v/0.4f) - "revenue growth trends", "margin evolution"
- comparison: Higher vector (0.7v/0.3f) - "compare 2023 vs 2024"
- default: Standard hybrid (0.7v/0.3f)
"""

import re
from dataclasses import dataclass
from typing import Literal

QueryType = Literal["exact_metric", "conceptual", "entity", "trend", "comparison", "default"]


@dataclass
class QueryClassification:
    """Result of query classification."""

    query_type: QueryType
    vector_weight: float
    fts_weight: float
    confidence: float  # 0.0 to 1.0


# Weight configurations for each query type
QUERY_WEIGHTS: dict[QueryType, tuple[float, float]] = {
    "exact_metric": (0.3, 0.7),  # High FTS for exact lookups
    "conceptual": (0.8, 0.2),  # High vector for semantic understanding
    "entity": (0.2, 0.8),  # High FTS for named entities
    "trend": (0.6, 0.4),  # Balanced for trend analysis
    "comparison": (0.7, 0.3),  # Higher vector for semantic comparison
    "default": (0.7, 0.3),  # Standard hybrid
}

# Patterns for query classification
EXACT_METRIC_PATTERNS = [
    r"\b(Q[1-4]|FY)\s*\d{2,4}\b",  # Q1 2024, FY25, Q3'24
    r"\b\d{4}\s+(revenue|income|margin|EBITDA|ARR|NRR)\b",
    r"\b(revenue|income|margin|EBITDA|ARR|NRR|EPS)\s+\d{4}\b",
    r"\b(what was|what is|how much)\s+.*(revenue|income|margin|EBITDA)\b",
    r"\b(total|net|gross)\s+(revenue|income|margin)\b",
    r"\$\d+",  # Dollar amounts
]

CONCEPTUAL_PATTERNS = [
    r"\b(what are|explain|describe|why|how does|tell me about)\b",
    r"\b(main|key|primary|significant)\s+(risks?|factors?|drivers?|trends?|concerns?)\b",
    r"\b(strategy|approach|plan|outlook|guidance)\b",
    r"\b(impact|effect|implication|consequence)\b",
    r"\?$",  # Questions often conceptual
]

ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+\s+[A-Z][a-zA-Z]+\b",  # Person names (John Smith, Rick McConnell)
    # Only specific company/ticker acronyms, not all caps words
    r"\b(AWS|GCP|IBM|DT|DDOG|NEWR|ESTC|AI)\b",  # Known tickers/companies
    r"\b(Deutsche|Telekom|Akamai|Datadog|Splunk|Elastic|Dynatrace)\b",
    r"\b(CEO|CFO|CTO|CRO|CMO|COO)\s",  # Executive titles followed by space
    r"\b(board of directors|executive officer)\b",
]

TREND_PATTERNS = [
    r"\b(trend|growth|change|evolution|trajectory)\b",
    r"\b(over time|year over year|YoY|quarter over quarter|QoQ)\b",
    r"\b(increase|decrease|improve|decline|deteriorate)\b",
    r"\b(historical|historically)\b",
    r"\b(last|past|recent)\s+\d+\s+(year|quarter|month)s?\b",
]

COMPARISON_PATTERNS = [
    r"\b(compare|comparison|vs|versus|differ|contrast)\b",
    r"\b(better|worse|stronger|weaker)\s+than\b",
    r"\b\d{4}\s+(vs|versus|and|compared to)\s+\d{4}\b",
    r"\b(relative to|against|benchmar)\b",
]


def classify_query(query: str) -> QueryClassification:
    """
    Classify a query to determine optimal search weights.

    Args:
        query: The search query

    Returns:
        QueryClassification with type and weights
    """
    query_lower = query.lower()
    scores: dict[QueryType, float] = {
        "exact_metric": 0.0,
        "conceptual": 0.0,
        "entity": 0.0,
        "trend": 0.0,
        "comparison": 0.0,
    }

    # Score each pattern type
    for pattern in EXACT_METRIC_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            scores["exact_metric"] += 1.0

    for pattern in CONCEPTUAL_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            scores["conceptual"] += 1.0

    for pattern in ENTITY_PATTERNS:
        # Don't use IGNORECASE for entity patterns - they rely on capitalization
        if re.search(pattern, query):
            scores["entity"] += 1.0

    for pattern in TREND_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            scores["trend"] += 1.0

    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            scores["comparison"] += 1.0

    # Find highest scoring type
    max_score = max(scores.values())

    if max_score == 0:
        query_type: QueryType = "default"
        confidence = 0.5  # Low confidence for default
    else:
        # Get the type(s) with max score
        top_types = [t for t, s in scores.items() if s == max_score]
        # Prefer certain types in tie-breakers
        priority = ["exact_metric", "entity", "comparison", "trend", "conceptual"]
        for t in priority:
            if t in top_types:
                query_type = t  # type: ignore
                break
        else:
            query_type = top_types[0]  # type: ignore

        # Confidence based on score and separation from second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1] > 0:
            confidence = min(0.9, 0.5 + (max_score - sorted_scores[1]) * 0.2)
        else:
            confidence = min(0.9, 0.5 + max_score * 0.15)

    weights = QUERY_WEIGHTS[query_type]
    return QueryClassification(
        query_type=query_type,
        vector_weight=weights[0],
        fts_weight=weights[1],
        confidence=confidence,
    )


def get_weights_for_query(query: str) -> tuple[float, float]:
    """
    Convenience function to get just the weights for a query.

    Args:
        query: The search query

    Returns:
        Tuple of (vector_weight, fts_weight)
    """
    classification = classify_query(query)
    return classification.vector_weight, classification.fts_weight
