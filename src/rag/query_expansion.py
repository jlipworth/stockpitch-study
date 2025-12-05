"""Query expansion for improved recall in financial document search.

Expands queries with:
- Acronym definitions (NRR -> net revenue retention)
- Synonym clusters (revenue -> sales, top line)
- Domain-specific financial terminology
"""

import re
from dataclasses import dataclass, field


@dataclass
class ExpandedQuery:
    """Result of query expansion."""

    original: str
    expansions: list[str] = field(default_factory=list)
    acronyms_found: list[tuple[str, str]] = field(default_factory=list)  # (acronym, expansion)
    synonyms_found: list[tuple[str, list[str]]] = field(default_factory=list)  # (term, synonyms)

    @property
    def all_terms(self) -> list[str]:
        """All query variations including original."""
        return [self.original] + self.expansions

    @property
    def fts_query(self) -> str:
        """Build FTS query with OR clauses for expansions."""
        if not self.expansions:
            return self.original

        # Quote multi-word phrases
        terms = [self.original]
        for exp in self.expansions:
            if " " in exp:
                terms.append(f'"{exp}"')
            else:
                terms.append(exp)

        return " OR ".join(terms)


class QueryExpander:
    """Expand queries with financial domain knowledge.

    Provides bidirectional acronym expansion and synonym substitution
    to improve recall for financial document search.
    """

    # ==========================================================================
    # FINANCIAL ACRONYMS - Bidirectional expansion
    # ==========================================================================
    ACRONYMS: dict[str, str] = {
        # Revenue & Growth Metrics
        "ARR": "annual recurring revenue",
        "MRR": "monthly recurring revenue",
        "NRR": "net revenue retention",
        "GRR": "gross revenue retention",
        "NDR": "net dollar retention",
        "TCV": "total contract value",
        "ACV": "annual contract value",
        "RPO": "remaining performance obligations",
        "CRPO": "current remaining performance obligations",
        "DBNER": "dollar-based net expansion rate",
        "GMV": "gross merchandise value",
        "GTV": "gross transaction value",
        "TPV": "total payment volume",
        # Profitability Metrics
        "EBITDA": "earnings before interest taxes depreciation and amortization",
        "EBIT": "earnings before interest and taxes",
        "EBT": "earnings before taxes",
        "EPS": "earnings per share",
        "GAAP": "generally accepted accounting principles",
        "NOI": "net operating income",
        "FFO": "funds from operations",
        "AFFO": "adjusted funds from operations",
        "NIM": "net interest margin",
        "NII": "net interest income",
        "ROE": "return on equity",
        "ROA": "return on assets",
        "ROIC": "return on invested capital",
        "ROI": "return on investment",
        "ROCE": "return on capital employed",
        # Cash Flow & Capital
        "FCF": "free cash flow",
        "OCF": "operating cash flow",
        "CFO": "cash flow from operations",
        "CAPEX": "capital expenditures",
        "OPEX": "operating expenditures",
        "D&A": "depreciation and amortization",
        "PP&E": "property plant and equipment",
        "WC": "working capital",
        "NWC": "net working capital",
        # Valuation Multiples
        "EV": "enterprise value",
        "P/E": "price to earnings",
        "PE": "price to earnings",
        "P/S": "price to sales",
        "PS": "price to sales",
        "P/B": "price to book",
        "PB": "price to book",
        "PEG": "price earnings to growth",
        "NAV": "net asset value",
        "DCF": "discounted cash flow",
        "WACC": "weighted average cost of capital",
        "IRR": "internal rate of return",
        "NPV": "net present value",
        "LTV": "lifetime value",
        "CAC": "customer acquisition cost",
        # SaaS & Subscription Metrics
        "ARPU": "average revenue per user",
        "ARPA": "average revenue per account",
        "ARPPU": "average revenue per paying user",
        "DAU": "daily active users",
        "MAU": "monthly active users",
        "WAU": "weekly active users",
        "MoM": "month over month",
        "QoQ": "quarter over quarter",
        "YoY": "year over year",
        "YTD": "year to date",
        "TTM": "trailing twelve months",
        "LTM": "last twelve months",
        "NTM": "next twelve months",
        "CLTV": "customer lifetime value",
        "CLV": "customer lifetime value",
        # Cost & Efficiency
        "COGS": "cost of goods sold",
        "COS": "cost of services",
        "COR": "cost of revenue",
        "SG&A": "selling general and administrative",
        "SGA": "selling general and administrative",
        "R&D": "research and development",
        "S&M": "sales and marketing",
        "G&A": "general and administrative",
        "SBC": "stock-based compensation",
        "RSU": "restricted stock unit",
        "PSU": "performance stock unit",
        "ESPP": "employee stock purchase plan",
        # Leverage & Credit
        "DSCR": "debt service coverage ratio",
        "ICR": "interest coverage ratio",
        # Note: LTV defined above as "lifetime value" - context-dependent
        "CDS": "credit default swap",
        "HY": "high yield",
        "IG": "investment grade",
        "NPA": "non-performing assets",
        "NPL": "non-performing loans",
        "LGD": "loss given default",
        "PD": "probability of default",
        "EAD": "exposure at default",
        "ALLL": "allowance for loan and lease losses",
        "ACL": "allowance for credit losses",
        "CECL": "current expected credit losses",
        # Market & Trading
        "AUM": "assets under management",
        # Note: NAV defined above in valuation multiples
        "ADV": "average daily volume",
        "ATH": "all-time high",
        "ATL": "all-time low",
        "VWAP": "volume weighted average price",
        "IPO": "initial public offering",
        "SPO": "secondary public offering",
        "M&A": "mergers and acquisitions",
        "LBO": "leveraged buyout",
        "MBO": "management buyout",
        "SPAC": "special purpose acquisition company",
        # Regulatory & Compliance
        "SEC": "securities and exchange commission",
        "SOX": "sarbanes oxley",
        "FASB": "financial accounting standards board",
        "IFRS": "international financial reporting standards",
        "KYC": "know your customer",
        "AML": "anti-money laundering",
        "BSA": "bank secrecy act",
        "GDPR": "general data protection regulation",
        "PCI": "payment card industry",
        "SOC": "service organization control",
        # Industry-Specific (Tech/Telecom)
        "TAM": "total addressable market",
        "SAM": "serviceable addressable market",
        "SOM": "serviceable obtainable market",
        "API": "application programming interface",
        "SDK": "software development kit",
        "SLA": "service level agreement",
        "NPS": "net promoter score",
        "CSAT": "customer satisfaction",
        "CES": "customer effort score",
        "MQL": "marketing qualified lead",
        "SQL": "sales qualified lead",
        "PQL": "product qualified lead",
        "SMB": "small and medium business",
        "SME": "small and medium enterprise",
        "VSB": "very small business",
        "SOHO": "small office home office",
        "B2B": "business to business",
        "B2C": "business to consumer",
        "B2B2C": "business to business to consumer",
        "D2C": "direct to consumer",
        "DTC": "direct to consumer",
        "ARPC": "average revenue per customer",
        "ASP": "average selling price",
        # Note: ACV defined above as "annual contract value"
        "BPO": "business process outsourcing",
        "ITO": "information technology outsourcing",
        "MSP": "managed service provider",
        "ISV": "independent software vendor",
        "VAR": "value added reseller",
        "OEM": "original equipment manufacturer",
        "ODM": "original design manufacturer",
        "MNO": "mobile network operator",
        "MVNO": "mobile virtual network operator",
        # Telecom-Specific (ARPU defined above in SaaS section)
        "AMPU": "average margin per user",
        "CPGA": "cost per gross addition",
        "SAC": "subscriber acquisition cost",
        "CCPU": "cash cost per user",
        "MOU": "minutes of use",
        "DOU": "data usage per user",
        "VoLTE": "voice over LTE",
        "FWA": "fixed wireless access",
        "FTTH": "fiber to the home",
        "FTTP": "fiber to the premises",
        "HFC": "hybrid fiber coaxial",
        "RAN": "radio access network",
        "ORAN": "open radio access network",
        "vRAN": "virtualized radio access network",
        "PSTN": "public switched telephone network",
        "CPE": "customer premises equipment",
        "OSS": "operations support system",
        "BSS": "business support system",
    }

    # Build reverse mapping (expansion -> acronym)
    ACRONYM_REVERSE: dict[str, str] = {v.lower(): k for k, v in ACRONYMS.items()}

    # ==========================================================================
    # SYNONYM CLUSTERS - Groups of interchangeable terms
    # ==========================================================================
    SYNONYMS: dict[str, list[str]] = {
        # Revenue terms
        "revenue": ["sales", "top line", "turnover", "income"],
        "sales": ["revenue", "top line", "turnover"],
        "top line": ["revenue", "sales"],
        # Profit terms
        "profit": ["earnings", "net income", "bottom line", "income"],
        "earnings": ["profit", "net income", "bottom line"],
        "net income": ["profit", "earnings", "bottom line"],
        "bottom line": ["profit", "earnings", "net income"],
        "income": ["profit", "earnings", "revenue"],
        # Margin terms
        "margin": ["profitability", "spread", "markup"],
        "gross margin": ["gross profit margin", "GPM"],
        "operating margin": ["operating profit margin", "EBIT margin"],
        "net margin": ["net profit margin", "profit margin"],
        # Growth terms
        "growth": ["expansion", "increase", "acceleration", "improvement"],
        "expansion": ["growth", "increase", "improvement"],
        "increase": ["growth", "rise", "gain", "improvement"],
        "decrease": ["decline", "reduction", "drop", "contraction"],
        "decline": ["decrease", "reduction", "drop", "deterioration", "contraction"],
        # Customer terms
        "churn": ["attrition", "customer loss", "turnover"],
        "attrition": ["churn", "customer loss"],
        "retention": ["renewal", "loyalty"],
        "customer": ["client", "subscriber", "user", "account"],
        "subscriber": ["customer", "user", "member"],
        # Financial position terms
        "debt": ["liabilities", "borrowings", "obligations", "leverage"],
        "leverage": ["debt", "gearing", "borrowings"],
        "liabilities": ["debt", "obligations", "payables"],
        "assets": ["holdings", "investments", "property"],
        # Cash terms
        "cash": ["liquidity", "cash position", "cash balance"],
        "liquidity": ["cash", "cash position"],
        "cash flow": ["cash generation", "cash conversion"],
        # Guidance & Outlook
        "guidance": ["outlook", "forecast", "expectations", "projections"],
        "outlook": ["guidance", "forecast", "expectations"],
        "forecast": ["guidance", "outlook", "projections", "estimates"],
        # Backlog & Pipeline
        "backlog": ["pipeline", "order book", "bookings", "orders"],
        "pipeline": ["backlog", "order book", "funnel"],
        "bookings": ["backlog", "orders", "new business"],
        # Headcount & Personnel
        "headcount": ["employees", "staff", "workforce", "FTEs"],
        "employees": ["headcount", "staff", "workforce", "personnel"],
        "workforce": ["employees", "headcount", "staff"],
        # Cost terms
        "costs": ["expenses", "expenditures", "spending"],
        "expenses": ["costs", "expenditures", "spending"],
        "spending": ["costs", "expenses", "expenditures", "outlay"],
        # Competition
        "competition": ["competitors", "rivals", "peers"],
        "competitors": ["competition", "rivals", "peers", "competitive landscape"],
        "market share": ["share of market", "market position"],
        # Strategy terms
        "strategy": ["strategic plan", "approach", "roadmap"],
        "initiative": ["program", "project", "effort"],
        "transformation": ["restructuring", "reorganization", "turnaround"],
        # Risk terms
        "risk": ["exposure", "vulnerability", "threat"],
        "exposure": ["risk", "vulnerability"],
        "headwind": ["challenge", "pressure", "obstacle", "risk"],
        "tailwind": ["opportunity", "benefit", "catalyst", "driver"],
        # Deal terms
        "acquisition": ["purchase", "buyout", "takeover", "deal"],
        "merger": ["combination", "consolidation", "union"],
        "divestiture": ["sale", "disposal", "spinoff"],
        # Performance terms
        "performance": ["results", "execution", "achievement"],
        "outperform": ["beat", "exceed", "surpass"],
        "underperform": ["miss", "lag", "fall short"],
    }

    def __init__(self, enable_acronyms: bool = True, enable_synonyms: bool = True):
        """Initialize query expander.

        Args:
            enable_acronyms: Enable acronym expansion
            enable_synonyms: Enable synonym expansion
        """
        self.enable_acronyms = enable_acronyms
        self.enable_synonyms = enable_synonyms

    def expand(self, query: str) -> ExpandedQuery:
        """Expand query with acronyms and synonyms.

        Args:
            query: Original search query

        Returns:
            ExpandedQuery with original and expanded terms
        """
        result = ExpandedQuery(original=query)
        query_lower = query.lower()
        words = re.findall(r"\b[\w&/-]+\b", query_lower)

        seen_expansions: set[str] = set()

        if self.enable_acronyms:
            self._expand_acronyms(query, query_lower, words, result, seen_expansions)

        if self.enable_synonyms:
            self._expand_synonyms(query_lower, words, result, seen_expansions)

        return result

    def _expand_acronyms(
        self,
        query: str,
        query_lower: str,
        words: list[str],
        result: ExpandedQuery,
        seen: set[str],
    ) -> None:
        """Add acronym expansions to result."""
        for word in words:
            word_upper = word.upper()

            # Check if word is an acronym
            if word_upper in self.ACRONYMS:
                expansion = self.ACRONYMS[word_upper]
                if expansion.lower() not in seen and expansion.lower() not in query_lower:
                    result.acronyms_found.append((word_upper, expansion))
                    result.expansions.append(expansion)
                    seen.add(expansion.lower())

            # Check if query contains the full expansion (add acronym)
            if word in self.ACRONYM_REVERSE:
                acronym = self.ACRONYM_REVERSE[word]
                if acronym.lower() not in seen and acronym.lower() not in query_lower:
                    result.acronyms_found.append((acronym, word))
                    result.expansions.append(acronym)
                    seen.add(acronym.lower())

        # Also check for multi-word expansions in the query
        for expansion, acronym in self.ACRONYM_REVERSE.items():
            if expansion in query_lower and acronym.lower() not in seen:
                result.acronyms_found.append((acronym, expansion))
                result.expansions.append(acronym)
                seen.add(acronym.lower())

    def _expand_synonyms(
        self,
        query_lower: str,
        words: list[str],
        result: ExpandedQuery,
        seen: set[str],
    ) -> None:
        """Add synonym expansions to result."""
        # Check single words
        for word in words:
            if word in self.SYNONYMS:
                synonyms_to_add = []
                for syn in self.SYNONYMS[word]:
                    if syn.lower() not in seen and syn.lower() not in query_lower:
                        synonyms_to_add.append(syn)
                        result.expansions.append(syn)
                        seen.add(syn.lower())
                if synonyms_to_add:
                    result.synonyms_found.append((word, synonyms_to_add))

        # Check multi-word phrases
        for phrase, synonyms in self.SYNONYMS.items():
            if " " in phrase and phrase in query_lower:
                synonyms_to_add = []
                for syn in synonyms:
                    if syn.lower() not in seen and syn.lower() not in query_lower:
                        synonyms_to_add.append(syn)
                        result.expansions.append(syn)
                        seen.add(syn.lower())
                if synonyms_to_add:
                    result.synonyms_found.append((phrase, synonyms_to_add))

    def get_acronym_definition(self, acronym: str) -> str | None:
        """Get definition for an acronym.

        Args:
            acronym: Acronym to look up (case-insensitive)

        Returns:
            Definition string or None if not found
        """
        return self.ACRONYMS.get(acronym.upper())

    def get_synonyms(self, term: str) -> list[str]:
        """Get synonyms for a term.

        Args:
            term: Term to look up (case-insensitive)

        Returns:
            List of synonyms or empty list if not found
        """
        return self.SYNONYMS.get(term.lower(), [])


# Module-level instance for convenience
_default_expander: QueryExpander | None = None


def get_expander() -> QueryExpander:
    """Get default QueryExpander instance (lazy initialization)."""
    global _default_expander
    if _default_expander is None:
        _default_expander = QueryExpander()
    return _default_expander


def expand_query(query: str) -> ExpandedQuery:
    """Convenience function to expand a query using default expander."""
    return get_expander().expand(query)
