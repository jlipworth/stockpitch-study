"""Analyst report parser for PDF research reports from investment banks."""

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from .base_parser import BaseParser


@dataclass
class AnalystReportMetadata:
    """Metadata extracted from analyst report."""

    ticker: str = ""
    company_name: str = ""
    analyst_name: str = ""
    firm: str = ""  # Bank/firm name
    report_date: str = ""
    rating: str = ""  # Buy/Sell/Hold/Overweight etc.
    price_target: str = ""
    report_type: str = ""  # Initiation, Update, Industry, etc.


@dataclass
class ReportSection:
    """A section from an analyst report."""

    title: str
    content: str
    page_num: int


@dataclass
class ParsedAnalystReport:
    """Result of parsing an analyst report."""

    metadata: AnalystReportMetadata
    sections: list[ReportSection]
    full_text: str
    source_path: str


# Common section patterns in analyst reports
ANALYST_SECTION_PATTERNS = {
    # Investment thesis/summary
    "thesis": re.compile(
        r"(?:Investment\s+(?:Thesis|Summary|Highlights?)|"
        r"Key\s+(?:Thesis|Points|Takeaways)|"
        r"Executive\s+Summary|"
        r"Investment\s+Case|"
        r"Our\s+View)",
        re.IGNORECASE,
    ),
    # Valuation
    "valuation": re.compile(
        r"(?:Valuation|Price\s+Target\s+(?:Analysis|Derivation)|"
        r"Target\s+Price|DCF\s+Analysis|"
        r"Sum.of.the.Parts|SOTP|"
        r"Comparable\s+(?:Analysis|Companies))",
        re.IGNORECASE,
    ),
    # Financial estimates
    "estimates": re.compile(
        r"(?:(?:Financial\s+)?(?:Estimates?|Projections?|Forecasts?)|"
        r"Model\s+Update|Earnings\s+(?:Model|Estimates?)|"
        r"Revenue\s+(?:Model|Breakdown)|"
        r"Quarterly\s+(?:Model|Estimates?))",
        re.IGNORECASE,
    ),
    # Catalysts
    "catalysts": re.compile(
        r"(?:Catalysts?|Upcoming\s+Events?|"
        r"Key\s+(?:Dates?|Events?)|"
        r"Near.?Term\s+(?:Catalysts?|Drivers?)|"
        r"Potential\s+(?:Catalysts?|Upside))",
        re.IGNORECASE,
    ),
    # Risks
    "risks": re.compile(
        r"(?:Risks?\s*(?:to\s+(?:Our\s+)?(?:Thesis|View|Rating))?|"
        r"Key\s+Risks?|Downside\s+Risks?|"
        r"Investment\s+Risks?|"
        r"Risk\s+Factors?)",
        re.IGNORECASE,
    ),
    # Industry/Competitive analysis
    "industry": re.compile(
        r"(?:Industry\s+(?:Overview|Analysis|Outlook)|"
        r"Competitive\s+(?:Landscape|Position|Analysis)|"
        r"Market\s+(?:Overview|Position|Share)|"
        r"Sector\s+(?:Outlook|Analysis))",
        re.IGNORECASE,
    ),
    # Company overview
    "company": re.compile(
        r"(?:Company\s+(?:Overview|Description|Profile|Background)|"
        r"Business\s+(?:Description|Overview|Model)|"
        r"About\s+(?:the\s+)?Company)",
        re.IGNORECASE,
    ),
    # Management
    "management": re.compile(
        r"(?:Management\s+(?:Team|Discussion|Overview)|"
        r"Leadership|Senior\s+Management|"
        r"Key\s+(?:Personnel|Executives?))",
        re.IGNORECASE,
    ),
}

# Patterns to extract metadata from cover pages
RATING_PATTERNS = re.compile(
    r"\b(Buy|Sell|Hold|Neutral|Overweight|Underweight|"
    r"Outperform|Underperform|Market\s*Perform|Equal.?Weight|"
    r"Strong\s*Buy|Accumulate|Reduce)\b",
    re.IGNORECASE,
)

PRICE_TARGET_PATTERNS = re.compile(
    r"(?:Price\s*Target|Target\s*Price|PT|TP)[:\s]*\$?([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)

# Common investment bank names
BANK_PATTERNS = re.compile(
    r"\b(Goldman\s*Sachs|Morgan\s*Stanley|JPMorgan|J\.?P\.?\s*Morgan|"
    r"Bank\s*of\s*America|BofA|Merrill|Citi(?:group)?|"
    r"UBS|Credit\s*Suisse|Barclays|Deutsche\s*Bank|"
    r"Wells\s*Fargo|RBC|Raymond\s*James|"
    r"Jefferies|Piper\s*Sandler|Cowen|Wedbush|"
    r"Bernstein|Evercore|Wolfe|Stifel|BMO|BTIG|"
    r"Canaccord|Needham|Oppenheimer|KeyBanc|Baird)\b",
    re.IGNORECASE,
)


class AnalystReportParser(BaseParser["ParsedAnalystReport"]):
    """Parser for PDF analyst reports from investment banks."""

    def parse_file(self, pdf_path: Path, ticker: str = "") -> ParsedAnalystReport:
        """
        Parse an analyst report PDF.

        Args:
            pdf_path: Path to PDF file
            ticker: Stock ticker (optional, will try to extract)

        Returns:
            ParsedAnalystReport with extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Analyst report PDF not found: {pdf_path}")

        # Extract text from PDF
        full_text, page_texts = self._extract_text(pdf_path)

        # Parse metadata from cover/early pages
        metadata = self._parse_metadata(full_text, page_texts[:3], ticker)

        # Detect sections across all pages
        sections = self._detect_sections(page_texts)

        return ParsedAnalystReport(
            metadata=metadata,
            sections=sections,
            full_text=full_text,
            source_path=str(pdf_path),
        )

    def _extract_text(self, pdf_path: Path) -> tuple[str, list[str]]:
        """
        Extract text from PDF file.

        Returns:
            Tuple of (full_text, list of per-page texts)
        """
        doc = fitz.open(pdf_path)
        page_texts = []
        full_parts = []

        for page in doc:
            text = page.get_text()
            page_texts.append(text)
            full_parts.append(text)

        doc.close()
        return "\n".join(full_parts), page_texts

    def _parse_metadata(
        self,
        full_text: str,
        cover_pages: list[str],
        ticker: str = "",
    ) -> AnalystReportMetadata:
        """Parse metadata from report cover pages."""
        metadata = AnalystReportMetadata()
        metadata.ticker = ticker

        # Combine cover pages for analysis
        cover_text = "\n".join(cover_pages[:3])

        # Extract rating
        rating_match = RATING_PATTERNS.search(cover_text)
        if rating_match:
            metadata.rating = rating_match.group(1)

        # Extract price target
        pt_match = PRICE_TARGET_PATTERNS.search(cover_text)
        if pt_match:
            metadata.price_target = f"${pt_match.group(1)}"

        # Extract bank/firm name
        bank_match = BANK_PATTERNS.search(cover_text)
        if bank_match:
            metadata.firm = bank_match.group(1)

        # Try to extract date (various formats)
        date_patterns = [
            r"(\w+\s+\d{1,2},?\s+\d{4})",  # January 15, 2024
            r"(\d{1,2}\s+\w+\s+\d{4})",  # 15 January 2024
            r"(\d{1,2}/\d{1,2}/\d{2,4})",  # 01/15/2024
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, cover_text)
            if date_match:
                metadata.report_date = date_match.group(1)
                break

        # Try to extract ticker if not provided
        if not metadata.ticker:
            # Look for common ticker patterns (NYSE:AAPL, NASDAQ:GOOG, etc.)
            ticker_match = re.search(
                r"(?:NYSE|NASDAQ|AMEX|TSX)[:\s]+([A-Z]{1,5})",
                cover_text,
            )
            if ticker_match:
                metadata.ticker = ticker_match.group(1)
            else:
                # Try to find standalone ticker
                ticker_match = re.search(r"\(([A-Z]{1,5})\)", cover_text)
                if ticker_match:
                    metadata.ticker = ticker_match.group(1)

        return metadata

    def _detect_sections(self, page_texts: list[str]) -> list[ReportSection]:
        """Detect and extract sections from report pages."""
        sections = []

        for page_num, page_text in enumerate(page_texts, start=1):
            # Check each section pattern against this page
            for section_name, pattern in ANALYST_SECTION_PATTERNS.items():
                matches = list(pattern.finditer(page_text))

                for match in matches:
                    # Extract content from match to next section header or end of page
                    start_pos = match.end()

                    # Find next section header on this page
                    end_pos = len(page_text)
                    for other_name, other_pattern in ANALYST_SECTION_PATTERNS.items():
                        next_match = other_pattern.search(page_text, start_pos)
                        if next_match and next_match.start() < end_pos:
                            end_pos = next_match.start()

                    content = page_text[start_pos:end_pos].strip()

                    # Skip if content is too short (likely TOC entry)
                    if len(content) < 100:
                        continue

                    sections.append(
                        ReportSection(
                            title=section_name,
                            content=content,
                            page_num=page_num,
                        )
                    )

        # Deduplicate sections (same title appearing on consecutive pages)
        deduplicated = []
        seen_titles = set()
        for section in sections:
            key = f"{section.title}_{section.page_num}"
            if key not in seen_titles:
                seen_titles.add(key)
                deduplicated.append(section)

        return deduplicated


def parse_analyst_report(pdf_path: Path, ticker: str = "") -> ParsedAnalystReport:
    """
    Convenience function to parse an analyst report PDF.

    Args:
        pdf_path: Path to PDF file
        ticker: Stock ticker (optional)

    Returns:
        ParsedAnalystReport with extracted content
    """
    parser = AnalystReportParser()
    return parser.parse_file(pdf_path, ticker)
