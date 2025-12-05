"""SEC filing HTML parser with section detection and table-aware chunking."""

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from .base_parser import BaseParser
from .constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    TABLE_MAX_TOKENS,
)
from .section_patterns import (
    CHUNK_SIZE_BY_SECTION,
    get_section_patterns,
    get_section_regex,
)
from .table_utils import (
    TABLE_END_MARKER,
    TABLE_START_MARKER,
    Table,
    convert_tables_to_readable,
)

# Suppress XML-as-HTML warning for iXBRL files
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class Section:
    """A section extracted from an SEC filing.

    Parent-child retrieval: Sections are the "parent" documents.
    Chunks reference their parent section via section_id.
    """

    item: str  # e.g., "Item 1", "Item 7A"
    title: str  # e.g., "Business", "Risk Factors"
    content: str  # Full text content
    start_pos: int  # Character position in original text
    end_pos: int  # End position
    section_id: str = ""  # Unique ID: "{source_hash}_{section_key}"
    # Table Intelligence - store Table objects for propagation to chunks
    tables: list = field(default_factory=list)  # List of Table objects

    @property
    def approx_tokens(self) -> int:
        """Approximate token count (words * 1.3).

        Used for size-limited parent retrieval.
        """
        return int(len(self.content.split()) * 1.3)


@dataclass
class Chunk:
    """A chunk of text ready for embedding.

    Quick Win #2: Now includes position tracking for context expansion.
    Phase 1.1: Added prev/next chunk linking for context traversal.
    Parent-child: Added section_id for parent document retrieval.
    Phase 2: Table Intelligence - content_type, table_type for filtering.
    """

    text: str
    metadata: dict = field(default_factory=dict)
    # Position tracking (Quick Win #2)
    chunk_id: str = ""  # Unique ID: "{source_hash}_{section}_{index}"
    position_in_section: int = 0
    total_chunks_in_section: int = 0
    is_section_start: bool = False
    is_section_end: bool = False
    # Chunk linking (Phase 1.1) - enables context expansion
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    # Parent-child retrieval - links chunk to parent section
    section_id: str = ""  # Parent section ID: "{source_hash}_{section_key}"
    # Phase 2: Table Intelligence
    content_type: str = "text"  # "text" or "table"
    has_table: bool = False  # Whether this chunk contains a table
    table_html: str = ""  # Original HTML for re-rendering
    table_caption: str = ""  # Table caption/title for display
    table_type: str = ""  # financial_statement, compensation, governance, comparison, schedule, metrics, other
    is_table_continuation: bool = False  # True if this is a split table chunk (not first)

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (words * 1.3)."""
        return int(len(self.text.split()) * 1.3)


@dataclass
class ParsedFiling:
    """Result of parsing an SEC filing."""

    ticker: str
    filing_type: str
    filing_date: str
    sections: list[Section]
    full_text: str
    source_path: str
    # iXBRL fiscal period metadata (extracted from hidden tags)
    fiscal_year: str = ""  # e.g., "2026"
    fiscal_period: str = ""  # e.g., "Q2", "FY"
    fiscal_year_end: str = ""  # e.g., "3/31" (month/day)


class SECFilingParser(BaseParser["ParsedFiling"]):
    """Parser for SEC HTML filings (10-K, 10-Q, 8-K, etc.)."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        """
        Initialize parser.

        Args:
            chunk_size: Target chunk size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size (don't split below this)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def parse_file(
        self,
        file_path: Path,
        ticker: str = "",
        filing_type: str = "",
        filing_date: str = "",
    ) -> ParsedFiling:
        """
        Parse an SEC HTML filing.

        Args:
            file_path: Path to HTML file
            ticker: Stock ticker (optional, extracted from filename if not provided)
            filing_type: Filing type (e.g., "10-K")
            filing_date: Filing date (YYYY-MM-DD)

        Returns:
            ParsedFiling with extracted sections and full text
        """
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Parse HTML
        soup = BeautifulSoup(content, "lxml")

        # Extract fiscal metadata BEFORE removing hidden elements (they contain the data)
        fiscal_year, fiscal_period, fiscal_year_end = self._extract_fiscal_metadata(soup)

        # Remove hidden XBRL elements
        self._remove_hidden_elements(soup)

        # Extract text
        full_text = self._extract_text(soup)

        # Detect sections - use HTML-based detection for 10-Q/10-K/8-K/DEF14A if available
        filing_type_upper = filing_type.upper().replace("-", "").replace(" ", "")
        if filing_type_upper in ("10Q", "10K", "8K", "DEF14A", "PROXY"):
            # Try HTML-based detection first (more reliable)
            html_sections = self._detect_sections_from_html(content, filing_type)
            if html_sections:
                # Convert HTML positions to text content
                sections = self._html_sections_to_text(html_sections, content, full_text, filing_type)
            else:
                # Fall back to text-based detection
                sections = self._detect_sections(full_text, filing_type)
        else:
            sections = self._detect_sections(full_text, filing_type)

        return ParsedFiling(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            sections=sections,
            full_text=full_text,
            source_path=str(file_path),
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            fiscal_year_end=fiscal_year_end,
        )

    def _html_sections_to_text(
        self,
        html_sections: list[tuple[str, str, int, int]],
        html: str,
        full_text: str,
        filing_type: str,
    ) -> list[Section]:
        """Convert HTML-detected sections to text-based Section objects.

        Extracts clean text content for each section detected from HTML.
        """
        sections = []

        for item_key, title, html_start, html_end in html_sections:
            # Extract HTML slice for this section
            section_html = html[html_start:html_end]

            # Parse and extract text
            section_soup = BeautifulSoup(section_html, "lxml")
            self._remove_hidden_elements(section_soup)
            # Table Intelligence: use readable format with markers, capture HTML for display
            table_metadata = convert_tables_to_readable(section_soup)
            section_text = section_soup.get_text(separator="\n", strip=True)

            # Clean up whitespace (preserve table markers)
            lines = []
            for line in section_text.split("\n"):
                stripped = line.strip()
                if stripped.startswith("|") or stripped.startswith("[TABLE:") or stripped == "[/TABLE]":
                    lines.append(stripped)
                elif stripped:
                    lines.append(re.sub(r"\s+", " ", stripped))
            content = "\n".join(lines)

            if len(content) < 50:
                continue

            sections.append(
                Section(
                    item=item_key,
                    title=title,
                    content=content,
                    start_pos=html_start,
                    end_pos=html_end,
                    tables=table_metadata,
                )
            )

        return sections

    def _extract_fiscal_metadata(self, soup: BeautifulSoup) -> tuple[str, str, str]:
        """
        Extract fiscal period metadata from iXBRL hidden tags.

        Must be called BEFORE _remove_hidden_elements destroys the ix:hidden section.

        Returns:
            Tuple of (fiscal_year, fiscal_period, fiscal_year_end)
            e.g., ("2026", "Q2", "3/31")
        """
        fiscal_year = ""
        fiscal_period = ""
        fiscal_year_end = ""

        # Find ix:hidden section which contains DEI (Document and Entity Information)
        hidden_elem = soup.find("ix:hidden")
        if hidden_elem and isinstance(hidden_elem, Tag):
            # Look for DocumentFiscalYearFocus (e.g., "2026")
            year_elem = hidden_elem.find("ix:nonnumeric", {"name": "dei:DocumentFiscalYearFocus"})
            if year_elem and hasattr(year_elem, "get_text"):
                fiscal_year = year_elem.get_text(strip=True)

            # Look for DocumentFiscalPeriodFocus (e.g., "Q1", "Q2", "Q3", "Q4", "FY")
            period_elem = hidden_elem.find("ix:nonnumeric", {"name": "dei:DocumentFiscalPeriodFocus"})
            if period_elem and hasattr(period_elem, "get_text"):
                fiscal_period = period_elem.get_text(strip=True)

            # Look for CurrentFiscalYearEndDate (e.g., "3/31")
            year_end_elem = hidden_elem.find("ix:nonnumeric", {"name": "dei:CurrentFiscalYearEndDate"})
            if year_end_elem and hasattr(year_end_elem, "get_text"):
                fiscal_year_end = year_end_elem.get_text(strip=True)

        return fiscal_year, fiscal_period, fiscal_year_end

    @staticmethod
    def extract_board_members_from_ixbrl(soup: BeautifulSoup) -> list[str]:
        """
        Extract board member names from iXBRL context elements.

        DEF 14A proxy statements contain rich iXBRL metadata with member tags
        like dt:RickMcConnellMember in xbrli:context elements with IndividualAxis
        dimension. This extracts those names.

        Args:
            soup: BeautifulSoup object of the HTML content

        Returns:
            List of board member names (e.g., ["Rick McConnell", "John Van Siclen"])
        """
        members = set()

        # Find all xbrli:context elements
        for context in soup.find_all("xbrli:context"):
            # Look for explicitMember elements with IndividualAxis dimension
            for member_elem in context.find_all("xbrldi:explicitmember"):
                dimension_attr = member_elem.get("dimension", "")
                dimension = str(dimension_attr) if dimension_attr else ""

                # Check if this is an individual (board member, executive, etc.)
                if "IndividualAxis" in dimension or "individualaxis" in dimension.lower():
                    member_tag = member_elem.get_text(strip=True)

                    # Extract name from tags like "dt:RickMcConnellMember"
                    if ":" in member_tag:
                        member_name = member_tag.split(":")[-1]
                        # Remove "Member" suffix if present
                        if member_name.endswith("Member"):
                            member_name = member_name[:-6]

                        # Convert CamelCase to readable name (RickMcConnell -> Rick McConnell)
                        # Insert space before uppercase letters
                        readable_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", member_name)
                        members.add(readable_name)

        return sorted(list(members))

    def _remove_hidden_elements(self, soup: BeautifulSoup) -> None:
        """Remove hidden XBRL and other non-visible elements."""
        # Remove elements with display:none
        for elem in soup.find_all(style=lambda x: bool(x and "display:none" in str(x).lower())):
            elem.decompose()

        # Remove ix:header (XBRL metadata)
        for elem in soup.find_all(["ix:header", "ix:hidden"]):
            elem.decompose()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from parsed HTML, preserving table structure."""
        # Convert tables to readable format with markers
        convert_tables_to_readable(soup)

        # Get text with newlines as separators
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace while preserving paragraph structure
        # Preserve table markers and pipe-formatted lines
        lines = []
        in_table = False
        for line in text.split("\n"):
            stripped = line.strip()

            # Detect table markers and pipe-formatted lines
            if stripped.startswith("[TABLE:") or stripped == "[/TABLE]" or stripped.startswith("|"):
                in_table = stripped.startswith("[TABLE:")
                lines.append(stripped)
            elif in_table and not stripped:
                # End of table content
                in_table = False
                lines.append("")
            elif stripped:
                # Normal text - normalize whitespace
                normalized = re.sub(r"\s+", " ", stripped)
                lines.append(normalized)

        return "\n".join(lines)

    def _merge_duplicate_sections(self, sections: list[Section], min_content_length: int = 500) -> list[Section]:
        """Merge sections with the same item number.

        SEC filings often have multiple references to the same item (TOC, headers,
        cross-references). This merges them in document order, filtering out
        small fragments that are likely TOC entries or cross-references.

        Args:
            sections: List of detected sections (may have duplicates)
            min_content_length: Minimum content length to include (filters TOC junk)

        Returns:
            List of merged sections, one per unique item
        """
        from collections import defaultdict

        # Group sections by item, filtering out tiny fragments
        section_groups: dict[str, list[Section]] = defaultdict(list)
        for section in sections:
            if len(section.content) >= min_content_length:
                section_groups[section.item].append(section)

        # Merge each group in document order
        merged = []
        for item, group in section_groups.items():
            if not group:
                continue

            # Sort by position in document
            group.sort(key=lambda s: s.start_pos)

            # Combine content with separators
            combined_content = "\n\n---\n\n".join(s.content for s in group)

            merged.append(
                Section(
                    item=item,
                    title=group[0].title,  # Use title from first occurrence
                    content=combined_content,
                    start_pos=group[0].start_pos,
                    end_pos=group[-1].end_pos,
                )
            )

        # Sort by position for consistent output order
        merged.sort(key=lambda s: s.start_pos)
        return merged

    def _detect_sections(self, text: str, filing_type: str = "") -> list[Section]:
        """Detect and extract sections from filing text.

        For plain text input, uses regex matching with heuristics to distinguish
        real sections from TOC entries.
        """
        # Get appropriate regex and patterns for this filing type
        header_regex = get_section_regex(filing_type)
        patterns = get_section_patterns(filing_type)

        # Find all section headers
        matches = list(header_regex.finditer(text))

        # Handle DEF 14A differently (text-based section detection)
        filing_type_upper = filing_type.upper().replace("-", "").replace(" ", "")
        if filing_type_upper in ("DEF14A", "PROXY"):
            return self._detect_sections_def14a(text, matches, patterns)

        # For 10-Q, we need to track Part I vs Part II context
        part_positions = []
        if filing_type_upper == "10Q":
            # Find Part I and Part II markers
            part_regex = re.compile(r"PART\s+(I{1,2})(?:\s|\.|\b)", re.IGNORECASE)
            for part_match in part_regex.finditer(text):
                part_num = part_match.group(1).upper()
                part_positions.append((part_match.start(), f"Part {part_num}"))
            part_positions.sort(key=lambda x: x[0])

        def get_part_for_position(pos: int) -> str:
            """Determine which Part an item belongs to based on position."""
            if not part_positions:
                return ""
            current_part = ""
            for part_pos, part_name in part_positions:
                if pos >= part_pos:
                    current_part = part_name
                else:
                    break
            return current_part

        # Build list of all candidate sections with their item keys
        candidates: list[tuple[str, str, re.Match]] = []
        for match in matches:
            item_num = match.group(1).upper()

            # Build the item key - include Part prefix for 10-Q
            if filing_type_upper == "10Q":
                part = get_part_for_position(match.start())
                if part:
                    item_key = f"{part}, Item {item_num}"
                else:
                    item_key = f"Item {item_num}"
            else:
                item_key = f"Item {item_num}"

            matched_title = match.group(2).strip() if match.group(2) else ""
            title = patterns.get(item_key, matched_title)
            candidates.append((item_key, title, match))

        # For 10-Q, filter out invalid items
        if filing_type_upper == "10Q":
            valid_10q_items = {
                "Part I, Item 1",
                "Part I, Item 2",
                "Part I, Item 3",
                "Part I, Item 4",
                "Part II, Item 1",
                "Part II, Item 1A",
                "Part II, Item 2",
                "Part II, Item 3",
                "Part II, Item 4",
                "Part II, Item 5",
                "Part II, Item 6",
            }
            candidates = [(k, t, m) for k, t, m in candidates if k in valid_10q_items]

        # For each occurrence, calculate content length until next DIFFERENT item
        from collections import defaultdict

        all_matches_sorted = sorted(candidates, key=lambda x: x[2].start())

        occurrence_with_length: list[tuple[str, str, re.Match, int]] = []
        for i, (item_key, title, match) in enumerate(all_matches_sorted):
            end_pos = len(text)
            for j in range(i + 1, len(all_matches_sorted)):
                if all_matches_sorted[j][0] != item_key:
                    end_pos = all_matches_sorted[j][2].start()
                    break
            content_len = end_pos - match.end()
            occurrence_with_length.append((item_key, title, match, content_len))

        # Group by item_key and select occurrence with LONGEST content
        item_occurrences: dict[str, list[tuple[str, re.Match, int]]] = defaultdict(list)
        for item_key, title, match, content_len in occurrence_with_length:
            item_occurrences[item_key].append((title, match, content_len))

        best_occurrences: dict[str, tuple[str, re.Match]] = {}
        for item_key, occurrences in item_occurrences.items():
            occurrences.sort(key=lambda x: x[2], reverse=True)
            title, match, _ = occurrences[0]
            best_occurrences[item_key] = (title, match)

        # Build sections using only the best occurrences
        sorted_items = sorted(best_occurrences.items(), key=lambda x: x[1][1].start())

        sections = []
        for i, (item_key, (title, match)) in enumerate(sorted_items):
            start_pos = match.start()
            end_pos = sorted_items[i + 1][1][1].start() if i + 1 < len(sorted_items) else len(text)
            content = text[match.end() : end_pos].strip()

            if len(content) < 50:
                continue

            sections.append(
                Section(
                    item=item_key,
                    title=title,
                    content=content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )

        return sections

    def _detect_sections_from_html(self, html: str, filing_type: str = "") -> list[tuple[str, str, int, int]]:
        """Detect sections directly from HTML using structural patterns.

        This is more reliable than text-based detection because:
        1. Real section headers are bold (font-weight:700) and all-caps
        2. TOC entries are wrapped in anchor tags
        3. Cross-references are mixed-case and inline

        Returns list of (item_key, title, start_pos, end_pos) tuples.
        """
        filing_type_upper = filing_type.upper().replace("-", "").replace(" ", "")
        patterns = get_section_patterns(filing_type)

        # Handle DEF 14A proxy statements separately
        if filing_type_upper in ("DEF14A", "PROXY"):
            return self._detect_def14a_sections_from_html(html, patterns)

        # Pattern for bold section headers (font-weight:700)
        # Matches PART headers and ITEM headers (both 10-K/10-Q and 8-K formats)
        # 10-K/10-Q: Item 1, Item 1A, Item 7A, etc.
        # 8-K: Item 1.01, Item 2.02, Item 9.01, etc.
        section_pattern = re.compile(
            r"<span[^>]*font-weight:\s*700[^>]*>"
            r"([^<]*(?:PART\s+I{1,4}|ITEM\s+(?:\d+[A-C]?|\d+\.\d{2}))[^<]*)"
            r"</span>",
            re.IGNORECASE,
        )

        sections: list[tuple[str, str, int, int]] = []
        current_part = ""

        for match in section_pattern.finditer(html):
            pos = match.start()
            heading_text = match.group(1).strip()

            # Skip if inside a TOC link (check for href="#" before this span)
            before_context = html[max(0, pos - 200) : pos]
            if re.search(r'href="#[^"]*"\s*>?\s*$', before_context):
                continue

            # Skip if not a standalone header (real headers end with </span></div>)
            after_context = html[match.end() : match.end() + 100]
            if not re.match(r"\s*</div>", after_context):
                continue

            # Parse PART marker
            part_match = re.match(r"PART\s+(I{1,2})", heading_text, re.IGNORECASE)
            if part_match:
                current_part = f"Part {part_match.group(1).upper()}"
                continue  # PART headers don't have content themselves

            # Parse ITEM marker (handles both 10-K/10-Q and 8-K formats)
            # 10-K/10-Q: Item 1, Item 1A, etc.
            # 8-K: Item 1.01, Item 2.02, etc.
            item_match = re.match(r"ITEM\s+(\d+[A-C]?|\d+\.\d{2})", heading_text, re.IGNORECASE)
            if not item_match:
                continue

            item_num = item_match.group(1).upper()

            # Build item key
            if filing_type_upper == "10Q" and current_part:
                item_key = f"{current_part}, Item {item_num}"
            else:
                item_key = f"Item {item_num}"

            # Get title from patterns or extract from heading
            title = patterns.get(item_key, "")
            if not title:
                # Extract title after item number (handles both formats)
                title_match = re.search(
                    r"ITEM\s+(?:\d+[A-C]?|\d+\.\d{2})\.?\s*[-–—]?\s*(.+)",
                    heading_text,
                    re.IGNORECASE,
                )
                if title_match:
                    title = title_match.group(1).strip()

            sections.append((item_key, title, pos, 0))  # end_pos filled later

        # Fill in end positions
        result = []
        for i, (item_key, title, start_pos, _) in enumerate(sections):
            end_pos = sections[i + 1][2] if i + 1 < len(sections) else len(html)
            result.append((item_key, title, start_pos, end_pos))

        return result

    def _detect_def14a_sections_from_html(self, html: str, patterns: dict[str, str]) -> list[tuple[str, str, int, int]]:
        """Detect sections in DEF 14A proxy statements from HTML.

        DEF 14A sections are typically:
        - PROPOSAL 1, PROPOSAL 2, etc. (voting proposals)
        - Board of Directors
        - Executive Compensation
        - Compensation Discussion and Analysis
        - Director Compensation
        - Corporate Governance
        - Related Party Transactions
        - Stock Ownership

        Returns list of (section_key, title, start_pos, end_pos) tuples.
        """
        sections: list[tuple[str, str, int, int]] = []

        # Pattern 1: PROPOSAL headings (bold, often all-caps or Title Case)
        # Matches: "PROPOSAL 1", "PROPOSAL NO. 1", "Proposal 1:", etc.
        proposal_pattern = re.compile(
            r"<(?:p|div|td)[^>]*(?:font-weight:\s*(?:bold|700)|font-size:\s*1[4-9])[^>]*>"
            r"[^<]*?(PROPOSAL\s+(?:NO\.\s*)?\d+[^<]{0,150}?)"
            r"</(?:p|div|td)>",
            re.IGNORECASE,
        )

        # Pattern 2: Standard section headings (bold, typically all-caps or Title Case)
        # Matches key proxy sections like Board of Directors, Compensation, etc.
        section_pattern = re.compile(
            r"<(?:p|div|td)[^>]*(?:font-weight:\s*(?:bold|700)|font-size:\s*1[4-9])[^>]*>"
            r"[^<]*?("
            r"(?:BOARD\s+OF\s+DIRECTORS)|"
            r"(?:EXECUTIVE\s+COMPENSATION)|"
            r"(?:COMPENSATION\s+DISCUSSION\s+AND\s+ANALYSIS)|"
            r"(?:DIRECTOR\s+COMPENSATION)|"
            r"(?:CORPORATE\s+GOVERNANCE)|"
            r"(?:RELATED\s+PARTY\s+TRANSACTIONS)|"
            r"(?:STOCK\s+OWNERSHIP)|"
            r"(?:NOMINEES\s+FOR\s+DIRECTOR)|"
            r"(?:SECURITY\s+OWNERSHIP)"
            r")[^<]{0,100}?"
            r"</(?:p|div|td)>",
            re.IGNORECASE,
        )

        # Find all proposals
        for match in proposal_pattern.finditer(html):
            pos = match.start()
            heading_text = match.group(1).strip()

            # Skip if inside a TOC link
            before_context = html[max(0, pos - 200) : pos]
            if re.search(r'href="#[^"]*"\s*>?\s*$', before_context, re.IGNORECASE):
                continue

            # Clean up heading text and normalize
            heading_clean = re.sub(r"\s+", " ", heading_text)

            # Extract proposal number
            prop_match = re.match(r"PROPOSAL\s+(?:NO\.\s*)?(\d+)", heading_clean, re.IGNORECASE)
            if prop_match:
                prop_num = prop_match.group(1)
                section_key = f"Proposal {prop_num}"

                # Try to extract full title
                title_match = re.search(
                    r"PROPOSAL\s+(?:NO\.\s*)?\d+\s*[:\-–—]?\s*(.+)",
                    heading_clean,
                    re.IGNORECASE,
                )
                if title_match:
                    title = title_match.group(1).strip()
                else:
                    title = patterns.get(section_key, heading_clean)

                sections.append((section_key, title, pos, 0))

        # Find all standard sections
        for match in section_pattern.finditer(html):
            pos = match.start()
            heading_text = match.group(1).strip()

            # Skip if inside a TOC link
            before_context = html[max(0, pos - 200) : pos]
            if re.search(r'href="#[^"]*"\s*>?\s*$', before_context, re.IGNORECASE):
                continue

            # Normalize heading text
            heading_clean = re.sub(r"\s+", " ", heading_text).title()

            # Match to our pattern keys
            section_key = None
            for key in patterns:
                if key.upper() in heading_text.upper():
                    section_key = key
                    break

            if not section_key:
                # Create a normalized key from the heading
                section_key = heading_clean

            title = patterns.get(section_key, heading_clean)
            sections.append((section_key, title, pos, 0))

        # Remove duplicates (prefer the one with more context/further in doc, not TOC)
        seen_keys: dict[str, tuple[str, int, int]] = {}
        for section_key, title, start_pos, _ in sections:
            if section_key not in seen_keys:
                seen_keys[section_key] = (title, start_pos, 0)
            else:
                # Prefer the later occurrence (actual section vs TOC)
                if start_pos > seen_keys[section_key][1]:
                    seen_keys[section_key] = (title, start_pos, 0)

        # Rebuild sections list from deduplicated entries
        sections = [(k, v[0], v[1], 0) for k, v in seen_keys.items()]
        sections.sort(key=lambda x: x[2])  # Sort by position

        # Fill in end positions
        result = []
        for i, (section_key, title, start_pos, _) in enumerate(sections):
            end_pos = sections[i + 1][2] if i + 1 < len(sections) else len(html)
            result.append((section_key, title, start_pos, end_pos))

        return result

    def _detect_sections_def14a(self, text: str, matches: list, patterns: dict[str, str]) -> list[Section]:
        """Detect sections in DEF 14A (proxy statement) filings."""
        sections = []

        for i, match in enumerate(matches):
            # For DEF 14A, the match group is the section title itself
            section_title = match.group(1).strip()

            # Normalize the title to find in patterns
            title_key = section_title
            for key in patterns:
                if key.lower() in section_title.lower():
                    title_key = key
                    break

            title = patterns.get(title_key) or section_title

            # Determine section boundaries
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Extract content
            content = text[match.end() : end_pos].strip()

            # Skip empty sections
            if len(content) < 50:
                continue

            sections.append(
                Section(
                    item=title_key,
                    title=title,
                    content=content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )

        # Merge duplicate sections (TOC refs, cross-refs, etc.)
        return self._merge_duplicate_sections(sections)

    def _get_chunk_size_for_section(self, section_item: str) -> int:
        """Get the optimal chunk size for a given section type.

        Uses CHUNK_SIZE_BY_SECTION mapping, falling back to instance default.
        """
        return CHUNK_SIZE_BY_SECTION.get(section_item, CHUNK_SIZE_BY_SECTION.get("default", self.chunk_size))

    def chunk_section(
        self,
        section: Section,
        filing_metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Split a section into chunks for embedding.

        Args:
            section: Section to chunk
            filing_metadata: Additional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        metadata = {
            "section": section.item,
            "section_title": section.title,
            **(filing_metadata or {}),
        }

        text = section.content
        chunks = []

        # Get section-specific chunk size (Quick Win #1)
        section_chunk_size = self._get_chunk_size_for_section(section.item)

        # Simple paragraph-based chunking
        paragraphs = self._split_into_paragraphs(text)

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_tokens = len(para.split())
            is_table = self._is_table_paragraph(para)

            # Phase 2: Special handling for tables
            if is_table:
                # Flush any pending text chunk first
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text="\n\n".join(current_chunk),
                            metadata={**metadata, "chunk_index": len(chunks)},
                        )
                    )
                    current_chunk = []
                    current_size = 0

                if para_tokens <= TABLE_MAX_TOKENS:
                    # Table fits in one chunk - keep intact
                    chunks.append(
                        Chunk(
                            text=para,
                            metadata={**metadata, "chunk_index": len(chunks)},
                        )
                    )
                else:
                    # Large table - split by rows with repeated headers
                    table_chunks = self._split_large_table(para, TABLE_MAX_TOKENS)
                    for table_text, is_continuation in table_chunks:
                        chunk = Chunk(
                            text=table_text,
                            metadata={**metadata, "chunk_index": len(chunks)},
                            is_table_continuation=is_continuation,
                        )
                        chunks.append(chunk)
                continue

            # If single paragraph exceeds chunk size, split it
            if para_tokens > section_chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text="\n\n".join(current_chunk),
                            metadata={**metadata, "chunk_index": len(chunks)},
                        )
                    )
                    current_chunk = []
                    current_size = 0

                # Split large paragraph (use section-specific size)
                sub_chunks = self._split_large_paragraph(para, section_chunk_size)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(
                            text=sub,
                            metadata={**metadata, "chunk_index": len(chunks)},
                        )
                    )
            # If adding this paragraph would exceed chunk size
            elif current_size + para_tokens > section_chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text="\n\n".join(current_chunk),
                            metadata={**metadata, "chunk_index": len(chunks)},
                        )
                    )

                # Start new chunk (with overlap)
                if self.chunk_overlap > 0 and current_chunk:
                    # Include last paragraph(s) for overlap
                    overlap_paras = []
                    overlap_size = 0
                    for p in reversed(current_chunk):
                        p_size = len(p.split())
                        if overlap_size + p_size <= self.chunk_overlap:
                            overlap_paras.insert(0, p)
                            overlap_size += p_size
                        else:
                            break
                    current_chunk = overlap_paras + [para]
                    current_size = overlap_size + para_tokens
                else:
                    current_chunk = [para]
                    current_size = para_tokens
            else:
                current_chunk.append(para)
                current_size += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(
                Chunk(
                    text="\n\n".join(current_chunk),
                    metadata={**metadata, "chunk_index": len(chunks)},
                )
            )

        # Add position tracking, linking, and table metadata to chunks
        self._add_chunk_linking_metadata(chunks, section, filing_metadata)
        self._add_table_metadata(chunks, section)

        return chunks

    def _add_chunk_linking_metadata(
        self,
        chunks: list[Chunk],
        section: Section,
        filing_metadata: dict | None,
    ) -> None:
        """Add position tracking and chunk linking metadata.

        Sets chunk IDs, section IDs, and prev/next links for context traversal.
        Modifies chunks in place.
        """
        total_chunks = len(chunks)
        source_hash = filing_metadata.get("source", "")[:8] if filing_metadata else ""
        section_key = section.item.replace(" ", "_").replace(",", "")
        section_id = f"{source_hash}_{section_key}"

        for i, chunk in enumerate(chunks):
            chunk.position_in_section = i
            chunk.total_chunks_in_section = total_chunks
            chunk.is_section_start = i == 0
            chunk.is_section_end = i == total_chunks - 1
            chunk.chunk_id = f"{section_id}_{i}"
            chunk.section_id = section_id

            # Link to adjacent chunks for context traversal
            if i > 0:
                chunk.prev_chunk_id = f"{section_id}_{i - 1}"
            if i < total_chunks - 1:
                chunk.next_chunk_id = f"{section_id}_{i + 1}"

            # Also add to metadata for storage
            chunk.metadata["position_in_section"] = i
            chunk.metadata["total_chunks_in_section"] = total_chunks
            chunk.metadata["is_section_start"] = chunk.is_section_start
            chunk.metadata["is_section_end"] = chunk.is_section_end
            chunk.metadata["chunk_id"] = chunk.chunk_id
            chunk.metadata["section_id"] = chunk.section_id
            chunk.metadata["prev_chunk_id"] = chunk.prev_chunk_id
            chunk.metadata["next_chunk_id"] = chunk.next_chunk_id

    def _add_table_metadata(self, chunks: list[Chunk], section: Section) -> None:
        """Detect table markers and add table metadata to chunks.

        Looks for TABLE_START_MARKER in chunk text and links to Table objects
        from the section. Modifies chunks in place.
        """
        for chunk in chunks:
            if TABLE_START_MARKER in chunk.text:
                chunk.has_table = True
                chunk.content_type = "table"
                # Extract caption from marker: [TABLE: caption]
                table_match = re.search(r"\[TABLE:\s*([^\]]+)\]", chunk.text)
                if table_match:
                    caption = table_match.group(1).strip()
                    chunk.table_caption = caption
                    # Look up Table object from section's tables by caption match
                    for table_obj in section.tables:
                        # Handle both Table objects and legacy dicts
                        if isinstance(table_obj, Table):
                            if table_obj.caption.strip() == caption:
                                chunk.table_html = table_obj.html
                                chunk.table_type = table_obj.table_type
                                break
                        elif isinstance(table_obj, dict):
                            # Legacy dict support
                            if table_obj.get("caption", "").strip() == caption:
                                chunk.table_html = table_obj.get("html", "")
                                break

            # Add table metadata to chunk.metadata for storage
            chunk.metadata["content_type"] = chunk.content_type
            chunk.metadata["has_table"] = chunk.has_table
            chunk.metadata["table_type"] = chunk.table_type
            chunk.metadata["is_table_continuation"] = chunk.is_table_continuation

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, keeping tables together with structure.

        Phase 2: Tables preserve their newline structure for proper splitting.
        """
        paragraphs = []
        current = []
        in_table = False

        for line in text.split("\n"):
            stripped = line.strip()

            # Detect table boundaries
            if TABLE_START_MARKER in stripped:
                # Flush any pending text before table
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                in_table = True
                current = [stripped]
            elif TABLE_END_MARKER in stripped:
                current.append(stripped)
                # End of table - join with newlines to preserve row structure
                paragraphs.append("\n".join(current))
                current = []
                in_table = False
            elif in_table:
                # Inside table - keep each line separate
                current.append(stripped)
            elif not stripped:
                # Empty line outside table - paragraph break
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
            else:
                # Normal text line
                current.append(stripped)

        # Flush any remaining content
        if current:
            if in_table:
                # Unclosed table - preserve structure anyway
                paragraphs.append("\n".join(current))
            else:
                paragraphs.append(" ".join(current))

        return [p for p in paragraphs if p.strip()]

    def _split_large_paragraph(self, text: str, chunk_size: int | None = None) -> list[str]:
        """Split a large paragraph into smaller chunks at sentence boundaries.

        Args:
            text: Text to split
            chunk_size: Target chunk size in tokens (uses self.chunk_size if None)
        """
        target_size = chunk_size if chunk_size is not None else self.chunk_size

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current = []
        current_size = 0

        for sentence in sentences:
            sent_tokens = len(sentence.split())

            if current_size + sent_tokens > target_size:
                if current:
                    chunks.append(" ".join(current))
                current = [sentence]
                current_size = sent_tokens
            else:
                current.append(sentence)
                current_size += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_large_table(self, table_text: str, max_tokens: int = TABLE_MAX_TOKENS) -> list[tuple[str, bool]]:
        """Split a large table into smaller chunks, repeating header row.

        Phase 2: Table Intelligence - ensures tables are split intelligently.

        Args:
            table_text: Full table text with markers [TABLE: caption]...rows...[/TABLE]
            max_tokens: Maximum tokens per chunk (default 2000)

        Returns:
            List of (chunk_text, is_continuation) tuples.
            is_continuation=False for first chunk, True for subsequent.
        """
        # Parse the table structure
        # Format: [TABLE: caption]\nrow1\nrow2\n...\n[/TABLE]
        lines = table_text.strip().split("\n")

        if len(lines) < 3:
            # Too small to split
            return [(table_text, False)]

        # Extract parts
        header_line = lines[0]  # [TABLE: caption]
        footer_line = lines[-1]  # [/TABLE]
        data_lines = lines[1:-1]  # The actual table rows

        if not data_lines:
            return [(table_text, False)]

        # First data line is typically the header row (column names)
        table_header = data_lines[0]
        data_rows = data_lines[1:] if len(data_lines) > 1 else []

        # If no data rows beyond header, return as-is
        if not data_rows:
            return [(table_text, False)]

        # Calculate tokens for fixed parts
        header_tokens = len(header_line.split()) + len(table_header.split()) + len(footer_line.split())

        # Split data rows into chunks
        chunks = []
        current_rows = []
        current_tokens = header_tokens

        for row in data_rows:
            row_tokens = len(row.split())

            if current_tokens + row_tokens > max_tokens and current_rows:
                # Save current chunk
                chunk_lines = [header_line, table_header] + current_rows + [footer_line]
                chunk_text = "\n".join(chunk_lines)
                is_continuation = len(chunks) > 0
                chunks.append((chunk_text, is_continuation))

                # Start new chunk with this row
                current_rows = [row]
                current_tokens = header_tokens + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        # Don't forget the last chunk
        if current_rows:
            chunk_lines = [header_line, table_header] + current_rows + [footer_line]
            chunk_text = "\n".join(chunk_lines)
            is_continuation = len(chunks) > 0
            chunks.append((chunk_text, is_continuation))

        return chunks if chunks else [(table_text, False)]

    def _is_table_paragraph(self, para: str) -> bool:
        """Check if a paragraph is a table (has table markers)."""
        return TABLE_START_MARKER in para and TABLE_END_MARKER in para

    def chunk_filing(self, filing: ParsedFiling, source_hash: str = "") -> list[Chunk]:
        """
        Chunk an entire filing.

        Args:
            filing: ParsedFiling to chunk
            source_hash: Optional file hash for consistent section_id generation.
                         If provided, used for section_id; otherwise uses source_path.

        Returns:
            List of all chunks from all sections
        """
        all_chunks = []
        # Use provided hash for section_id, or fall back to source_path
        source_for_id = source_hash if source_hash else filing.source_path
        metadata = {
            "ticker": filing.ticker,
            "filing_type": filing.filing_type,
            "filing_date": filing.filing_date,
            "source": source_for_id,  # Used for section_id generation
        }

        for section in filing.sections:
            section_chunks = self.chunk_section(section, metadata)
            all_chunks.extend(section_chunks)

        return all_chunks
