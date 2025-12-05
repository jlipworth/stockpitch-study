"""Earnings call transcript parser for S&P Capital IQ format PDFs."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from .base_parser import BaseParser


@dataclass
class Speaker:
    """A speaker in the earnings call."""

    name: str
    title: str = ""
    company: str = ""
    role: str = ""  # "executive" or "analyst"


@dataclass
class SpeakerBlock:
    """A block of text from a single speaker."""

    speaker: Speaker
    text: str
    section: str  # "presentation" or "qa"


@dataclass
class TranscriptMetadata:
    """Metadata extracted from transcript cover page."""

    company_name: str = ""
    ticker: str = ""
    event_type: str = ""  # e.g., "Q3 2024 Earnings Call"
    event_date: str = ""
    executives: list[Speaker] = field(default_factory=list)
    analysts: list[Speaker] = field(default_factory=list)


@dataclass
class ParsedTranscript:
    """Result of parsing an earnings call transcript."""

    metadata: TranscriptMetadata
    presentation_blocks: list[SpeakerBlock]
    qa_blocks: list[SpeakerBlock]
    full_text: str
    source_path: str

    @property
    def all_blocks(self) -> list[SpeakerBlock]:
        """All speaker blocks in order."""
        return self.presentation_blocks + self.qa_blocks


# Patterns for S&P Capital IQ transcript format
SECTION_PATTERNS = {
    "participants": re.compile(
        r"(?:CALL\s+PARTICIPANTS?|CORPORATE\s+PARTICIPANTS?|CONFERENCE\s+CALL\s+PARTICIPANTS?)",
        re.IGNORECASE,
    ),
    "executives": re.compile(r"EXECUTIVES?", re.IGNORECASE),
    "analysts": re.compile(r"ANALYSTS?", re.IGNORECASE),
    "presentation": re.compile(
        r"(?:PRESENTATION|PREPARED\s+REMARKS?|MANAGEMENT\s+DISCUSSION)",
        re.IGNORECASE,
    ),
    "qa": re.compile(
        r"(?:QUESTION\s*(?:AND|&)\s*ANSWER|Q\s*&\s*A|Q&A)",
        re.IGNORECASE,
    ),
}

# Pattern to detect speaker introductions (name in bold, usually followed by title)
# S&P format: "John Smith" (bold) followed by "CEO, Company Name" (italic or normal)
SPEAKER_INTRO_PATTERN = re.compile(
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*$",  # Name on its own line
    re.MULTILINE,
)

# Pattern for speaker with title on same/next line
SPEAKER_WITH_TITLE_PATTERN = re.compile(
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[,\-–—]?\s*(.+?)$",
    re.MULTILINE,
)


class TranscriptParser(BaseParser["ParsedTranscript"]):
    """Parser for S&P Capital IQ earnings call transcript PDFs."""

    def parse_file(self, pdf_path: Path, ticker: str = "") -> ParsedTranscript:
        """
        Parse a transcript PDF file.

        Args:
            pdf_path: Path to PDF file
            ticker: Stock ticker (optional, will try to extract from document)

        Returns:
            ParsedTranscript with extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Transcript PDF not found: {pdf_path}")

        # Extract text from PDF
        full_text = self._extract_text(pdf_path)

        # Parse metadata from cover page
        metadata = self._parse_metadata(full_text, ticker)

        # Split into sections
        sections = self._split_sections(full_text)

        # Parse speaker blocks from each section
        presentation_blocks = self._parse_speaker_blocks(sections.get("presentation", ""), "presentation", metadata)
        qa_blocks = self._parse_speaker_blocks(sections.get("qa", ""), "qa", metadata)

        return ParsedTranscript(
            metadata=metadata,
            presentation_blocks=presentation_blocks,
            qa_blocks=qa_blocks,
            full_text=full_text,
            source_path=str(pdf_path),
        )

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        doc = fitz.open(pdf_path)
        text_parts = []

        for page in doc:
            text_parts.append(page.get_text())

        doc.close()
        return "\n".join(text_parts)

    def _parse_metadata(self, text: str, ticker: str = "") -> TranscriptMetadata:
        """Parse metadata from transcript cover page."""
        metadata = TranscriptMetadata()

        # First few hundred characters usually contain metadata
        header = text[:2000]

        # Try to extract ticker (often in format "TICKER" or "(TICKER)")
        ticker_match = re.search(r"\(([A-Z]{1,5})\)", header)
        if ticker_match:
            metadata.ticker = ticker_match.group(1)
        elif ticker:
            metadata.ticker = ticker

        # Try to extract event type (e.g., "Q3 2024 Earnings Call")
        event_match = re.search(
            r"(Q[1-4]\s+\d{4}\s+(?:Earnings|Results)\s+(?:Call|Conference))",
            header,
            re.IGNORECASE,
        )
        if event_match:
            metadata.event_type = event_match.group(1)

        # Try to extract date
        date_match = re.search(
            r"(\w+\s+\d{1,2},?\s+\d{4})",
            header,
        )
        if date_match:
            metadata.event_date = date_match.group(1)

        # Parse participants section
        participants_section = self._extract_section(text, "participants")
        if participants_section:
            metadata.executives = self._parse_participants(participants_section, "executive")
            metadata.analysts = self._parse_participants(participants_section, "analyst")

        return metadata

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the text."""
        pattern = SECTION_PATTERNS.get(section_name)
        if not pattern:
            return ""

        # Find all occurrences and pick the one with substantial content after it
        # This skips TOC entries which have minimal content (just "..." and page numbers)
        matches = list(pattern.finditer(text))
        if not matches:
            return ""

        # Pick the match that has the most content before the next section
        # TOC entries typically have <100 chars before next pattern
        match = None
        for m in matches:
            # Check what follows this match
            next_100 = text[m.end() : m.end() + 100]
            # TOC entries have dots/page numbers, real sections have actual content
            if not re.match(r"^\s*\.{3,}|\s*\d+\s*$", next_100):
                match = m
                break

        if not match:
            # Fallback to last match if all look like TOC
            match = matches[-1]

        start = match.end()

        # Find the next section header
        # For participants section, only end at PRESENTATION or Q&A (not executives/analysts)
        end = len(text)
        if section_name == "participants":
            # Participants section contains executives and analysts subsections
            end_patterns = [SECTION_PATTERNS["presentation"], SECTION_PATTERNS["qa"]]
        else:
            end_patterns = [pat for name, pat in SECTION_PATTERNS.items() if name != section_name]

        for pat in end_patterns:
            next_match = pat.search(text, start)
            if next_match and next_match.start() < end:
                end = next_match.start()

        return text[start:end].strip()

    def _parse_participants(self, section: str, role_type: str) -> list[Speaker]:
        """Parse participant list from participants section."""
        participants = []

        # Find the relevant subsection (EXECUTIVES or ANALYSTS)
        if role_type == "executive":
            pattern = SECTION_PATTERNS["executives"]
            end_pattern = SECTION_PATTERNS["analysts"]
        else:
            pattern = SECTION_PATTERNS["analysts"]
            end_pattern = SECTION_PATTERNS["presentation"]

        match = pattern.search(section)
        if not match:
            return participants

        start = match.end()

        # Find end (next section header or end of text)
        end = len(section)
        # For executives, analysts section ends it; for analysts, presentation ends it
        next_match = end_pattern.search(section, start)
        if next_match:
            end = next_match.start()

        subsection = section[start:end]

        # Parse participants - handles two formats:
        # Format 1: "Name - Title" or "Name, Title" on same line
        # Format 2: Name on one line, title on following line(s)
        lines = [line.strip() for line in subsection.split("\n") if line.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]

            # Skip section headers and page markers
            if re.match(r"^(Page\s+\d+|S&P|Copyright|\d+$|EXECUTIVES?|ANALYSTS?)", line, re.IGNORECASE):
                i += 1
                continue

            # Format 1: "Name - Title" or "Name, Title" on same line
            # Pattern: 2-4 capitalized name words, then separator, then title
            inline_match = re.match(
                r"^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,3})\s*[-–—,]\s*(.+)$",
                line,
            )
            if inline_match:
                name = inline_match.group(1).strip()
                title = inline_match.group(2).strip()
                if name and len(name.split()) >= 2:
                    participants.append(Speaker(name=name, title=title, role=role_type))
                i += 1
                continue

            # Format 2: Name only on this line (title on next line)
            name_match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,3})$", line)

            if name_match:
                name = name_match.group(1).strip()
                title_parts = []

                # Collect title from following lines until we hit another name or section
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    # Stop if this looks like another name (standalone or with inline title)
                    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,3}(?:\s*[-–—,]|$)", next_line):
                        break
                    # Stop if this is a section marker
                    if re.match(r"^(Page\s+\d+|S&P|Copyright|EXECUTIVES?|ANALYSTS?)", next_line, re.IGNORECASE):
                        break
                    title_parts.append(next_line)
                    j += 1

                title = " ".join(title_parts).strip()
                # Clean up title (remove extra spaces, line break artifacts)
                title = re.sub(r"\s+", " ", title)

                if name and len(name.split()) >= 2:  # Valid name has at least 2 parts
                    participants.append(Speaker(name=name, title=title, role=role_type))

                i = j  # Skip to next potential name
            else:
                i += 1

        return participants

    def _split_sections(self, text: str) -> dict[str, str]:
        """Split transcript into main sections."""
        sections = {}

        def find_real_section(pattern: re.Pattern) -> re.Match | None:
            """Find the actual section match, skipping TOC entries."""
            matches = list(pattern.finditer(text))
            for m in matches:
                next_100 = text[m.end() : m.end() + 100]
                # TOC entries have dots/page numbers, real sections have content
                if not re.match(r"^\s*\.{3,}|\s*\d+\s*$", next_100):
                    return m
            return matches[-1] if matches else None

        # Find presentation and Q&A sections (skipping TOC entries)
        pres_match = find_real_section(SECTION_PATTERNS["presentation"])
        qa_match = find_real_section(SECTION_PATTERNS["qa"])

        if pres_match:
            pres_start = pres_match.end()
            pres_end = qa_match.start() if qa_match else len(text)
            sections["presentation"] = text[pres_start:pres_end].strip()

        if qa_match:
            qa_start = qa_match.end()
            sections["qa"] = text[qa_start:].strip()

        return sections

    def _parse_speaker_blocks(
        self,
        section_text: str,
        section_name: str,
        metadata: TranscriptMetadata,
    ) -> list[SpeakerBlock]:
        """Parse speaker blocks from a section."""
        blocks = []
        if not section_text:
            return blocks

        # Build speaker lookup from metadata
        known_speakers = {}
        for speaker in metadata.executives + metadata.analysts:
            # Store by first name, last name, and full name
            known_speakers[speaker.name.lower()] = speaker
            parts = speaker.name.split()
            if len(parts) >= 2:
                known_speakers[parts[-1].lower()] = speaker  # Last name
                known_speakers[parts[0].lower()] = speaker  # First name

        # Split on speaker name patterns
        # Look for lines that are just a name (typical in transcript format)
        lines = section_text.split("\n")

        current_speaker = None
        current_text = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and page markers
            if not stripped or re.match(r"^(Page\s+\d+|S&P|Copyright|\d+$)", stripped, re.IGNORECASE):
                if current_text and current_speaker:
                    current_text.append("")  # Preserve paragraph breaks
                continue

            # Check if this line is a speaker name
            # Names are typically 2-3 words, all starting with capitals
            is_speaker_name = (
                re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$", stripped) and stripped.lower() in known_speakers
            )

            if is_speaker_name:
                # Save previous block
                if current_speaker and current_text:
                    text = "\n".join(current_text).strip()
                    if text:
                        blocks.append(
                            SpeakerBlock(
                                speaker=current_speaker,
                                text=text,
                                section=section_name,
                            )
                        )

                # Start new block
                current_speaker = known_speakers.get(stripped.lower())
                if not current_speaker:
                    # Create unknown speaker
                    current_speaker = Speaker(name=stripped, role="unknown")
                current_text = []
            else:
                # Add to current block
                if current_speaker:
                    current_text.append(stripped)
                # If no speaker yet, this might be intro text - skip or handle

        # Save final block
        if current_speaker and current_text:
            text = "\n".join(current_text).strip()
            if text:
                blocks.append(
                    SpeakerBlock(
                        speaker=current_speaker,
                        text=text,
                        section=section_name,
                    )
                )

        return blocks


def parse_transcript(pdf_path: Path, ticker: str = "") -> ParsedTranscript:
    """
    Convenience function to parse a transcript PDF.

    Args:
        pdf_path: Path to PDF file
        ticker: Stock ticker (optional)

    Returns:
        ParsedTranscript with extracted content
    """
    parser = TranscriptParser()
    return parser.parse_file(pdf_path, ticker)
