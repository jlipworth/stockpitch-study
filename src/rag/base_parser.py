"""Abstract base class for document parsers.

Provides a common interface for all parser implementations:
- SECFilingParser
- TranscriptParser
- AnalystReportParser
- Form4Parser
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedDocument:
    """Base class for parsed document results.

    All parsed document types should inherit from this to ensure
    a consistent interface for downstream processing.
    """

    # Subclasses should define their own fields
    pass


class BaseParser[T](ABC):
    """Abstract base class for document parsers.

    All parsers should inherit from this class and implement
    the parse_file method. The generic type T represents the
    specific parsed result type (e.g., ParsedFiling, ParsedTranscript).

    Note: Subclasses may define additional parameters beyond file_path.
    The ABC only enforces the return type, not the full signature.

    Example:
        class SECFilingParser(BaseParser[ParsedFiling]):
            def parse_file(self, file_path: Path, ticker: str = "") -> ParsedFiling:
                ...
    """

    @abstractmethod
    def parse_file(self, file_path: Path, /) -> T:  # type: ignore[override]
        """Parse a document file and return structured data.

        Args:
            file_path: Path to the document file

        Returns:
            Parsed document of type T

        Note:
            Subclasses may accept additional keyword arguments
            (ticker, filing_type, etc.) specific to their document type.
        """
        ...
