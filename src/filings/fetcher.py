"""SEC EDGAR filing fetcher using edgartools."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from edgar import Company, set_identity
from rich.console import Console

from src.utils.file_locks import file_lock

console = Console()
logger = logging.getLogger(__name__)

# Supported filing types
FilingType = Literal["10-K", "10-Q", "8-K", "DEF 14A", "4"]

SUPPORTED_FILINGS: list[FilingType] = ["10-K", "10-Q", "8-K", "DEF 14A", "4"]


@dataclass
class FilingMetadata:
    """Metadata for a downloaded filing."""

    accession_number: str
    filing_type: str
    filing_date: str
    company_name: str
    ticker: str
    cik: str
    file_path: str
    file_hash: str
    file_format: str = "html"  # html, xml, or txt
    downloaded_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Manifest:
    """Tracks all downloaded filings for a company."""

    ticker: str
    cik: str
    company_name: str
    filings: dict[str, FilingMetadata] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: Path) -> None:
        """Save manifest to JSON file with locking."""
        with file_lock(path):
            data = {
                "ticker": self.ticker,
                "cik": self.cik,
                "company_name": self.company_name,
                "last_updated": datetime.now().isoformat(),
                "filings": {k: vars(v) for k, v in self.filings.items()},
            }
            path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load manifest from JSON file."""
        data = json.loads(path.read_text())
        filings = {k: FilingMetadata(**v) for k, v in data.get("filings", {}).items()}
        return cls(
            ticker=data["ticker"],
            cik=data["cik"],
            company_name=data["company_name"],
            filings=filings,
            last_updated=data.get("last_updated", ""),
        )


def compute_file_hash(content: bytes) -> str:
    """Compute MD5 hash of file content."""
    return hashlib.md5(content).hexdigest()


class SECFetcher:
    """Fetches SEC filings for a company."""

    def __init__(self, data_dir: Path, user_agent: str | None = None):
        """
        Initialize the fetcher.

        Args:
            data_dir: Base directory for storing filings (e.g., ./data)
            user_agent: SEC requires identification (e.g., "Name email@example.com")
        """
        self.data_dir = data_dir
        if user_agent:
            set_identity(user_agent)

    def _get_company_dir(self, ticker: str) -> Path:
        """Get or create company-specific directory."""
        company_dir = self.data_dir / ticker.upper()
        company_dir.mkdir(parents=True, exist_ok=True)
        return company_dir

    def _get_manifest(self, ticker: str) -> Manifest:
        """Load or create manifest for a company."""
        company_dir = self._get_company_dir(ticker)
        manifest_path = company_dir / "manifest.json"

        if manifest_path.exists():
            return Manifest.load(manifest_path)

        # Will be populated when we fetch company info
        return Manifest(ticker=ticker.upper(), cik="", company_name="")

    def _save_manifest(self, ticker: str, manifest: Manifest) -> None:
        """Save manifest to disk."""
        company_dir = self._get_company_dir(ticker)
        manifest_path = company_dir / "manifest.json"
        manifest.save(manifest_path)

    def fetch(
        self,
        ticker: str,
        filing_types: list[FilingType] | None = None,
        years: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        cik: str | None = None,
    ) -> list[FilingMetadata]:
        """
        Fetch SEC filings for a company.

        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types to fetch (default: all supported)
            years: Number of years of filings to retrieve (ignored if dates provided)
            start_date: Fetch filings on or after this date
            end_date: Fetch filings on or before this date
            cik: Optional CIK number to use instead of ticker lookup

        Returns:
            List of FilingMetadata for downloaded filings

        Note:
            If start_date or end_date is provided, the years parameter is ignored.
            If neither dates nor years are provided, defaults to 3 years.
        """
        if filing_types is None:
            filing_types = list(SUPPORTED_FILINGS)

        # Validate filing types
        for ft in filing_types:
            if ft not in SUPPORTED_FILINGS:
                raise ValueError(f"Unsupported filing type: {ft}")

        # Determine filtering mode
        use_date_filter = start_date is not None or end_date is not None
        if not use_date_filter and years is None:
            years = 3  # Default

        # Use CIK directly if provided, otherwise lookup by ticker
        lookup_id = cik if cik else ticker
        console.print(f"[bold]Looking up company: {lookup_id}[/bold]")
        logger.info(f"Looking up company: {lookup_id}")
        company = Company(lookup_id)

        manifest = self._get_manifest(ticker)
        manifest.cik = str(company.cik)
        manifest.company_name = company.name

        console.print(f"  Company: {company.name}")
        console.print(f"  CIK: {company.cik}")
        logger.info(f"Found company: {company.name} (CIK: {company.cik})")

        downloaded: list[FilingMetadata] = []
        company_dir = self._get_company_dir(ticker)

        for filing_type in filing_types:
            console.print(f"\n[bold]Fetching {filing_type} filings...[/bold]")
            logger.info(f"Fetching {filing_type} filings (start={start_date}, end={end_date}, years={years})")

            # Get filings of this type
            filings = company.get_filings(form=filing_type)

            # Set up filtering
            # edgartools returns filings in reverse chronological order (newest first)
            count = 0
            max_filings = None if use_date_filter else self._estimate_filing_count(filing_type, years)

            for filing in filings:
                # Count-based limit (when using years)
                if max_filings is not None and count >= max_filings:
                    break

                filing_dt = filing.filing_date  # This is a date object

                # Date-based filtering
                if use_date_filter:
                    # Skip if before start_date
                    if start_date and filing_dt < start_date:
                        # Since filings are in reverse chronological order,
                        # once we're before start_date, we can stop
                        break
                    # Skip if after end_date
                    if end_date and filing_dt > end_date:
                        continue

                accession = filing.accession_no

                # Skip if already downloaded with same hash
                if accession in manifest.filings:
                    console.print(f"  [dim]Skipping {accession} (already downloaded)[/dim]")
                    logger.debug(f"Skipping {filing_type} {accession} (already in manifest)")
                    count += 1
                    continue

                # Download the filing
                try:
                    # Get the best available content format
                    content, ext = self._get_filing_content(filing)

                    if content is None:
                        console.print(f"  [yellow]No content found for {accession}[/yellow]")
                        logger.warning(f"No content found for {filing_type} {accession}")
                        count += 1
                        continue

                    # Compute hash
                    file_hash = compute_file_hash(content)

                    # Save to disk
                    safe_date = filing.filing_date.strftime("%Y-%m-%d")
                    safe_type = filing_type.replace(" ", "_")
                    safe_accn = accession.replace("/", "_")
                    filename = f"{safe_type}_{safe_date}_{safe_accn}{ext}"
                    file_path = company_dir / filename
                    file_path.write_bytes(content)

                    # Create metadata
                    metadata = FilingMetadata(
                        accession_number=accession,
                        filing_type=filing_type,
                        filing_date=safe_date,
                        company_name=company.name,
                        ticker=ticker.upper(),
                        cik=str(company.cik),
                        file_path=str(file_path.relative_to(self.data_dir)),
                        file_hash=file_hash,
                        file_format=ext.lstrip("."),
                    )

                    manifest.filings[accession] = metadata
                    downloaded.append(metadata)

                    fmt_label = ext.lstrip(".")
                    console.print(f"  [green]Downloaded {filing_type} {safe_date} ({fmt_label})[/green]")
                    logger.info(f"Downloaded {filing_type} {safe_date} ({fmt_label}) - {file_path}")
                    count += 1

                except Exception as e:
                    console.print(f"  [red]Error fetching {accession}: {e}[/red]")
                    logger.error(f"Error fetching {filing_type} {accession}: {e}", exc_info=True)
                    count += 1
                    continue

        # Save updated manifest
        self._save_manifest(ticker, manifest)
        console.print(f"\n[bold green]Downloaded {len(downloaded)} new filings[/bold green]")
        logger.info(f"✓ Fetch complete: {len(downloaded)} new filings downloaded for {ticker}")

        return downloaded

    def _estimate_filing_count(self, filing_type: FilingType, years: int) -> int:
        """Estimate how many filings to fetch based on type and years."""
        # Approximate filing frequency per year
        frequencies = {
            "10-K": 1,  # Annual
            "10-Q": 3,  # Quarterly (3 per year, 10-K covers Q4)
            "8-K": 10,  # Variable, estimate ~10/year
            "DEF 14A": 1,  # Annual proxy
            "4": 20,  # Insider trades, highly variable
        }
        return frequencies.get(filing_type, 5) * years

    def _get_filing_content(self, filing) -> tuple[bytes | None, str]:
        """
        Get the best available content from a filing.

        Tries formats in order: HTML > XML > Text

        Returns:
            Tuple of (content_bytes, file_extension)
        """
        # Try HTML first (best for 10-K, 10-Q, 8-K, DEF 14A)
        try:
            html = filing.html()
            if html:
                content = html.encode("utf-8") if isinstance(html, str) else html
                return content, ".html"
        except Exception as e:
            logger.debug(f"HTML format not available: {e}")

        # Try XML (Form 4 and some other filings are XML-based)
        try:
            xml = filing.xml()
            if xml:
                content = xml.encode("utf-8") if isinstance(xml, str) else xml
                return content, ".xml"
        except Exception as e:
            logger.debug(f"XML format not available: {e}")

        # Fall back to text
        try:
            text = filing.text()
            if text:
                content = text.encode("utf-8") if isinstance(text, str) else text
                return content, ".txt"
        except Exception as e:
            logger.debug(f"Text format not available: {e}")

        return None, ""
