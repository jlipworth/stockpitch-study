"""Summarize command for generating document summaries."""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from src.filings import Manifest
from src.summarizer import Summarizer

from . import (
    ANALYST_DIR,
    CONFERENCES_DIR,
    DATA_DIR,
    MISC_DIR,
    PRESENTATIONS_DIR,
    PROCESSED_DIR,
    TRANSCRIPTS_DIR,
    Progress,
    SourceType,
    SpinnerColumn,
    TextColumn,
    app,
    console,
)


def _summarize_sec_filings(
    ticker: str,
    doc_type: str | None,
    latest: bool,
    summarizer: Summarizer,
    console: Console,
) -> int:
    """Summarize SEC filings from manifest. Returns count of documents summarized."""
    company_dir = DATA_DIR / ticker
    manifest_path = company_dir / "manifest.json"

    if not manifest_path.exists():
        console.print(f"[dim]  No SEC filings found for {ticker}.[/dim]")
        return 0

    manifest = Manifest.load(manifest_path)

    # Filter filings
    filings_to_summarize = []
    for accession, filing_meta in manifest.filings.items():
        if doc_type and filing_meta.filing_type != doc_type:
            continue
        if filing_meta.filing_type == "4":
            continue
        filings_to_summarize.append(filing_meta)

    if not filings_to_summarize:
        console.print("[dim]  No SEC filings match criteria.[/dim]")
        return 0

    filings_to_summarize.sort(key=lambda f: f.filing_date, reverse=True)

    if latest:
        seen_types = set()
        filtered = []
        for f in filings_to_summarize:
            if f.filing_type not in seen_types:
                filtered.append(f)
                seen_types.add(f.filing_type)
        filings_to_summarize = filtered

    console.print(f"[bold cyan]SEC Filings ({len(filings_to_summarize)} documents)[/bold cyan]")
    count = 0

    try:
        for filing_meta in filings_to_summarize:
            file_path = DATA_DIR / filing_meta.file_path

            if not file_path.exists():
                console.print(f"  [yellow]File not found: {filing_meta.file_path}[/yellow]")
                continue

            output_path = summarizer.get_output_path(ticker, filing_meta.filing_type, filing_meta.filing_date)
            if output_path.exists():
                ftype, fdate = filing_meta.filing_type, filing_meta.filing_date
                console.print(f"  [dim]Skipping {ftype} {fdate} (exists)[/dim]")
                continue

            console.print(f"  Summarizing {filing_meta.filing_type} ({filing_meta.filing_date})")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Parsing...", total=None)

                def update_progress(current: int, total: int, section: str) -> None:
                    progress.update(task, description=f"[{current}/{total}] {section}...")

                result = summarizer.summarize_filing(
                    file_path=file_path,
                    ticker=ticker,
                    filing_type=filing_meta.filing_type,
                    filing_date=filing_meta.filing_date,
                    progress_callback=update_progress,
                )

                progress.update(task, description="[green]Done[/green]")

            result.save(output_path)
            count += 1
            console.print(f"    Saved to: {output_path}")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    return count


def _summarize_source_pdfs(
    ticker: str,
    source_dir: Path,
    doc_type: str,
    latest: bool,
    summarizer: Summarizer,
    console: Console,
) -> int:
    """Summarize PDFs from a source directory. Returns count summarized."""
    pdf_dir = source_dir / ticker
    if not pdf_dir.exists():
        console.print(f"[dim]  No {doc_type}s found for {ticker}.[/dim]")
        return 0

    pdf_files = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not pdf_files:
        console.print(f"[dim]  No {doc_type} PDFs found.[/dim]")
        return 0

    if latest:
        pdf_files = pdf_files[:1]

    console.print(f"[bold cyan]{doc_type.title()}s ({len(pdf_files)} documents)[/bold cyan]")
    count = 0

    for pdf_file in pdf_files:
        if _summarize_pdf_file(pdf_file, ticker, doc_type, summarizer, console):
            count += 1

    return count


def _generate_summary_filename(pdf_file: Path, doc_type: str) -> str:
    """Generate a clean summary filename based on document type and PDF name.

    Parses Capital IQ naming conventions to extract key information.
    Falls back to sanitized stem + date if parsing fails.
    """
    stem = pdf_file.stem
    file_mtime = datetime.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d")

    if doc_type == "transcript":
        # Pattern: "Company, Q# YYYY Earnings Call, Mon DD, YYYY"
        match = re.search(r"Q(\d)\s+(\d{4})\s+Earnings\s+Call,\s+(\w+)\s+(\d{1,2}),\s+(\d{4})", stem)
        if match:
            quarter, fiscal_year, month, day, call_year = match.groups()
            # Convert month name to number
            try:
                month_num = datetime.strptime(month, "%b").month
            except ValueError:
                try:
                    month_num = datetime.strptime(month, "%B").month
                except ValueError:
                    month_num = 1
            call_date = f"{call_year}-{month_num:02d}-{int(day):02d}"
            return f"earnings_Q{quarter}_FY{fiscal_year[-2:]}_{call_date}.md"

    elif doc_type == "conference":
        # Pattern: "Company Presents at [Conference Name], Mon-DD-YYYY HH:MM AM/PM"
        # Extract conference host and date
        match = re.search(r"Presents\s+at\s+(.+?),\s+(\w+)-(\d{2})-(\d{4})", stem)
        if match:
            conf_name, month, day, year = match.groups()
            # Extract the host/bank name (first few words, typically the bank)
            # Common patterns: "Goldman Sachs...", "Morgan Stanley...", "UBS...", "Citi..."
            host_match = re.match(
                r"(Goldman\s+Sachs|Morgan\s+Stanley|UBS|Citi|JPMorgan|Wells\s+Fargo|"
                r"Bank\s+of\s+America|Barclays|Deutsche\s+Bank|Credit\s+Suisse|"
                r"RBC|TD|Jefferies|Piper|Stifel|Baird|KeyBanc|Evercore|Wolfe|Bernstein)",
                conf_name,
                re.IGNORECASE,
            )
            if host_match:
                host = host_match.group(1).replace(" ", "_")
            else:
                # Take first two words as host
                words = conf_name.split()[:2]
                host = "_".join(words)
            # Clean up host name
            host = re.sub(r"[^\w]+", "_", host).strip("_")
            try:
                month_num = datetime.strptime(month, "%b").month
            except ValueError:
                try:
                    month_num = datetime.strptime(month, "%B").month
                except ValueError:
                    month_num = 1
            conf_date = f"{year}-{month_num:02d}-{day}"
            return f"conference_{host}_{conf_date}.md"

    elif doc_type == "analyst":
        # Pattern varies by source, try to extract bank name
        # Common: "Company - Bank Name - Report Type - Date"
        banks = [
            "Goldman Sachs",
            "Morgan Stanley",
            "JPMorgan",
            "Citi",
            "Bank of America",
            "Barclays",
            "Deutsche Bank",
            "UBS",
            "Credit Suisse",
            "RBC",
            "TD Securities",
            "Jefferies",
            "Piper Sandler",
            "Stifel",
            "William Blair",
            "Baird",
            "KeyBanc",
            "Evercore",
            "Wolfe Research",
            "Bernstein",
            "Cowen",
            "Needham",
            "Oppenheimer",
            "Raymond James",
            "Truist",
            "BTIG",
            "Mizuho",
            "Wedbush",
            "Loop Capital",
        ]
        for bank in banks:
            if bank.lower() in stem.lower():
                bank_name = bank.replace(" ", "_")
                return f"analyst_{bank_name}_{file_mtime}.md"
        # Fallback: use first recognizable word pattern
        return f"analyst_{file_mtime}_{_sanitize_filename(stem[:30])}.md"

    elif doc_type == "presentation":
        # Try to identify presentation type (Investor Day, Analyst Day, etc.)
        if re.search(r"investor\s*(day|session|meeting)", stem, re.IGNORECASE):
            return f"presentation_Investor_Day_{file_mtime}.md"
        elif re.search(r"analyst\s*day", stem, re.IGNORECASE):
            return f"presentation_Analyst_Day_{file_mtime}.md"
        elif re.search(r"capital\s*markets", stem, re.IGNORECASE):
            return f"presentation_Capital_Markets_Day_{file_mtime}.md"
        # Fallback
        return f"presentation_{file_mtime}_{_sanitize_filename(stem[:30])}.md"

    # Fallback for misc or unrecognized patterns
    return f"{doc_type}_{file_mtime}_{_sanitize_filename(stem[:40])}.md"


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use in a filename."""
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r"[^\w\-]", "_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def _summarize_pdf_file(
    pdf_file: Path,
    ticker: str,
    doc_type: str,
    summarizer: Summarizer,
    console: Console,
) -> bool:
    """Summarize a single PDF file. Returns True if summarized."""
    import os

    # Generate clean output filename
    summary_filename = _generate_summary_filename(pdf_file, doc_type)
    output_path = PROCESSED_DIR / ticker / summary_filename

    if output_path.exists():
        console.print(f"  [dim]Skipping {pdf_file.name} (exists)[/dim]")
        return False

    console.print(f"  Summarizing {pdf_file.name}")

    try:
        # Try to extract date from filename (common patterns: YYYY-MM-DD, Month DD, YYYY, etc.)
        date_match = re.search(r"(\w{3})[- ](\d{1,2})[,\- ](\d{4})", pdf_file.stem)
        if date_match:
            month_str, day, year = date_match.groups()
            try:
                parsed_date = datetime.strptime(f"{month_str} {day} {year}", "%b %d %Y")
                file_date = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                file_date = datetime.fromtimestamp(os.path.getmtime(pdf_file)).strftime("%Y-%m-%d")
        else:
            # Try ISO date format
            iso_match = re.search(r"(\d{4})[- ](\d{2})[- ](\d{2})", pdf_file.stem)
            if iso_match:
                file_date = f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"
            else:
                file_date = datetime.fromtimestamp(os.path.getmtime(pdf_file)).strftime("%Y-%m-%d")

        # Use appropriate parser based on doc_type
        if doc_type == "transcript":
            from src.rag.transcript_parser import TranscriptParser

            parser = TranscriptParser()
            parsed = parser.parse_file(pdf_file, ticker)

            # Create sections from transcript blocks
            sections = []
            from src.rag.parser import Section

            # Group by section (prepared_remarks, qa)
            for section_name in ["prepared_remarks", "qa"]:
                blocks = [b for b in parsed.all_blocks if b.section == section_name]
                if blocks:
                    content = "\n\n".join(
                        [f"**{b.speaker.name}** ({b.speaker.title or 'N/A'}):\n{b.text}" for b in blocks]
                    )
                    sections.append(
                        Section(
                            item=f"Transcript {section_name.replace('_', ' ').title()}",
                            title=section_name.replace("_", " ").title(),
                            content=content,
                            start_pos=0,
                            end_pos=len(content),
                        )
                    )

        elif doc_type == "analyst":
            from src.rag.analyst_parser import AnalystReportParser

            parser = AnalystReportParser()
            parsed = parser.parse_file(pdf_file, ticker)

            # Create sections from analyst report
            sections = []
            from src.rag.parser import Section

            for report_section in parsed.sections:  # type: ignore[attr-defined]
                # Map to SECTION_WEIGHTS keys
                section_key = f"Analyst {report_section.title.title()}"
                sections.append(
                    Section(
                        item=section_key,
                        title=report_section.title.title(),
                        content=report_section.content,
                        start_pos=0,
                        end_pos=len(report_section.content),
                    )
                )

        else:
            # Generic PDF parsing
            import fitz

            doc = fitz.open(pdf_file)
            full_text = "\n".join(str(page.get_text()) for page in doc)
            doc.close()

            from src.rag.parser import Section

            sections = [
                Section(
                    item="Full Document",
                    title="Document Content",
                    content=full_text,
                    start_pos=0,
                    end_pos=len(full_text),
                )
            ]

        if not sections:
            console.print("    [yellow]No content extracted[/yellow]")
            return False

        # Summarize using the summarizer's approach
        from src.summarizer.summarizer import FilingSummary

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Summarizing...", total=None)

            section_summaries = []
            total_in = 0
            total_out = 0

            for i, section in enumerate(sections):
                progress.update(task, description=f"[{i+1}/{len(sections)}] {section.item}...")
                summary = summarizer.summarize_section(
                    section,
                    filing_context=f"{ticker} {doc_type.title()} ({file_date})",
                )
                section_summaries.append(summary)
                total_in += summary.input_tokens
                total_out += summary.output_tokens

            # Create executive summary without investment thesis for non-10-K docs
            progress.update(task, description="Creating summary...")
            exec_summary, exec_in, exec_out = summarizer.create_executive_summary(
                section_summaries,
                filing_context=f"{ticker} {doc_type.title()} ({file_date})",
                include_thesis=False,
            )
            total_in += exec_in
            total_out += exec_out

            progress.update(task, description="[green]Done[/green]")

        # Save as FilingSummary
        result = FilingSummary(
            ticker=ticker,
            filing_type=doc_type.title(),
            filing_date=file_date,
            source_path=str(pdf_file),
            executive_summary=exec_summary,
            section_summaries=section_summaries,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
        result.save(output_path)
        console.print(f"    Saved to: {output_path}")
        return True

    except Exception as e:
        console.print(f"    [red]Error: {e}[/red]")
        return False


@app.command()
def summarize(
    company: Annotated[str, typer.Argument(help="Company identifier")],
    source: Annotated[SourceType, typer.Option("--source", "-s", help="Document source to summarize")] = SourceType.SEC,
    doc_type: Annotated[
        str | None,
        typer.Option("--doc-type", "-d", help="Summarize specific doc type (e.g., 10-K)"),
    ] = None,
    latest: Annotated[bool, typer.Option("--latest", "-l", help="Only summarize the latest of each type")] = False,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Summarize a specific file")] = None,
) -> None:
    """Generate summaries of company documents.

    Sources:
    - sec: SEC filings from data/{ticker}/ manifest
    - transcripts: Earnings call PDFs from transcripts/{ticker}/
    - analyst: Bank research PDFs from analyst/{ticker}/
    - presentations: Investor day PDFs from presentations/{ticker}/
    - misc: Other materials from misc/{ticker}/
    - all: Summarize all available sources

    Examples:
        pitch summarize AAPL                         # SEC filings (default)
        pitch summarize AAPL -s transcripts          # Earnings transcripts
        pitch summarize AAPL -s analyst --latest     # Latest analyst report
        pitch summarize AAPL -s all                  # Everything
        pitch summarize AAPL -f transcripts/AAPL/q3_2024.pdf
    """

    ticker = company.upper()
    summarizer = Summarizer(processed_dir=PROCESSED_DIR)

    # Handle specific file summarization
    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)

        console.print(f"[bold]Summarizing file: {file.name}[/bold]")
        _summarize_pdf_file(file, ticker, "misc", summarizer, console)
        console.print("\n[bold green]Summarization complete![/bold green]")
        return

    # Determine which sources to process
    sources_to_process = []
    if source == SourceType.ALL:
        for src in [
            SourceType.SEC,
            SourceType.TRANSCRIPTS,
            SourceType.ANALYST,
            SourceType.PRESENTATIONS,
            SourceType.CONFERENCES,
            SourceType.MISC,
        ]:
            sources_to_process.append(src)
    else:
        sources_to_process.append(source)

    total_summarized = 0

    for src in sources_to_process:
        if src == SourceType.SEC:
            count = _summarize_sec_filings(ticker, doc_type, latest, summarizer, console)
            total_summarized += count
        elif src == SourceType.TRANSCRIPTS:
            count = _summarize_source_pdfs(ticker, TRANSCRIPTS_DIR, "transcript", latest, summarizer, console)
            total_summarized += count
        elif src == SourceType.ANALYST:
            count = _summarize_source_pdfs(ticker, ANALYST_DIR, "analyst", latest, summarizer, console)
            total_summarized += count
        elif src == SourceType.PRESENTATIONS:
            count = _summarize_source_pdfs(ticker, PRESENTATIONS_DIR, "presentation", latest, summarizer, console)
            total_summarized += count
        elif src == SourceType.CONFERENCES:
            count = _summarize_source_pdfs(ticker, CONFERENCES_DIR, "conference", latest, summarizer, console)
            total_summarized += count
        elif src == SourceType.MISC:
            count = _summarize_source_pdfs(ticker, MISC_DIR, "misc", latest, summarizer, console)
            total_summarized += count

    if total_summarized > 0:
        console.print(f"\n[bold green]Summarization complete! ({total_summarized} documents)[/bold green]")
    else:
        console.print("\n[yellow]No documents found to summarize.[/yellow]")
