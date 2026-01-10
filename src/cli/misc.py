"""Miscellaneous commands: notes, process, inventory, wrap."""

import os
from datetime import datetime as dt
from pathlib import Path
from typing import Annotated

import typer

from src.filings import SUPPORTED_FILINGS, Manifest, SECFetcher
from src.notes import NotesConverter
from src.rag import DEFAULT_MODEL, EmbeddingModel, SECFilingParser, VectorStore
from src.summarizer import Summarizer

from . import (
    ANALYST_DIR,
    CONFERENCES_DIR,
    DATA_DIR,
    INDEX_DIR,
    MISC_DIR,
    NOTES_DIR,
    PRESENTATIONS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TRANSCRIPTS_DIR,
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    app,
    console,
)


@app.command()
def notes(
    pdf_path: Annotated[Path, typer.Argument(help="Path to handwritten notes PDF")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output markdown path")] = None,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution for PDF rendering (higher = better quality)")] = 200,
) -> None:
    """Convert handwritten notes PDF to markdown using Claude vision."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        console.print(f"[red]File not found: {pdf_path}[/red]")
        raise typer.Exit(1)

    if not pdf_path.suffix.lower() == ".pdf":
        console.print(f"[red]File must be a PDF: {pdf_path}[/red]")
        raise typer.Exit(1)

    # Default output path: notes/{filename}.md
    output_path = output or (NOTES_DIR / pdf_path.stem).with_suffix(".md")

    console.print("[bold]Converting handwritten notes[/bold]")
    console.print(f"  Input:  {pdf_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  DPI:    {dpi}")

    try:
        converter = NotesConverter(dpi=dpi)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting pages...", total=None)

            def update_progress(page_num: int, total: int) -> None:
                progress.update(task, description=f"Transcribing page {page_num}/{total}...")

            result = converter.convert(pdf_path, progress_callback=update_progress)
            progress.update(task, description="[green]Conversion complete[/green]")

        # Save the result
        result.save(output_path)

        # Summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Pages converted: {result.num_pages}")
        console.print(f"  Tokens used: {result.total_input_tokens} in / {result.total_output_tokens} out")
        console.print(f"  Output saved to: {output_path}")

        # Preview first 500 chars
        console.print("\n[bold]Preview:[/bold]")
        preview = result.markdown[:500] + "..." if len(result.markdown) > 500 else result.markdown
        console.print(f"[dim]{preview}[/dim]")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def process(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol")],
    types: Annotated[str, typer.Option("--types", "-t", help="Filing types, comma-separated")] = "10-K,10-Q",
    years: Annotated[int, typer.Option("--years", "-y", help="Number of years to fetch")] = 2,
    skip_summarize: Annotated[bool, typer.Option("--skip-summarize", help="Skip summarization step")] = False,
) -> None:
    """Full pipeline: fetch, index, and summarize for a company.

    Runs the complete workflow:
    1. Fetch SEC filings (10-K, 10-Q by default)
    2. Build/update vector index
    3. Generate summaries of key filings

    Examples:
        pitch process AAPL
        pitch process MSFT -t "10-K" -y 3
        pitch process NVDA --skip-summarize
    """
    ticker = ticker.upper()
    console.print(f"\n[bold blue]{'=' * 50}[/bold blue]")
    console.print(f"[bold blue] Stock Pitch Pipeline: {ticker}[/bold blue]")
    console.print(f"[bold blue]{'=' * 50}[/bold blue]\n")

    filing_types = [t.strip() for t in types.split(",")]

    # Validate filing types
    for ft in filing_types:
        if ft not in SUPPORTED_FILINGS:
            console.print(f"[red]Unsupported filing type: {ft}[/red]")
            console.print(f"Supported types: {', '.join(SUPPORTED_FILINGS)}")
            raise typer.Exit(1)

    # Get user agent
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        console.print("[yellow]Warning: SEC_USER_AGENT not set in .env[/yellow]")

    # ========== STEP 1: FETCH ==========
    console.print("[bold cyan]Step 1/3: Fetching SEC filings[/bold cyan]")
    console.print(f"  Types: {', '.join(filing_types)}")
    console.print(f"  Years: {years}")

    fetcher = SECFetcher(data_dir=DATA_DIR, user_agent=user_agent)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching filings...", total=None)

            downloaded = fetcher.fetch(
                ticker=ticker,
                filing_types=filing_types,  # type: ignore
                years=years,
            )

            progress.update(task, description=f"[green]Fetched {len(downloaded)} filings[/green]")

        console.print(f"  [green]✓[/green] Downloaded {len(downloaded)} filings\n")

    except Exception as e:
        console.print(f"[red]Error fetching filings: {e}[/red]")
        raise typer.Exit(1)

    # ========== STEP 2: INDEX ==========
    console.print("[bold cyan]Step 2/3: Building vector index[/bold cyan]")

    company_dir = DATA_DIR / ticker
    manifest_path = company_dir / "manifest.json"

    if not manifest_path.exists():
        console.print("[red]No manifest found after fetch. Something went wrong.[/red]")
        raise typer.Exit(1)

    manifest = Manifest.load(manifest_path)
    total_files = len(manifest.filings)
    console.print(f"  Filings to index: {total_files}")

    try:
        # Load model with spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading embedding model...", total=None)
            embedding_model = EmbeddingModel(model_name=DEFAULT_MODEL)
            _ = embedding_model.model  # Force load
            device_info = f"device={embedding_model.device}, batch_size={embedding_model.batch_size}"
            progress.update(task, description=f"[green]Loaded {DEFAULT_MODEL} ({device_info})[/green]")

        store = VectorStore(
            index_dir=INDEX_DIR,
            ticker=ticker,
            embedding_dim=embedding_model.embedding_dim,
            embedding_model=DEFAULT_MODEL,
        )

        parser = SECFilingParser()
        indexed_count = 0
        skipped_count = 0
        chunk_count = 0

        # Index with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[chunks]}[/cyan] chunks"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task("Indexing...", total=total_files, chunks=0)

            for accession, filing_meta in manifest.filings.items():
                file_path = DATA_DIR / filing_meta.file_path
                if not file_path.exists():
                    progress.advance(main_task)
                    continue

                # Skip if already indexed
                if store.is_file_indexed(str(file_path), filing_meta.file_hash):
                    skipped_count += 1
                    progress.advance(main_task)
                    continue

                progress.update(
                    main_task,
                    description=f"SEC {filing_meta.filing_type} {filing_meta.filing_date}",
                )

                try:
                    parsed = parser.parse_file(
                        file_path,
                        ticker=ticker,
                        filing_type=filing_meta.filing_type,
                        filing_date=filing_meta.filing_date,
                    )
                    chunks = parser.chunk_filing(parsed)

                    if chunks:
                        embedded = embedding_model.embed_chunks(chunks, show_progress=True)
                        store.add(
                            embedded,
                            source_file=str(file_path),
                            file_hash=filing_meta.file_hash,
                        )
                        indexed_count += 1
                        chunk_count += len(embedded)
                        progress.update(main_task, chunks=chunk_count)
                except Exception as e:
                    console.print(f"  [yellow]Warning: {e}[/yellow]")

                progress.advance(main_task)

            progress.update(main_task, description="[green]Complete[/green]")

        console.print(f"  [green]✓[/green] Indexed {indexed_count} filings ({chunk_count} chunks)")
        if skipped_count > 0:
            console.print(f"  [dim]Skipped: {skipped_count} (already indexed)[/dim]")
        console.print(f"  [green]✓[/green] Total chunks in store: {store.count()}\n")

    except Exception as e:
        console.print(f"[red]Error indexing: {e}[/red]")
        raise typer.Exit(1)

    # ========== STEP 3: SUMMARIZE ==========
    if skip_summarize:
        console.print("[dim]Step 3/3: Summarization skipped[/dim]\n")
    else:
        console.print("[bold cyan]Step 3/3: Generating summaries[/bold cyan]")

        # Find filings to summarize (skip Form 4)
        filings_to_summarize = [f for f in manifest.filings.values() if f.filing_type != "4"]

        # Sort by date and take latest of each type
        filings_to_summarize.sort(key=lambda f: f.filing_date, reverse=True)
        seen_types = set()
        latest_filings = []
        for f in filings_to_summarize:
            if f.filing_type not in seen_types:
                latest_filings.append(f)
                seen_types.add(f.filing_type)

        console.print(f"  Summarizing latest: {', '.join(seen_types)}")

        try:
            summarizer = Summarizer(processed_dir=PROCESSED_DIR)
            summaries_created = 0

            for filing_meta in latest_filings:
                file_path = DATA_DIR / filing_meta.file_path
                if not file_path.exists():
                    continue

                output_path = summarizer.get_output_path(ticker, filing_meta.filing_type, filing_meta.filing_date)

                # Skip if already summarized
                if output_path.exists():
                    console.print(f"  [dim]Skipping {filing_meta.filing_type} (exists)[/dim]")
                    continue

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Summarizing {filing_meta.filing_type}...", total=None)

                    def update_progress(current: int, total: int, section: str) -> None:
                        progress.update(task, description=f"[{current}/{total}] {section}...")

                    result = summarizer.summarize_filing(
                        file_path=file_path,
                        ticker=ticker,
                        filing_type=filing_meta.filing_type,
                        filing_date=filing_meta.filing_date,
                        progress_callback=update_progress,
                    )

                    result.save(output_path)
                    summaries_created += 1

                    progress.update(task, description=f"[green]{filing_meta.filing_type} complete[/green]")

            console.print(f"  [green]✓[/green] Created {summaries_created} summaries\n")

        except ValueError as e:
            console.print(f"[yellow]Summarization skipped: {e}[/yellow]\n")
        except Exception as e:
            console.print(f"[yellow]Summarization error: {e}[/yellow]\n")

    # ========== SUMMARY ==========
    console.print(f"[bold green]{'=' * 50}[/bold green]")
    console.print(f"[bold green] Pipeline complete for {ticker}![/bold green]")
    console.print(f"[bold green]{'=' * 50}[/bold green]")
    console.print(f"\nData:      {DATA_DIR / ticker}")
    console.print(f"Index:     {INDEX_DIR / ticker}")
    console.print(f"Summaries: {PROCESSED_DIR / ticker}")
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f'  pitch search {ticker} "revenue growth"')
    console.print(f'  pitch ask {ticker} "What are the main risks?"')
    console.print()


@app.command()
def inventory(
    company: Annotated[str | None, typer.Argument(help="Company identifier (optional for forked repos)")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output path for inventory file")] = Path(
        "MATERIALS.md"
    ),
) -> None:
    """Generate an inventory of all materials in the project.

    Creates a MATERIALS.md file with a quick reference list of:
    - SEC filings (from manifest)
    - Transcripts
    - Analyst reports
    - Presentations
    - Misc documents

    Examples:
        pitch inventory AAPL              # For template repo
        pitch inventory                   # For forked company repo
        pitch inventory -o docs/inv.md    # Custom output path
    """
    lines = []
    lines.append("# Materials Inventory")
    lines.append("")
    lines.append(f"*Generated: {dt.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    ticker = company.upper() if company else None

    # ========== SEC FILINGS ==========
    lines.append("## SEC Filings")
    lines.append("")

    # Check for manifest in data/{ticker}/ or filings/
    manifest_paths = []
    if ticker:
        manifest_paths.append(DATA_DIR / ticker / "manifest.json")
    else:
        # Forked repo: look in data/ or filings/
        for d in [DATA_DIR, PROJECT_ROOT / "filings"]:
            if d.exists():
                for p in d.glob("*/manifest.json"):
                    manifest_paths.append(p)

    filings_found = False
    for manifest_path in manifest_paths:
        if manifest_path.exists():
            manifest = Manifest.load(manifest_path)
            filings_found = True

            lines.append(f"**{manifest.ticker}** ({manifest.company_name})")
            lines.append(f"*Last updated: {manifest.last_updated[:10]}*")
            lines.append("")
            lines.append("| Type | Filing Date | Accession |")
            lines.append("|------|-------------|-----------|")

            # Sort by date descending
            sorted_filings = sorted(
                manifest.filings.values(),
                key=lambda f: f.filing_date,
                reverse=True,
            )

            for filing in sorted_filings:
                lines.append(f"| {filing.filing_type} | {filing.filing_date} | {filing.accession_number} |")

            lines.append("")

    if not filings_found:
        lines.append("*No SEC filings found. Run `pitch fetch <ticker>` first.*")
        lines.append("")

    # ========== TRANSCRIPTS ==========
    lines.append("## Transcripts")
    lines.append("")

    transcripts_dir = TRANSCRIPTS_DIR / ticker if ticker else TRANSCRIPTS_DIR
    transcript_files = []
    if transcripts_dir.exists():
        for pdf_file in transcripts_dir.rglob("*.pdf"):
            rel_path = pdf_file.relative_to(PROJECT_ROOT) if PROJECT_ROOT in pdf_file.parents else pdf_file
            mtime = dt.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d")
            transcript_files.append((pdf_file.name, str(rel_path), mtime))

    if transcript_files:
        lines.append("| File | Path | Modified |")
        lines.append("|------|------|----------|")
        for name, path, mtime in sorted(transcript_files, key=lambda x: x[0]):
            lines.append(f"| {name} | {path} | {mtime} |")
        lines.append("")
    else:
        lines.append("*No transcripts found.*")
        lines.append("")

    # ========== ANALYST REPORTS ==========
    lines.append("## Analyst Reports")
    lines.append("")

    analyst_dir = ANALYST_DIR / ticker if ticker else ANALYST_DIR
    analyst_files = []
    if analyst_dir.exists():
        for pdf_file in analyst_dir.rglob("*.pdf"):
            rel_path = pdf_file.relative_to(PROJECT_ROOT) if PROJECT_ROOT in pdf_file.parents else pdf_file
            mtime = dt.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d")
            analyst_files.append((pdf_file.name, str(rel_path), mtime))

    if analyst_files:
        lines.append("| File | Path | Modified |")
        lines.append("|------|------|----------|")
        for name, path, mtime in sorted(analyst_files, key=lambda x: x[0]):
            lines.append(f"| {name} | {path} | {mtime} |")
        lines.append("")
    else:
        lines.append("*No analyst reports found.*")
        lines.append("")

    # ========== PRESENTATIONS ==========
    lines.append("## Presentations")
    lines.append("")

    pres_dir = PRESENTATIONS_DIR / ticker if ticker else PRESENTATIONS_DIR
    pres_files = []
    if pres_dir.exists():
        for pdf_file in pres_dir.rglob("*.pdf"):
            rel_path = pdf_file.relative_to(PROJECT_ROOT) if PROJECT_ROOT in pdf_file.parents else pdf_file
            mtime = dt.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d")
            pres_files.append((pdf_file.name, str(rel_path), mtime))

    if pres_files:
        lines.append("| File | Path | Modified |")
        lines.append("|------|------|----------|")
        for name, path, mtime in sorted(pres_files, key=lambda x: x[0]):
            lines.append(f"| {name} | {path} | {mtime} |")
        lines.append("")
    else:
        lines.append("*No presentations found.*")
        lines.append("")

    # ========== CONFERENCES ==========
    lines.append("## Conference Transcripts")
    lines.append("")

    conf_dir = CONFERENCES_DIR / ticker if ticker else CONFERENCES_DIR
    conf_files = []
    if conf_dir.exists():
        for pdf_file in conf_dir.rglob("*.pdf"):
            rel_path = pdf_file.relative_to(PROJECT_ROOT) if PROJECT_ROOT in pdf_file.parents else pdf_file
            mtime = dt.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d")
            conf_files.append((pdf_file.name, str(rel_path), mtime))

    if conf_files:
        lines.append("| File | Path | Modified |")
        lines.append("|------|------|----------|")
        for name, path, mtime in sorted(conf_files, key=lambda x: x[0]):
            lines.append(f"| {name} | {path} | {mtime} |")
        lines.append("")
    else:
        lines.append("*No conference transcripts found.*")
        lines.append("")

    # ========== MISC ==========
    lines.append("## Misc Documents")
    lines.append("")

    misc_dir = MISC_DIR / ticker if ticker else MISC_DIR
    misc_files = []
    if misc_dir.exists():
        for pdf_file in misc_dir.rglob("*.pdf"):
            rel_path = pdf_file.relative_to(PROJECT_ROOT) if PROJECT_ROOT in pdf_file.parents else pdf_file
            mtime = dt.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d")
            misc_files.append((pdf_file.name, str(rel_path), mtime))

    if misc_files:
        lines.append("| File | Path | Modified |")
        lines.append("|------|------|----------|")
        for name, path, mtime in sorted(misc_files, key=lambda x: x[0]):
            lines.append(f"| {name} | {path} | {mtime} |")
        lines.append("")
    else:
        lines.append("*No misc documents found.*")
        lines.append("")

    # ========== SUMMARY ==========
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    total_sec = sum(len(Manifest.load(p).filings) for p in manifest_paths if p.exists())
    total_transcripts = len(transcript_files)
    total_analyst = len(analyst_files)
    total_pres = len(pres_files)
    total_conf = len(conf_files)
    total_misc = len(misc_files)
    total = total_sec + total_transcripts + total_analyst + total_pres + total_conf + total_misc

    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| SEC Filings | {total_sec} |")
    lines.append(f"| Transcripts | {total_transcripts} |")
    lines.append(f"| Analyst Reports | {total_analyst} |")
    lines.append(f"| Presentations | {total_pres} |")
    lines.append(f"| Conferences | {total_conf} |")
    lines.append(f"| Misc Documents | {total_misc} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    # Write the file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    console.print(f"[bold green]Materials inventory saved to: {output_path}[/bold green]")
    console.print(f"  SEC Filings: {total_sec}")
    console.print(f"  Transcripts: {total_transcripts}")
    console.print(f"  Analyst Reports: {total_analyst}")
    console.print(f"  Presentations: {total_pres}")
    console.print(f"  Conferences: {total_conf}")
    console.print(f"  Misc: {total_misc}")
    console.print(f"  [bold]Total: {total}[/bold]")


@app.command()
def wrap(
    file_path: Path = typer.Argument(..., help="Markdown file to reformat"),
    width: int = typer.Option(100, "--width", "-w", help="Maximum line width"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file (default: overwrite input)"),
    preview: bool = typer.Option(False, "--preview", "-p", help="Preview changes without writing"),
):
    """Wrap long lines in a markdown file for better readability.

    Reformats markdown files by wrapping long lines while preserving:
    - Headers
    - Tables
    - Code blocks
    - Bullet point indentation
    """
    from src.summarizer import wrap_markdown_lines

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    content = file_path.read_text(encoding="utf-8")
    wrapped = wrap_markdown_lines(content, width=width)

    if preview:
        console.print("[bold]Preview (first 2000 chars):[/bold]")
        console.print(wrapped[:2000])
        if len(wrapped) > 2000:
            console.print("[dim]...(truncated)[/dim]")
        return

    output_path = output or file_path
    output_path.write_text(wrapped, encoding="utf-8")
    console.print(f"[green]Reformatted {file_path} -> {output_path}[/green]")
    console.print(f"  Line width: {width}")
