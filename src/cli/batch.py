"""Batch processing commands for async summarization."""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from src.filings import Manifest
from src.summarizer.batch import BatchProcessor

from . import DATA_DIR, PROCESSED_DIR, Progress, SpinnerColumn, TextColumn, app, console


@app.command("batch-submit")
def batch_submit(
    company: Annotated[str, typer.Argument(help="Company identifier")],
    doc_type: Annotated[
        str, typer.Option("--doc-type", "-d", help="Document type (10-K, 10-Q, transcript, presentation)")
    ] = "10-K",
    filing_date: Annotated[str | None, typer.Option("--date", help="Specific filing date (YYYY-MM-DD)")] = None,
    file_path_arg: Annotated[
        str | None, typer.Option("--file", "-f", help="Direct path to file (for transcripts, presentations)")
    ] = None,
) -> None:
    """Submit a filing for batch summarization (50% cost savings).

    Batch processing runs asynchronously and can take up to 24 hours.
    Use 'pitch batch-status' to check progress.

    Examples:
        pitch batch-submit AAPL -d 10-K
        pitch batch-submit MSFT -d 10-Q --date 2024-03-31
        pitch batch-submit DT -d transcript --file transcripts/DT/earnings.pdf
    """
    import os

    ticker = company.upper()

    # Determine file path based on mode
    if file_path_arg:
        # Direct file mode
        file_path = Path(file_path_arg)
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            raise typer.Exit(1)
        # Use explicit date if provided, otherwise extract from filename
        if not filing_date:
            # Match "Aug 06, 2025" or "Aug-06-2025" patterns
            date_match = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2}),?\s*(\d{4})", file_path.stem)
            if date_match:
                month_str, day, year = date_match.groups()
                try:
                    parsed_date = datetime.strptime(f"{month_str} {day} {year}", "%b %d %Y")
                    filing_date = parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    filing_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d")
            else:
                iso_match = re.search(r"(\d{4})[- ](\d{2})[- ](\d{2})", file_path.stem)
                if iso_match:
                    filing_date = f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"
                else:
                    filing_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d")
        target_filing_type = doc_type
    else:
        # SEC filing mode via manifest
        company_dir = DATA_DIR / ticker

        if not company_dir.exists():
            console.print(f"[red]No data found for {ticker}. Run 'pitch fetch {ticker}' first.[/red]")
            raise typer.Exit(1)

        # Load manifest
        manifest_path = company_dir / "manifest.json"
        if not manifest_path.exists():
            console.print(f"[red]No manifest found for {ticker}.[/red]")
            raise typer.Exit(1)

        manifest = Manifest.load(manifest_path)

        # Find the filing to summarize
        target_filing = None
        for accession, filing_meta in manifest.filings.items():
            if filing_meta.filing_type != doc_type:
                continue
            if filing_date and filing_meta.filing_date != filing_date:
                continue
            if target_filing is None or filing_meta.filing_date > target_filing.filing_date:
                target_filing = filing_meta

        if not target_filing:
            console.print(f"[red]No {doc_type} filing found for {ticker}.[/red]")
            raise typer.Exit(1)

        file_path = DATA_DIR / target_filing.file_path
        if not file_path.exists():
            console.print(f"[red]File not found: {target_filing.file_path}[/red]")
            raise typer.Exit(1)

        target_filing_type = target_filing.filing_type
        filing_date = target_filing.filing_date

    console.print("[bold]Submitting batch job[/bold]")
    console.print(f"  Ticker: {ticker}")
    console.print(f"  Type: {target_filing_type} ({filing_date})")
    console.print(f"  File: {file_path.name}")

    try:
        processor = BatchProcessor(processed_dir=PROCESSED_DIR)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Submitting batch...", total=None)

            job = processor.submit_batch(
                file_path=file_path,
                ticker=ticker,
                filing_type=target_filing_type,
                filing_date=filing_date,
            )

            progress.update(task, description="[green]Submitted[/green]")

        console.print("\n[bold green]Batch job submitted![/bold green]")
        console.print(f"  Job ID: {job.batch_id}")
        console.print(f"  Requests: {job.request_count}")
        console.print(f"  Status: {job.status}")
        console.print(f"\n[dim]Check progress with: pitch batch-status {job.batch_id}[/dim]")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("batch-status")
def batch_status(
    batch_id: Annotated[str | None, typer.Argument(help="Batch job ID (omit to list all)")] = None,
    poll: Annotated[bool, typer.Option("--poll", "-p", help="Poll until complete")] = False,
    interval: Annotated[int, typer.Option("--interval", "-i", help="Poll interval in seconds")] = 60,
) -> None:
    """Check status of batch jobs.

    Examples:
        pitch batch-status                    # List all jobs
        pitch batch-status msgbatch_xxx       # Check specific job
        pitch batch-status msgbatch_xxx -p    # Poll until complete
    """
    try:
        processor = BatchProcessor(processed_dir=PROCESSED_DIR)

        if batch_id is None:
            # List all jobs
            jobs = processor.list_jobs()
            if not jobs:
                console.print("[yellow]No batch jobs found.[/yellow]")
                return

            # Sort by created_at descending (most recent first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            console.print(f"[bold]Batch Jobs ({len(jobs)} total)[/bold]\n")
            for job in jobs:
                status_color = {
                    "in_progress": "yellow",
                    "ended": "green",
                    "expired": "red",
                    "canceled": "dim",
                }.get(job.status, "white")

                # Show if results have been retrieved
                retrieved = "[green]✓[/green]" if job.results_path else "[dim]○[/dim]"

                # Parse created_at for display (show date + time)
                try:
                    created_dt = datetime.fromisoformat(job.created_at)
                    created_str = created_dt.strftime("%m-%d %H:%M")
                except (ValueError, TypeError):
                    created_str = "N/A"

                console.print(
                    f"  {retrieved} [{status_color}]{job.status:12}[/{status_color}] "
                    f"[dim]{created_str}[/dim]  "
                    f"{job.ticker} {job.filing_type} ({job.filing_date}) "
                    f"[dim]{job.batch_id}[/dim]"
                )
            return

        # Check specific job
        if poll:
            console.print(f"[bold]Polling batch job: {batch_id}[/bold]")
            console.print(f"  Interval: {interval}s\n")

            def progress_callback(status: str, details: dict) -> None:
                counts = details.get("request_counts", {})
                console.print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Status: {status} | "
                    f"Processing: {counts.get('processing', 0)} | "
                    f"Succeeded: {counts.get('succeeded', 0)} | "
                    f"Errored: {counts.get('errored', 0)}"
                )

            final_status = processor.poll_until_complete(
                batch_id,
                poll_interval=interval,
                progress_callback=progress_callback,
            )

            console.print(f"\n[bold green]Batch complete![/bold green] Status: {final_status}")
            console.print(f"[dim]Retrieve results with: pitch batch-results {batch_id}[/dim]")
        else:
            status, details = processor.check_status(batch_id)

            console.print(f"[bold]Batch Status: {batch_id}[/bold]\n")
            console.print(f"  Status: {status}")
            console.print(f"  Created: {details.get('created_at', 'N/A')}")

            if details.get("ended_at"):
                console.print(f"  Ended: {details['ended_at']}")

            counts = details.get("request_counts", {})
            console.print("\n  [bold]Request Counts:[/bold]")
            console.print(f"    Processing: {counts.get('processing', 0)}")
            console.print(f"    Succeeded:  {counts.get('succeeded', 0)}")
            console.print(f"    Errored:    {counts.get('errored', 0)}")
            console.print(f"    Canceled:   {counts.get('canceled', 0)}")
            console.print(f"    Expired:    {counts.get('expired', 0)}")

            if status == "ended":
                console.print(f"\n[dim]Retrieve results with: pitch batch-results {batch_id}[/dim]")
            elif status == "in_progress":
                console.print(f"\n[dim]Poll for completion with: pitch batch-status {batch_id} -p[/dim]")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("batch-results")
def batch_results(
    batch_id: Annotated[str, typer.Argument(help="Batch job ID")],
    no_save: Annotated[bool, typer.Option("--no-save", help="Don't save summary file")] = False,
) -> None:
    """Retrieve and process results from a completed batch job.

    Results are saved to the processed/{ticker}/ directory.

    Examples:
        pitch batch-results msgbatch_xxx
    """
    try:
        processor = BatchProcessor(processed_dir=PROCESSED_DIR)

        console.print(f"[bold]Retrieving batch results: {batch_id}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Retrieving results...", total=None)

            result = processor.retrieve_results(
                batch_id=batch_id,
                save_summary=not no_save,
            )

            progress.update(task, description="[green]Complete[/green]")

        if result is None:
            console.print("[yellow]Batch not complete or no results available.[/yellow]")
            console.print("[dim]Check status with: pitch batch-status {batch_id}[/dim]")
            return

        console.print("\n[bold green]Results retrieved![/bold green]")
        console.print(f"  Ticker: {result.ticker}")
        console.print(f"  Filing: {result.filing_type} ({result.filing_date})")
        console.print(f"  Sections: {len(result.section_summaries)}")
        console.print(f"  Tokens: {result.total_input_tokens} in / {result.total_output_tokens} out")

        if not no_save:
            output_path = processor.summarizer.get_output_path(result.ticker, result.filing_type, result.filing_date)
            console.print(f"  Saved to: {output_path}")

        # Preview executive summary
        console.print("\n[bold]Executive Summary Preview:[/bold]")
        preview = (
            result.executive_summary[:500] + "..." if len(result.executive_summary) > 500 else result.executive_summary
        )
        console.print(f"[dim]{preview}[/dim]")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("batch-results-all")
def batch_results_all(
    no_save: Annotated[bool, typer.Option("--no-save", help="Don't save summary files")] = False,
    max_concurrent: Annotated[int, typer.Option("--concurrent", "-c", help="Max concurrent API calls")] = 5,
) -> None:
    """Retrieve all unretrieved batch results in parallel.

    Finds all completed batch jobs that haven't been retrieved yet and
    processes them with parallel executive summary generation.

    This is much faster than running batch-results multiple times.

    Examples:
        pitch batch-results-all              # Retrieve all pending
        pitch batch-results-all -c 3         # Limit concurrency
    """
    import time

    try:
        processor = BatchProcessor(processed_dir=PROCESSED_DIR)

        # Find unretrieved jobs
        all_jobs = processor.list_jobs(refresh=True)
        pending_ids = [job.batch_id for job in all_jobs if job.status == "ended" and job.results_path is None]

        if not pending_ids:
            console.print("[yellow]No unretrieved batch results found.[/yellow]")
            return

        console.print(f"[bold]Retrieving {len(pending_ids)} batch results in parallel[/bold]\n")

        # Track progress
        status_map: dict[str, str] = {}

        def progress_callback(batch_id: str, status: str) -> None:
            status_map[batch_id] = status
            # Find job info for display
            job = processor.tracker.get_job(batch_id)
            if job:
                status_display = {
                    "collecting": "[yellow]collecting[/yellow]",
                    "generating_summary": "[cyan]generating summary[/cyan]",
                    "complete": "[green]complete[/green]",
                }.get(status, status)
                console.print(f"  {job.filing_type} ({job.filing_date}): {status_display}")

        start_time = time.time()

        results = processor.retrieve_results_parallel(
            batch_ids=pending_ids,
            save_summary=not no_save,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback,
        )

        elapsed = time.time() - start_time

        console.print(f"\n[bold green]Retrieved {len(results)} results in {elapsed:.1f}s[/bold green]")

        for result in results:
            console.print(f"  {result.filing_type} ({result.filing_date}): {len(result.section_summaries)} sections")
            if not no_save:
                output_path = processor.summarizer.get_output_path(
                    result.ticker, result.filing_type, result.filing_date
                )
                console.print(f"    → {output_path}")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("batch-resubmit")
def batch_resubmit(
    batch_id: Annotated[str, typer.Argument(help="Batch job ID to resubmit truncated sections from")],
    multiplier: Annotated[float, typer.Option("--multiplier", "-m", help="Token limit multiplier")] = 1.5,
    source_file: Annotated[Path | None, typer.Option("--file", "-f", help="Source file path (for older jobs)")] = None,
) -> None:
    """Resubmit truncated sections from a batch job with higher token limits.

    When a batch job completes with truncated sections (hit max_tokens limit),
    use this command to resubmit just those sections with increased limits.

    For older batch jobs that don't have source_path tracked, use --file to specify.

    Examples:
        pitch batch-resubmit msgbatch_xxx           # Resubmit with 1.5x tokens
        pitch batch-resubmit msgbatch_xxx -m 2.0    # Resubmit with 2x tokens
        pitch batch-resubmit msgbatch_xxx -f transcripts/DT/q4_2025.pdf
    """
    try:
        processor = BatchProcessor(processed_dir=PROCESSED_DIR)

        # Get job info
        job = processor.tracker.get_job(batch_id)
        if not job:
            console.print(f"[red]Job {batch_id} not found in tracker[/red]")
            raise typer.Exit(1)

        if job.status != "ended":
            console.print(f"[yellow]Job {batch_id} has not completed (status: {job.status})[/yellow]")
            raise typer.Exit(1)

        console.print(f"[bold]Checking batch for truncated sections: {batch_id}[/bold]")
        console.print(f"  Filing: {job.ticker} {job.filing_type} ({job.filing_date})")

        # Check for existing pending resubmit jobs for same filing
        existing_jobs = processor.tracker.load_jobs()
        pending_for_same_filing = [
            j
            for j in existing_jobs
            if (
                j.ticker == job.ticker
                and j.filing_type == job.filing_type
                and j.filing_date == job.filing_date
                and j.status in ("in_progress", "validating")
                and j.batch_id != batch_id
            )  # Exclude the original batch
        ]

        if pending_for_same_filing:
            console.print("\n[yellow]⚠️  Warning: There's already a pending job for this filing:[/yellow]")
            for pending in pending_for_same_filing:
                console.print(f"   Batch ID: {pending.batch_id}")
                console.print(f"   Status: {pending.status}")
                console.print(f"   Created: {pending.created_at}")

            if not typer.confirm("\nContinue anyway?"):
                raise typer.Exit(0)

        # First retrieve results to identify truncated sections
        result = processor.retrieve_results(batch_id=batch_id, save_summary=False)

        if result is None:
            console.print("[red]Failed to retrieve batch results[/red]")
            raise typer.Exit(1)

        # Check for truncated sections
        truncated = [s for s in result.section_summaries if s.truncated]
        if not truncated:
            console.print("[green]No truncated sections found - nothing to resubmit.[/green]")
            return

        console.print(f"\n[yellow]Found {len(truncated)} truncated section(s):[/yellow]")
        for section in truncated:
            console.print(f"  - {section.section_id}")

        # If source_file provided, update job with it
        if source_file:
            if not source_file.exists():
                console.print(f"[red]Source file not found: {source_file}[/red]")
                raise typer.Exit(1)
            job.source_path = str(source_file)
            # Update the tracker with the source path
            processor.tracker.update_job(batch_id, source_path=str(source_file))
            console.print(f"  Using source: {source_file}")

        # Resubmit
        console.print(f"\n[bold]Resubmitting with {multiplier}x token limit...[/bold]")
        new_job = processor.resubmit_truncated(batch_id, result, token_multiplier=multiplier)

        if new_job:
            console.print(f"\n[green]✓ New batch submitted: {new_job.batch_id}[/green]")
            console.print(f"  Requests: {new_job.request_count}")
            console.print("[dim]Check status with: pitch batch-status[/dim]")
        else:
            console.print("[yellow]Resubmission failed - check messages above[/yellow]")
            raise typer.Exit(1)

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
