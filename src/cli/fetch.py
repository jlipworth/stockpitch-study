"""Fetch command for downloading SEC filings."""

import os
from datetime import datetime
from typing import Annotated

import typer

from src.filings import SUPPORTED_FILINGS, Manifest, SECFetcher

from . import DATA_DIR, app, console, parse_date


@app.command()
def fetch(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g., AAPL)")],
    types: Annotated[str, typer.Option("--types", "-t", help="Filing types, comma-separated")] = "10-K,10-Q",
    years: Annotated[int, typer.Option("--years", "-y", help="Number of years (ignored if dates provided)")] = 0,
    start_date: Annotated[str, typer.Option("--start-date", "-s", help="Start date (YYYY-MM-DD)")] = "",
    end_date: Annotated[str, typer.Option("--end-date", "-e", help="End date (YYYY-MM-DD)")] = "",
    since: Annotated[bool, typer.Option("--since", help="Fetch filings since last manifest update")] = False,
    all_types: Annotated[bool, typer.Option("--all", "-a", help="Fetch all supported filing types")] = False,
    cik: Annotated[str, typer.Option("--cik", help="CIK number (overrides ticker lookup)")] = "",
) -> None:
    """Fetch SEC filings for a company.

    Examples:
        pitch fetch AAPL -t "10-K" -y 3
        pitch fetch AAPL -t "10-K,10-Q" --start-date 2020-01-01 --end-date 2023-12-31
        pitch fetch AAPL --all --start-date 2022-01-01
        pitch fetch AAPL --since                  # Fetch new filings since last update
        pitch fetch BLK --cik 1364742 -t "10-K" --start-date 2018-01-01  # Use specific CIK
    """
    # Determine filing types
    if all_types:
        filing_types = list(SUPPORTED_FILINGS)
    else:
        filing_types = [t.strip() for t in types.split(",")]  # type: ignore[misc]

    # Validate filing types
    for ft in filing_types:
        if ft not in SUPPORTED_FILINGS:
            console.print(f"[red]Unsupported filing type: {ft}[/red]")
            console.print(f"Supported types: {', '.join(SUPPORTED_FILINGS)}")
            raise typer.Exit(1)

    # Handle --since flag: use manifest's last_updated as start_date
    if since:
        if start_date or end_date:
            console.print("[red]Error: --since cannot be used with --start-date or --end-date[/red]")
            raise typer.Exit(1)

        manifest_path = DATA_DIR / ticker.upper() / "manifest.json"
        if manifest_path.exists():
            manifest = Manifest.load(manifest_path)
            if manifest.last_updated:
                # Parse the ISO timestamp and use the date
                last_updated_dt = datetime.fromisoformat(manifest.last_updated)
                start_date = last_updated_dt.strftime("%Y-%m-%d")
                console.print(f"[bold]Fetching filings since: {start_date}[/bold]")
            else:
                console.print("[yellow]Warning: Manifest has no last_updated. Fetching all.[/yellow]")
        else:
            console.print(f"[yellow]No manifest found for {ticker}. Fetching with defaults.[/yellow]")

    # Parse dates if provided
    parsed_start = parse_date(start_date) if start_date else None
    parsed_end = parse_date(end_date) if end_date else None

    # Validate date range
    if parsed_start and parsed_end and parsed_start > parsed_end:
        console.print("[red]Error: start-date must be before end-date[/red]")
        raise typer.Exit(1)

    # Convert sentinel values to None
    years_param = years if years > 0 else None

    # Get user agent from environment
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        console.print("[yellow]Warning: SEC_USER_AGENT not set in .env[/yellow]")
        console.print("SEC recommends identifying yourself. Set SEC_USER_AGENT='Name email@example.com'")

    # Create fetcher and run
    fetcher = SECFetcher(data_dir=DATA_DIR, user_agent=user_agent)

    try:
        downloaded = fetcher.fetch(
            ticker=ticker,
            filing_types=filing_types,  # type: ignore
            years=years_param,
            start_date=parsed_start,
            end_date=parsed_end,
            cik=cik if cik else None,
        )
        dest = DATA_DIR / ticker.upper()
        console.print(f"\n[bold]Summary:[/bold] {len(downloaded)} filings downloaded to {dest}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
