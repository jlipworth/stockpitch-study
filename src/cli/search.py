"""Search and ask commands for querying indexed documents."""

import logging
from typing import Annotated

import typer

from src.rag import SearchConfig, Searcher, render_table_as_markdown

from . import (
    INDEX_DIR,
    PROJECT_ROOT,
    Progress,
    SpinnerColumn,
    TextColumn,
    app,
    console,
    rich_escape,
)

logger = logging.getLogger(__name__)


def _get_fiscal_info(ticker: str) -> str:
    """Extract fiscal calendar info from CLAUDE.md if available."""
    claude_md = PROJECT_ROOT / "CLAUDE.md"
    if not claude_md.exists():
        return ""

    try:
        content = claude_md.read_text()
        # Look for fiscal calendar section
        if "Fiscal Calendar" in content:
            start = content.find("**Fiscal Calendar")
            if start != -1:
                end = content.find("\n\n", start)
                if end == -1:
                    end = start + 500
                return content[start:end].strip()
    except (OSError, UnicodeDecodeError):
        pass  # Silently ignore if file can't be read

    return ""


@app.command()
def search(
    company: Annotated[str, typer.Argument(help="Company identifier")],
    query: Annotated[str, typer.Argument(help="Search query")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", "--limit", help="Number of results to return")] = 10,
    doc_type: Annotated[
        str | None, typer.Option("--doc-type", "-d", help="Filter by document type (e.g., 10-K)")
    ] = None,
    section: Annotated[str | None, typer.Option("--section", "-s", help="Filter by section (e.g., 'Item 1A')")] = None,
    mode: Annotated[str, typer.Option("--mode", "-m", help="Search mode: hybrid, vector, or fts")] = "hybrid",
    rerank: Annotated[
        bool, typer.Option("--rerank/--no-rerank", "-r", help="Rerank results with cross-encoder (default: on)")
    ] = True,
    context: Annotated[
        str,
        typer.Option(
            "--context",
            "-c",
            help="Context expansion: none (chunk only), section (full parent section), window (chunk + neighbors)",
        ),
    ] = "none",
    max_chars: Annotated[int, typer.Option("--max-chars", help="Max characters per result (0 for unlimited)")] = 500,
    table: Annotated[
        bool | None,
        typer.Option(
            "--table/--no-table",
            "-t",
            help="Filter to only chunks containing tables (--table) or exclude tables (--no-table)",
        ),
    ] = None,
    table_type: Annotated[
        str | None,
        typer.Option(
            "--table-type",
            help="Filter by table type: financial_statement, compensation, comparison, other (implies --table)",
        ),
    ] = None,
) -> None:
    """Search indexed documents with hybrid search.

    Reranking is enabled by default for better precision. Use --no-rerank to disable.

    Context expansion modes:
    - none: Show only the matching chunk (default, 500 chars)
    - section: Show full parent section (deduplicates results from same section)
    - window: Show chunk + 2 neighboring chunks

    Table filtering:
    - --table: Show only chunks containing tables
    - --no-table: Exclude chunks containing tables
    - --table-type TYPE: Filter by specific table type (implies --table)
    """
    ticker = company.upper()

    # Validate mode
    if mode not in ("hybrid", "vector", "fts"):
        console.print(f"[red]Invalid mode: {mode}. Use hybrid, vector, or fts.[/red]")
        raise typer.Exit(1)

    # Validate context
    if context not in ("none", "section", "window"):
        console.print(f"[red]Invalid context: {context}. Use none, section, or window.[/red]")
        raise typer.Exit(1)

    # Validate table_type if provided
    valid_table_types = ("financial_statement", "compensation", "comparison", "other")
    if table_type is not None and table_type not in valid_table_types:
        console.print(f"[red]Invalid table type: {table_type}. " f"Use one of: {', '.join(valid_table_types)}[/red]")
        raise typer.Exit(1)

    # Determine content_type filter based on --table flag
    # --table-type implies --table
    content_type_filter: str | None = None
    if table_type is not None:
        # --table-type implies filtering to tables only
        content_type_filter = "table"
    elif table is True:
        content_type_filter = "table"
    elif table is False:
        content_type_filter = "text"
    # If table is None (default), no content_type filter is applied

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading search index...", total=None)

            config = SearchConfig(mode=mode, top_k=top_k, rerank=rerank)  # type: ignore
            searcher = Searcher(
                index_dir=INDEX_DIR,
                ticker=ticker,
                config=config,
            )

            if rerank:
                progress.update(task, description="Loading reranker model...")
                _ = searcher.reranker.model  # Pre-load reranker

            progress.update(task, description="Searching...")

            results = searcher.search(
                query=query,
                filing_type=doc_type,
                section=section,
                content_type=content_type_filter,
                table_type=table_type,
            )

            # Apply context expansion if requested
            expanded_results = None
            if context == "section" and results:
                progress.update(task, description="Expanding with parent sections...")
                expanded_results = searcher.expand_results_with_parent(results, deduplicate_sections=True)
            elif context == "window" and results:
                progress.update(task, description="Expanding with context window...")
                expanded_results = [(r, searcher._get_window_context(r, window=2), "window") for r in results]

            progress.update(task, description="[green]Search complete[/green]")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Display results
    console.print(f"\n[bold]Results for:[/bold] {query}")
    rerank_info = " + rerank" if rerank else ""
    context_info = f" | Context: {context}" if context != "none" else ""
    # Build table filter info string
    table_info = ""
    if table_type:
        table_info = f" | Table: {table_type}"
    elif table is True:
        table_info = " | Tables only"
    elif table is False:
        table_info = " | No tables"
    result_count = len(expanded_results) if expanded_results else len(results)
    console.print(f"[dim]Mode: {mode}{rerank_info}{context_info}{table_info} | Found: {result_count} results[/dim]\n")

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if expanded_results:
        # Display expanded results
        for i, (result, content, exp_mode) in enumerate(expanded_results, 1):
            mode_label = "[full section]" if exp_mode == "section" else "[window]"
            console.print(
                f"[bold cyan]{i}.[/bold cyan] "
                f"[bold]{result.filing_type}[/bold] {result.filing_date} - "
                f"[italic]{result.section}[/italic] "
                f"[dim](score: {result.score:.3f})[/dim] "
                f"[green]{mode_label}[/green]"
            )

            # Apply max_chars limit (0 = unlimited)
            if max_chars > 0 and len(content) > max_chars:
                text = content[:max_chars] + "..."
            else:
                text = content
            console.print(f"   {rich_escape(text)}\n")
    else:
        # Display basic results
        for i, result in enumerate(results, 1):
            # Check if this result has a table with HTML available
            has_table_display = result.has_table and result.table_html
            table_indicator = " [green][table][/green]" if has_table_display else ""

            console.print(
                f"[bold cyan]{i}.[/bold cyan] "
                f"[bold]{result.filing_type}[/bold] {result.filing_date} - "
                f"[italic]{result.section}[/italic] "
                f"[dim](score: {result.score:.3f})[/dim]"
                f"{table_indicator}"
            )

            if has_table_display:
                # Render table as clean markdown grid
                try:
                    table_md = render_table_as_markdown(result.table_html)
                    if table_md.strip():
                        # Show caption if available
                        if result.table_caption:
                            console.print(f"   [bold]{result.table_caption}[/bold]")
                        # Indent each line of the table
                        for line in table_md.split("\n"):
                            console.print(f"   {line}")
                        console.print()
                    else:
                        # Fallback to text if markdown render fails
                        if max_chars > 0 and len(result.text) > max_chars:
                            text = result.text[:max_chars] + "..."
                        else:
                            text = result.text
                        console.print(f"   {rich_escape(text)}\n")
                except Exception as e:
                    # Fallback to text on any error
                    logger.debug("Markdown table render failed, using text fallback: %s", e)
                    if max_chars > 0 and len(result.text) > max_chars:
                        text = result.text[:max_chars] + "..."
                    else:
                        text = result.text
                    console.print(f"   {rich_escape(text)}\n")
            else:
                # Apply max_chars limit for non-table results
                if max_chars > 0 and len(result.text) > max_chars:
                    text = result.text[:max_chars] + "..."
                else:
                    text = result.text
                console.print(f"   {rich_escape(text)}\n")


@app.command()
def ask(
    company: Annotated[str, typer.Argument(help="Company identifier")],
    question: Annotated[str, typer.Argument(help="Question to answer")],
    doc_type: Annotated[
        str | None, typer.Option("--doc-type", "-d", help="Filter by document type (e.g., 10-K)")
    ] = None,
    depth: Annotated[str, typer.Option("--depth", help="Search depth: fast (default), medium, deep")] = "fast",
    show_sources: Annotated[bool, typer.Option("--sources", "-s", help="Show source citations")] = True,
    rerank: Annotated[
        bool, typer.Option("--rerank/--no-rerank", "-r", help="Rerank results with cross-encoder (default: on)")
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show search details")] = False,
) -> None:
    """Ask a question and get an AI-generated answer with sources.

    Depth levels:
    - fast: Single search, no query decomposition (default)
    - medium: 3 sub-queries, 2 hops max, can read files
    - deep: 5 sub-queries, 3 hops max, more file reads

    Reranking is enabled by default for better precision. Use --no-rerank to disable.
    """
    from src.rag.agent import Depth, create_agent

    ticker = company.upper()

    # Validate depth
    try:
        depth_level = Depth(depth.lower())
    except ValueError:
        console.print(f"[red]Invalid depth: {depth}. Use fast, medium, or deep.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Question about {ticker}:[/bold] {question}")
    console.print(f"[dim]Depth: {depth_level.value} | Rerank: {rerank}[/dim]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading...", total=None)

            progress.update(task, description="Initializing RAG agent...")

            # Get fiscal info from CLAUDE.md if available
            fiscal_info = _get_fiscal_info(ticker)

            agent = create_agent(
                index_dir=INDEX_DIR,
                ticker=ticker,
                fiscal_info=fiscal_info,
                rerank=rerank,
            )

            if rerank:
                progress.update(task, description="Loading reranker model...")
                _ = agent.searcher.reranker.model  # Pre-load

            if depth_level != Depth.FAST:
                progress.update(task, description="Decomposing query...")

            progress.update(task, description="Searching and generating answer...")
            response = agent.ask(
                question=question,
                depth=depth_level,
                doc_type=doc_type,
                rerank=rerank,
            )

            progress.update(task, description="[green]Done[/green]")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Display sub-queries if verbose
    if verbose and response.sub_queries:
        console.print("[bold]Sub-queries generated:[/bold]")
        for sq in response.sub_queries:
            console.print(f"  - {sq}")
        console.print()

    # Display answer
    console.print("[bold cyan]Answer:[/bold cyan]")
    console.print(response.answer)

    # Display sources if requested
    if show_sources and response.sources:
        console.print("\n[bold]Sources:[/bold]")
        seen = set()
        for src in response.sources:
            key = (src.filing_type, src.filing_date, src.section)
            if key not in seen:
                seen.add(key)
                console.print(f"  [dim]{src.filing_type} ({src.filing_date}) - {src.section}[/dim]")

    # Show stats if verbose
    if verbose:
        console.print(
            f"\n[dim]Hops: {len(response.hops)} | Searches: {response.total_searches} | File reads: {response.total_file_reads}[/dim]"
        )

    # Token usage
    console.print(
        f"[dim]Tokens: {response.input_tokens} in / {response.output_tokens} out | Confidence: {response.confidence}[/dim]"
    )
