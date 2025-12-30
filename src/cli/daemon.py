"""Daemon management CLI commands.

Commands for managing the embedding daemon server.
"""

from typing import Annotated

import typer
from rich.table import Table

# Register commands with main app
from src.cli import PROJECT_ROOT, app, console


@app.command("serve")
def serve(
    background: Annotated[
        bool,
        typer.Option("--background", "-b", help="Run in background (detached)"),
    ] = False,
    no_timeout: Annotated[
        bool,
        typer.Option("--no-timeout", help="Disable idle auto-shutdown"),
    ] = False,
    timeout: Annotated[
        int,
        typer.Option("--timeout", "-t", help="Idle timeout in seconds (default: 900)"),
    ] = 900,
) -> None:
    """Start the embedding daemon server.

    The daemon keeps BGE-M3 and reranker models loaded in GPU memory,
    eliminating ~30s cold-start latency on searches.

    Examples:

        pitch serve              # Run in foreground (Ctrl+C to stop)

        pitch serve --background # Run detached in background

        pitch serve --no-timeout # Disable 15-min auto-shutdown
    """
    from src.daemon import SOCKET_FILENAME, run_daemon, spawn_daemon

    socket_path = PROJECT_ROOT / SOCKET_FILENAME
    idle_timeout = 0 if no_timeout else timeout

    if background:
        # Check if already running
        from src.daemon import DaemonClient

        client = DaemonClient(socket_path)
        if client.is_daemon_running():
            console.print("[yellow]Daemon is already running[/yellow]")
            raise typer.Exit(1)

        console.print("[blue]Starting daemon in background...[/blue]")
        spawn_daemon(
            socket_path=socket_path,
            background=True,
            idle_timeout=idle_timeout,
            wait_for_ready=True,
        )
        console.print("[green]Daemon started[/green]")
        console.print(f"Socket: {socket_path}")
        if idle_timeout > 0:
            console.print(f"Auto-shutdown: {idle_timeout}s idle timeout")
        else:
            console.print("Auto-shutdown: disabled")
    else:
        # Run in foreground
        console.print("[blue]Starting daemon in foreground...[/blue]")
        console.print("Press Ctrl+C to stop")
        console.print()

        run_daemon(
            socket_path=socket_path,
            idle_timeout=idle_timeout,
        )


@app.command("daemon-status")
def daemon_status() -> None:
    """Show daemon status and health information.

    Displays:
    - Whether daemon is running
    - GPU memory usage
    - Queue depth and current job
    - Uptime and requests processed
    """
    from src.daemon import SOCKET_FILENAME, DaemonClient

    socket_path = PROJECT_ROOT / SOCKET_FILENAME
    client = DaemonClient(socket_path)

    if not client.is_daemon_running():
        console.print("[yellow]Daemon is not running[/yellow]")
        console.print("Start with: pitch serve")
        raise typer.Exit(1)

    try:
        health = client.health()
    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")
        raise typer.Exit(1)

    # Create status table
    table = Table(title="Embedding Daemon Status", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    # Format uptime
    uptime_mins = int(health.uptime_seconds // 60)
    uptime_secs = int(health.uptime_seconds % 60)
    uptime_str = f"{uptime_mins}m {uptime_secs}s"

    table.add_row("Status", "[green]Running[/green]")
    table.add_row("Uptime", uptime_str)
    table.add_row("Requests Processed", str(health.requests_processed))
    table.add_row("Queue Depth", str(health.queue_depth))
    table.add_row(
        "Current Job",
        health.current_job if health.current_job else "[dim]idle[/dim]",
    )
    table.add_row("", "")  # Spacer

    if health.gpu_device_name:
        table.add_row("GPU", health.gpu_device_name)
        if health.gpu_memory_free_gb and health.gpu_memory_total_gb:
            used_gb = health.gpu_memory_total_gb - health.gpu_memory_free_gb
            table.add_row(
                "GPU Memory",
                f"{used_gb:.1f}GB / {health.gpu_memory_total_gb:.1f}GB used",
            )
    else:
        table.add_row("GPU", "[dim]CPU mode[/dim]")

    table.add_row("", "")  # Spacer
    table.add_row("Embed Batch Size", str(health.embed_batch_size))
    table.add_row("Rerank Batch Size", str(health.rerank_batch_size))

    console.print(table)


@app.command("daemon-stop")
def daemon_stop(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force stop (don't wait for current job)"),
    ] = False,
) -> None:
    """Stop the embedding daemon.

    By default, waits for the current job to finish before stopping.
    Use --force to stop immediately.

    Examples:

        pitch daemon-stop         # Graceful stop

        pitch daemon-stop --force # Immediate stop
    """
    from src.daemon import SOCKET_FILENAME, DaemonClient, force_stop_daemon

    socket_path = PROJECT_ROOT / SOCKET_FILENAME
    client = DaemonClient(socket_path)

    if not socket_path.exists():
        console.print("[yellow]Daemon is not running[/yellow]")
        raise typer.Exit(0)

    if force:
        console.print("[yellow]Force stopping daemon...[/yellow]")
        force_stop_daemon(socket_path)
        console.print("[green]Daemon stopped[/green]")
    else:
        if not client.is_daemon_running():
            # Socket exists but daemon not responsive - clean up
            socket_path.unlink()
            console.print("[yellow]Removed stale socket[/yellow]")
            raise typer.Exit(0)

        console.print("[blue]Requesting graceful shutdown...[/blue]")
        if client.shutdown():
            console.print("[green]Shutdown initiated[/green]")
            console.print("Daemon will stop after current job completes")
        else:
            console.print("[red]Shutdown request failed[/red]")
            raise typer.Exit(1)


@app.command("query")
def query(
    company: Annotated[str, typer.Argument(help="Company identifier (ticker)")],
    query_text: Annotated[str, typer.Argument(help="Search query")],
    wait: Annotated[
        bool,
        typer.Option("--wait", "-w", help="Wait for result (blocking)"),
    ] = False,
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", "--limit", help="Number of results"),
    ] = 10,
    rerank: Annotated[
        bool,
        typer.Option("--rerank", "-r", help="Enable reranking"),
    ] = True,
) -> None:
    """Submit a search query to the daemon.

    By default, returns immediately and saves result to output/queries/.
    Use --wait to block until result is ready.

    Examples:

        pitch query AAPL "revenue growth"          # Non-blocking

        pitch query AAPL "revenue growth" --wait   # Blocking, prints result

        pitch query AAPL "ARR trends" -w -k 5      # Wait, top 5 results
    """
    from src.daemon import (
        SOCKET_FILENAME,
        get_daemon_client,
    )

    socket_path = PROJECT_ROOT / SOCKET_FILENAME

    # Get or spawn daemon
    client = get_daemon_client(auto_spawn=True, socket_path=socket_path)
    if client is None:
        console.print("[red]Cannot connect to daemon and auto-spawn failed[/red]")
        console.print("Try starting manually: pitch serve")
        raise typer.Exit(1)

    # For now, query uses the search command directly since SEARCH type
    # is not fully implemented in daemon. This is a placeholder.
    console.print("[yellow]Note: Full daemon search not yet implemented. " "Using pitch search instead.[/yellow]")

    # Fall back to regular search command
    from src.cli.search import search

    search(
        company=company,
        query=query_text,
        top_k=top_k,
        rerank=rerank,
    )
