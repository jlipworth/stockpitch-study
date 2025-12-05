"""Index command for building vector indexes."""

import hashlib
from typing import Annotated

import typer

from src.filings import Manifest
from src.rag import (
    DEFAULT_MODEL,
    AnalystReportParser,
    Chunk,
    EmbeddingModel,
    SECFilingParser,
    TranscriptParser,
    VectorStore,
    calculate_batch_size,
    clear_gpu_cache,
    get_gpu_memory,
)

from . import (
    ANALYST_DIR,
    CONFERENCES_DIR,
    DATA_DIR,
    INDEX_DIR,
    MISC_DIR,
    PRESENTATIONS_DIR,
    TRANSCRIPTS_DIR,
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SourceType,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    app,
    console,
    logger,
)


def _add_chunk_linking(chunks: list[Chunk], source_hash: str, section_key: str = "main") -> None:
    """Add chunk linking (prev_chunk_id, next_chunk_id) to a list of chunks.

    Phase 1.1: This enables context expansion for non-SEC document types
    (transcripts, analyst reports, etc.) by linking adjacent chunks.

    Modifies chunks in-place.

    Args:
        chunks: List of Chunk objects to link
        source_hash: Hash of the source file (used in chunk IDs)
        section_key: Section identifier for the chunk IDs (default: "main")
    """
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        # Generate chunk ID
        chunk_id = f"{source_hash}_{section_key}_{i}"
        chunk.chunk_id = chunk_id
        chunk.position_in_section = i
        chunk.total_chunks_in_section = total_chunks
        chunk.is_section_start = i == 0
        chunk.is_section_end = i == total_chunks - 1

        # Link to adjacent chunks
        if i > 0:
            chunk.prev_chunk_id = f"{source_hash}_{section_key}_{i - 1}"
        if i < total_chunks - 1:
            chunk.next_chunk_id = f"{source_hash}_{section_key}_{i + 1}"

        # Add to metadata for storage
        chunk.metadata["chunk_id"] = chunk.chunk_id
        chunk.metadata["prev_chunk_id"] = chunk.prev_chunk_id
        chunk.metadata["next_chunk_id"] = chunk.next_chunk_id
        chunk.metadata["position_in_section"] = chunk.position_in_section
        chunk.metadata["is_section_start"] = chunk.is_section_start
        chunk.metadata["is_section_end"] = chunk.is_section_end


def _count_files_for_source(ticker: str, source: SourceType) -> int:
    """Count files to be indexed for a given source."""
    if source == SourceType.SEC:
        manifest_path = DATA_DIR / ticker / "manifest.json"
        if manifest_path.exists():
            manifest = Manifest.load(manifest_path)
            return len(manifest.filings)
        return 0
    elif source == SourceType.TRANSCRIPTS:
        transcripts_dir = TRANSCRIPTS_DIR / ticker
        if transcripts_dir.exists():
            return len(list(transcripts_dir.glob("*.pdf")))
        return 0
    elif source == SourceType.ANALYST:
        analyst_dir = ANALYST_DIR / ticker
        if analyst_dir.exists():
            return len(list(analyst_dir.glob("*.pdf")))
        return 0
    elif source == SourceType.PRESENTATIONS:
        pres_dir = PRESENTATIONS_DIR / ticker
        if pres_dir.exists():
            return len(list(pres_dir.glob("*.pdf")))
        return 0
    elif source == SourceType.CONFERENCES:
        conf_dir = CONFERENCES_DIR / ticker
        if conf_dir.exists():
            return len(list(conf_dir.glob("*.pdf")))
        return 0
    elif source == SourceType.MISC:
        misc_dir = MISC_DIR / ticker
        if misc_dir.exists():
            return len(list(misc_dir.glob("*.pdf")))
        return 0
    return 0


def _store_sections_from_filing(store: VectorStore, parsed, source_hash: str) -> int:
    """Store sections from a parsed filing for parent-child retrieval.

    Args:
        store: VectorStore to store sections in
        parsed: ParsedFiling with sections
        source_hash: File hash (first 8 chars used for section_id)

    Returns:
        Number of sections stored
    """
    hash_prefix = source_hash[:8] if source_hash else ""
    stored = 0

    for section in parsed.sections:
        section_key = section.item.replace(" ", "_").replace(",", "")
        section_id = f"{hash_prefix}_{section_key}"

        store.store_section(
            section_id=section_id,
            item=section.item,
            title=section.title,
            content=section.content,
            filing_type=parsed.filing_type,
            filing_date=parsed.filing_date,
            source=hash_prefix,
        )
        stored += 1

    return stored


def _index_sec_filings(
    ticker: str,
    store: VectorStore,
    embedding_model: EmbeddingModel,
    parser: SECFilingParser,
    rebuild: bool,
    progress,
    task,
    cumulative_chunks: int = 0,
    debug_gpu: bool = False,
) -> tuple[int, int, int]:
    """Index SEC filings from manifest. Returns (indexed, skipped, chunks)."""
    company_dir = DATA_DIR / ticker
    manifest_path = company_dir / "manifest.json"

    if not manifest_path.exists():
        return 0, 0, 0

    manifest = Manifest.load(manifest_path)
    indexed_count = 0
    skipped_count = 0
    chunk_count = 0

    for accession, filing_meta in manifest.filings.items():
        file_path = DATA_DIR / filing_meta.file_path
        if not file_path.exists():
            progress.advance(task)
            continue

        if not rebuild and store.is_file_indexed(str(file_path), filing_meta.file_hash):
            skipped_count += 1
            progress.advance(task)
            continue

        file_label = f"{filing_meta.filing_type}_{filing_meta.filing_date}"
        progress.update(
            task,
            description=f"SEC {filing_meta.filing_type} {filing_meta.filing_date}",
        )

        if debug_gpu:
            _log_gpu_memory("before parse", file_label)

        try:
            parsed = parser.parse_file(
                file_path,
                ticker=ticker,
                filing_type=filing_meta.filing_type,
                filing_date=filing_meta.filing_date,
            )
            # Parent-child: Store full sections for later retrieval
            _store_sections_from_filing(store, parsed, filing_meta.file_hash)

            # Pass file_hash for consistent section_id generation
            chunks = parser.chunk_filing(parsed, source_hash=filing_meta.file_hash)
            if debug_gpu:
                _log_chunk_stats(chunks, file_label)
            if chunks:
                if debug_gpu:
                    _log_gpu_memory("before embed", file_label)
                embedded = embedding_model.embed_chunks(chunks, show_progress=True)
                if debug_gpu:
                    _log_gpu_memory("after embed", file_label)
                    clear_gpu_cache()
                    _log_gpu_memory("after cache clear", file_label)
                store.add(embedded, source_file=str(file_path), file_hash=filing_meta.file_hash)
                indexed_count += 1
                chunk_count += len(embedded)
                progress.update(task, chunks=cumulative_chunks + chunk_count)
        except Exception as e:
            console.print(f"  [red]Error processing {filing_meta.file_path}: {e}[/red]")

        progress.advance(task)

    return indexed_count, skipped_count, chunk_count


def _index_transcripts(
    ticker: str,
    store: VectorStore,
    embedding_model: EmbeddingModel,
    rebuild: bool,
    progress,
    task,
    cumulative_chunks: int = 0,
    debug_gpu: bool = False,
) -> tuple[int, int, int]:
    """Index transcripts from transcripts directory. Returns (indexed, skipped, chunks)."""
    transcripts_dir = TRANSCRIPTS_DIR / ticker
    if not transcripts_dir.exists():
        return 0, 0, 0

    parser = TranscriptParser()
    indexed_count = 0
    skipped_count = 0
    chunk_count = 0

    for pdf_file in transcripts_dir.glob("*.pdf"):
        file_hash = hashlib.md5(pdf_file.read_bytes()).hexdigest()

        if not rebuild and store.is_file_indexed(str(pdf_file), file_hash):
            skipped_count += 1
            progress.advance(task)
            continue

        progress.update(task, description=f"Transcript: {pdf_file.name}")

        if debug_gpu:
            _log_gpu_memory("before parse", pdf_file.name)

        try:
            parsed = parser.parse_file(pdf_file, ticker)
            chunks = []
            for block in parsed.all_blocks:
                chunks.append(
                    Chunk(
                        text=block.text,
                        metadata={
                            "ticker": ticker,
                            "doc_type": "transcript",
                            "section": block.section,
                            "speaker": block.speaker.name,
                            "source": str(pdf_file),
                        },
                    )
                )

            if debug_gpu:
                _log_chunk_stats(chunks, pdf_file.name)

            # Phase 1.1: Add chunk linking for context expansion
            if chunks:
                _add_chunk_linking(chunks, file_hash, section_key="transcript")
                if debug_gpu:
                    _log_gpu_memory("before embed", pdf_file.name)
                embedded = embedding_model.embed_chunks(chunks, show_progress=True)
                if debug_gpu:
                    _log_gpu_memory("after embed", pdf_file.name)
                    clear_gpu_cache()
                    _log_gpu_memory("after cache clear", pdf_file.name)
                store.add(embedded, source_file=str(pdf_file), file_hash=file_hash)
                indexed_count += 1
                chunk_count += len(embedded)
                progress.update(task, chunks=cumulative_chunks + chunk_count)
        except Exception as e:
            console.print(f"  [red]Error processing {pdf_file.name}: {e}[/red]")

        progress.advance(task)

    return indexed_count, skipped_count, chunk_count


def _index_analyst_reports(
    ticker: str,
    store: VectorStore,
    embedding_model: EmbeddingModel,
    rebuild: bool,
    progress,
    task,
    cumulative_chunks: int = 0,
    debug_gpu: bool = False,
) -> tuple[int, int, int]:
    """Index analyst reports from analyst directory. Returns (indexed, skipped, chunks)."""
    analyst_dir = ANALYST_DIR / ticker
    if not analyst_dir.exists():
        return 0, 0, 0

    parser = AnalystReportParser()
    indexed_count = 0
    skipped_count = 0
    chunk_count = 0

    for pdf_file in analyst_dir.glob("*.pdf"):
        file_hash = hashlib.md5(pdf_file.read_bytes()).hexdigest()

        if not rebuild and store.is_file_indexed(str(pdf_file), file_hash):
            skipped_count += 1
            progress.advance(task)
            continue

        progress.update(task, description=f"Analyst: {pdf_file.name}")

        if debug_gpu:
            _log_gpu_memory("before parse", pdf_file.name)

        try:
            parsed = parser.parse_file(pdf_file, ticker)
            chunks = []
            for section in parsed.sections:
                chunks.append(
                    Chunk(
                        text=section.content,
                        metadata={
                            "ticker": ticker,
                            "doc_type": "analyst_report",
                            "section": section.title,
                            "firm": parsed.metadata.firm,
                            "rating": parsed.metadata.rating,
                            "source": str(pdf_file),
                        },
                    )
                )

            if debug_gpu:
                _log_chunk_stats(chunks, pdf_file.name)

            # Phase 1.1: Add chunk linking for context expansion
            if chunks:
                _add_chunk_linking(chunks, file_hash, section_key="analyst")
                if debug_gpu:
                    _log_gpu_memory("before embed", pdf_file.name)
                embedded = embedding_model.embed_chunks(chunks, show_progress=True)
                if debug_gpu:
                    _log_gpu_memory("after embed", pdf_file.name)
                    clear_gpu_cache()
                    _log_gpu_memory("after cache clear", pdf_file.name)
                store.add(embedded, source_file=str(pdf_file), file_hash=file_hash)
                indexed_count += 1
                chunk_count += len(embedded)
                progress.update(task, chunks=cumulative_chunks + chunk_count)
        except Exception as e:
            console.print(f"  [red]Error processing {pdf_file.name}: {e}[/red]")

        progress.advance(task)

    return indexed_count, skipped_count, chunk_count


def _index_generic_pdfs(
    ticker: str,
    source_dir,
    doc_type: str,
    store: VectorStore,
    embedding_model: EmbeddingModel,
    rebuild: bool,
    progress,
    task,
    cumulative_chunks: int = 0,
    debug_gpu: bool = False,
) -> tuple[int, int, int]:
    """Index generic PDFs using simple text extraction. Returns (indexed, skipped, chunks)."""
    import fitz

    pdf_dir = source_dir / ticker
    if not pdf_dir.exists():
        return 0, 0, 0

    indexed_count = 0
    skipped_count = 0
    chunk_count = 0

    for pdf_file in pdf_dir.glob("*.pdf"):
        file_hash = hashlib.md5(pdf_file.read_bytes()).hexdigest()

        if not rebuild and store.is_file_indexed(str(pdf_file), file_hash):
            skipped_count += 1
            progress.advance(task)
            continue

        progress.update(task, description=f"{doc_type.title()}: {pdf_file.name}")

        if debug_gpu:
            _log_gpu_memory("before parse", pdf_file.name)

        try:
            # Extract text from PDF
            doc = fitz.open(pdf_file)
            full_text = "\n".join(str(page.get_text()) for page in doc)
            doc.close()

            # Simple paragraph-based chunking
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            chunks = []
            current_chunk = []
            current_size = 0

            for para in paragraphs:
                para_words = len(para.split())
                if current_size + para_words > 800:  # ~1000 tokens
                    if current_chunk:
                        chunks.append(
                            Chunk(
                                text="\n\n".join(current_chunk),
                                metadata={
                                    "ticker": ticker,
                                    "doc_type": doc_type,
                                    "source": str(pdf_file),
                                },
                            )
                        )
                    current_chunk = [para]
                    current_size = para_words
                else:
                    current_chunk.append(para)
                    current_size += para_words

            if current_chunk:
                chunks.append(
                    Chunk(
                        text="\n\n".join(current_chunk),
                        metadata={
                            "ticker": ticker,
                            "doc_type": doc_type,
                            "source": str(pdf_file),
                        },
                    )
                )

            if debug_gpu:
                _log_chunk_stats(chunks, pdf_file.name)

            # Phase 1.1: Add chunk linking for context expansion
            if chunks:
                _add_chunk_linking(chunks, file_hash, section_key=doc_type)
                if debug_gpu:
                    _log_gpu_memory("before embed", pdf_file.name)
                embedded = embedding_model.embed_chunks(chunks, show_progress=True)
                if debug_gpu:
                    _log_gpu_memory("after embed", pdf_file.name)
                    clear_gpu_cache()
                    _log_gpu_memory("after cache clear", pdf_file.name)
                store.add(embedded, source_file=str(pdf_file), file_hash=file_hash)
                indexed_count += 1
                chunk_count += len(embedded)
                progress.update(task, chunks=cumulative_chunks + chunk_count)
        except Exception as e:
            console.print(f"  [red]Error processing {pdf_file.name}: {e}[/red]")

        progress.advance(task)

    return indexed_count, skipped_count, chunk_count


def _log_gpu_memory(label: str, file_name: str = "") -> None:
    """Print GPU memory stats for debugging OOM issues."""
    gpu_info = get_gpu_memory()
    if gpu_info:
        file_str = f" [{file_name}]" if file_name else ""
        console.print(f"  [dim]GPU{file_str} {label}: {gpu_info.used:.2f}GB used / {gpu_info.free:.2f}GB free[/dim]")


def _log_chunk_stats(chunks: list, file_name: str) -> None:
    """Print chunk size stats for debugging memory spikes."""
    if not chunks:
        return
    lengths = [len(c.text) for c in chunks]
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    # Estimate tokens (~4 chars per token)
    max_tokens_est = max_len // 4
    console.print(
        f"  [dim]  {file_name}: {len(chunks)} chunks, "
        f"max={max_len:,} chars (~{max_tokens_est:,} tokens), avg={avg_len:,.0f} chars[/dim]"
    )
    if max_tokens_est > 4000:
        console.print(
            f"  [yellow]  ⚠ Large chunk detected (~{max_tokens_est:,} tokens) - may spike GPU memory[/yellow]"
        )


@app.command()
def index(
    company: Annotated[str, typer.Argument(help="Company identifier (ticker or name)")],
    source: Annotated[SourceType, typer.Option("--source", "-s", help="Document source to index")] = SourceType.ALL,
    rebuild: Annotated[bool, typer.Option("--rebuild", "-r", help="Force rebuild entire index")] = False,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", "-b", help="Embedding batch size (auto-detected by device if not set)"),
    ] = None,
    debug_gpu: Annotated[
        bool,
        typer.Option("--debug-gpu", help="Print GPU memory before/after each file (for debugging OOM)"),
    ] = False,
) -> None:
    """Build or update the vector index for a company.

    Sources:
    - sec: SEC filings from data/{ticker}/ (requires prior fetch)
    - transcripts: Earnings call PDFs from transcripts/{ticker}/
    - analyst: Bank research PDFs from analyst/{ticker}/
    - presentations: Investor day PDFs from presentations/{ticker}/
    - conferences: Conference transcript PDFs from conferences/{ticker}/
    - misc: Other materials from misc/{ticker}/
    - all: Index all available sources

    Batch size defaults: CUDA=128, MPS=32, CPU=16
    """
    ticker = company.upper()
    logger.info(f"Starting index operation for {ticker} (source={source.value}, rebuild={rebuild})")

    # Check if at least one source has data
    has_sec = (DATA_DIR / ticker / "manifest.json").exists()
    has_transcripts = (TRANSCRIPTS_DIR / ticker).exists()
    has_analyst = (ANALYST_DIR / ticker).exists()
    has_presentations = (PRESENTATIONS_DIR / ticker).exists()
    has_conferences = (CONFERENCES_DIR / ticker).exists()
    has_misc = (MISC_DIR / ticker).exists()

    if source == SourceType.SEC and not has_sec:
        console.print(f"[red]No SEC filings found for {ticker}. Run 'pitch fetch {ticker}' first.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Indexing documents for {ticker}[/bold]")

    # Determine which sources to process
    sources_to_process = []
    if source == SourceType.ALL:
        if has_sec:
            sources_to_process.append(SourceType.SEC)
        if has_transcripts:
            sources_to_process.append(SourceType.TRANSCRIPTS)
        if has_analyst:
            sources_to_process.append(SourceType.ANALYST)
        if has_presentations:
            sources_to_process.append(SourceType.PRESENTATIONS)
        if has_conferences:
            sources_to_process.append(SourceType.CONFERENCES)
        if has_misc:
            sources_to_process.append(SourceType.MISC)
    else:
        sources_to_process.append(source)

    # Count total files across all sources
    total_files = sum(_count_files_for_source(ticker, src) for src in sources_to_process)
    logger.info(f"Found {total_files} files to index across {len(sources_to_process)} source(s)")

    # Check GPU memory and calculate batch size if not specified
    gpu_info = get_gpu_memory()
    if gpu_info and batch_size is None:
        batch_size = calculate_batch_size(gpu_info.free)
        console.print(f"[dim]GPU: {gpu_info}[/dim]")
        console.print(f"[dim]Auto batch size: {batch_size}[/dim]")
        logger.debug(f"GPU info: {gpu_info}, auto batch_size={batch_size}")

    # Load model first with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading embedding model...", total=None)
        embedding_model = EmbeddingModel(model_name=DEFAULT_MODEL, batch_size=batch_size)
        _ = embedding_model.model
        device_info = f"device={embedding_model.device}, batch_size={embedding_model.batch_size}"
        progress.update(task, description=f"[green]Loaded {DEFAULT_MODEL} ({device_info})[/green]")

    store = VectorStore(
        index_dir=INDEX_DIR,
        ticker=ticker,
        embedding_dim=embedding_model.embedding_dim,
        embedding_model=DEFAULT_MODEL,
    )

    if rebuild:
        console.print("  [yellow]Rebuilding index from scratch[/yellow]")
        store.clear()

    total_indexed = 0
    total_skipped = 0
    total_chunks = 0
    parser = SECFilingParser()

    # Process with progress bar
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
        main_task = progress.add_task(
            "Processing...",
            total=total_files,
            chunks=0,
        )

        for src in sources_to_process:
            if src == SourceType.SEC:
                i, s, c = _index_sec_filings(
                    ticker, store, embedding_model, parser, rebuild, progress, main_task, total_chunks, debug_gpu
                )
            elif src == SourceType.TRANSCRIPTS:
                i, s, c = _index_transcripts(
                    ticker, store, embedding_model, rebuild, progress, main_task, total_chunks, debug_gpu
                )
            elif src == SourceType.ANALYST:
                i, s, c = _index_analyst_reports(
                    ticker, store, embedding_model, rebuild, progress, main_task, total_chunks, debug_gpu
                )
            elif src == SourceType.PRESENTATIONS:
                i, s, c = _index_generic_pdfs(
                    ticker,
                    PRESENTATIONS_DIR,
                    "presentation",
                    store,
                    embedding_model,
                    rebuild,
                    progress,
                    main_task,
                    total_chunks,
                    debug_gpu,
                )
            elif src == SourceType.CONFERENCES:
                i, s, c = _index_generic_pdfs(
                    ticker,
                    CONFERENCES_DIR,
                    "conference",
                    store,
                    embedding_model,
                    rebuild,
                    progress,
                    main_task,
                    total_chunks,
                    debug_gpu,
                )
            elif src == SourceType.MISC:
                i, s, c = _index_generic_pdfs(
                    ticker,
                    MISC_DIR,
                    "misc",
                    store,
                    embedding_model,
                    rebuild,
                    progress,
                    main_task,
                    total_chunks,
                    debug_gpu,
                )
            else:
                i, s, c = 0, 0, 0

            total_indexed += i
            total_skipped += s
            total_chunks += c

        progress.update(main_task, description="[green]Complete[/green]")

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Indexed: {total_indexed} documents ({total_chunks} chunks)")
    console.print(f"  Skipped: {total_skipped} (already indexed)")
    console.print(f"  Total in store: {store.count()} chunks")
    console.print(f"  Index location: {INDEX_DIR / ticker}")
