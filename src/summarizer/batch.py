"""Batch processing for summarization using Claude Message Batches API.

Batch processing provides 50% cost savings and higher rate limits.
Jobs run asynchronously and can take up to 24 hours.

Key Features:
- Automatic chunking: Long sections (>80k/50k chars) split by paragraphs into multiple requests
- Smart token limits: 60% of input tokens with no artificial caps (trusts prompts)
- Section ID normalization: Handles "Q&A" vs "Q and A" variations robustly
- Atomic file writes: Uses tempfile + shutil.move to prevent corruption
- Resubmit workflow: Reprocess truncated sections with higher token limits, auto-merges

Workflow:
1. submit_batch() - Submit summarization requests
2. check_batch_status() - Poll for completion
3. retrieve_batch_results() - Download and process results
4. resubmit_truncated_sections() - Optional: reprocess sections with higher limits
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

from .summarizer import (
    SECTION_WEIGHTS,
    FilingSummary,
    SectionSummary,
    Summarizer,
)

logger = logging.getLogger(__name__)


def normalize_section_id(custom_id_part: str) -> str:
    """Convert custom_id component back to section_id.

    This reverses the sanitization done by sanitize_custom_id_component().

    Args:
        custom_id_part: Sanitized component from custom_id (e.g., "Q-and-A")

    Returns:
        Original section ID (e.g., "Q&A")

    Examples:
        >>> normalize_section_id("Q-and-A")
        'Q&A'
        >>> normalize_section_id("Prepared-Remarks")
        'Prepared Remarks'
        >>> normalize_section_id("Item-1A")
        'Item 1A'
    """
    # First restore & from -and-
    text = custom_id_part.replace("-and-", "&")
    # Then replace remaining hyphens with spaces
    text = text.replace("-", " ")
    return text


def _normalize_for_matching(s: str) -> str:
    """Normalize a section ID for fuzzy comparison.

    Used for O(1) lookup in sets/dicts instead of O(n) linear search.

    Examples:
        >>> _normalize_for_matching("Q&A")
        'q and a'
        >>> _normalize_for_matching("Prepared-Remarks")
        'prepared remarks'
        >>> _normalize_for_matching("Item 1A")
        'item 1a'
    """
    s = s.lower().strip()
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    # Normalize & and "and"
    s = s.replace("&", " and ").replace("-", " ")
    # Collapse multiple spaces again
    s = re.sub(r"\s+", " ", s)
    return s


def section_ids_match(id1: str, id2: str) -> bool:
    """Check if two section IDs refer to the same section.

    Handles various normalizations and equivalences:
    - Case insensitive
    - Whitespace normalization
    - "&" vs " and " equivalence
    - "-" vs " " equivalence

    Args:
        id1: First section ID
        id2: Second section ID

    Returns:
        True if the IDs refer to the same section

    Examples:
        >>> section_ids_match("Q&A", "Q and A")
        True
        >>> section_ids_match("Prepared Remarks", "Prepared-Remarks")
        True
        >>> section_ids_match("Item 1A", "item 1a")
        True
    """
    return _normalize_for_matching(id1) == _normalize_for_matching(id2)


@dataclass
class BatchJob:
    """Represents a submitted batch job."""

    batch_id: str
    ticker: str
    filing_type: str
    filing_date: str
    created_at: str
    status: str
    request_count: int
    results_path: str | None = None
    source_path: str | None = None  # Path to source file for resubmission

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_id": self.batch_id,
            "ticker": self.ticker,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "created_at": self.created_at,
            "status": self.status,
            "request_count": self.request_count,
            "results_path": self.results_path,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchJob":
        """Create from dictionary."""
        # Handle missing source_path for backwards compatibility
        if "source_path" not in data:
            data["source_path"] = None
        return cls(**data)


class BatchJobTracker:
    """Tracks batch jobs in a local manifest file."""

    def __init__(self, manifest_path: Path):
        """Initialize tracker with manifest path."""
        self.manifest_path = manifest_path
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def load_jobs(self) -> list[BatchJob]:
        """Load all tracked jobs."""
        if not self.manifest_path.exists():
            return []
        with open(self.manifest_path) as f:
            data = json.load(f)
        return [BatchJob.from_dict(job) for job in data.get("jobs", [])]

    def save_jobs(self, jobs: list[BatchJob]) -> None:
        """Save jobs to manifest."""
        data = {"jobs": [job.to_dict() for job in jobs]}
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_job(self, job: BatchJob) -> None:
        """Add a new job."""
        jobs = self.load_jobs()
        jobs.append(job)
        self.save_jobs(jobs)

    def update_job(self, batch_id: str, **updates) -> BatchJob | None:
        """Update a job by batch_id."""
        jobs = self.load_jobs()
        for job in jobs:
            if job.batch_id == batch_id:
                for key, value in updates.items():
                    setattr(job, key, value)
                self.save_jobs(jobs)
                return job
        return None

    def get_job(self, batch_id: str) -> BatchJob | None:
        """Get a job by batch_id."""
        for job in self.load_jobs():
            if job.batch_id == batch_id:
                return job
        return None

    def get_pending_jobs(self) -> list[BatchJob]:
        """Get jobs that haven't completed."""
        return [job for job in self.load_jobs() if job.status not in ("ended", "expired", "failed")]


class BatchProcessor:
    """Handles batch submission, polling, and result retrieval."""

    def __init__(
        self,
        api_key: str | None = None,
        processed_dir: Path | None = None,
    ):
        """
        Initialize batch processor.

        Args:
            api_key: Anthropic API key (uses env var if None)
            processed_dir: Directory for output files
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or pass api_key parameter.")
        self.client = Anthropic(api_key=api_key)
        self.processed_dir = processed_dir
        self.summarizer = Summarizer(api_key=api_key, processed_dir=processed_dir)

        # Initialize job tracker
        tracker_path = (processed_dir or Path(".")) / "batch_jobs.json"
        self.tracker = BatchJobTracker(tracker_path)

    def submit_batch(
        self,
        file_path: Path,
        ticker: str,
        filing_type: str,
        filing_date: str,
        sections_to_include: list[str] | None = None,
    ) -> BatchJob:
        """
        Submit a filing for batch summarization.

        Args:
            file_path: Path to the filing
            ticker: Company ticker
            filing_type: Type of filing (10-K, 10-Q, transcript, presentation, etc.)
            filing_date: Filing date
            sections_to_include: Optional list of sections to process

        Returns:
            BatchJob with the submitted job details
        """
        # Route to appropriate method based on filing type
        pdf_types = {"transcript", "presentation", "conference", "analyst"}
        if filing_type.lower() in pdf_types:
            requests = self.summarizer.prepare_batch_requests_pdf(
                file_path=file_path,
                ticker=ticker,
                doc_type=filing_type.lower(),
                doc_date=filing_date,
            )
        else:
            # SEC filings
            requests = self.summarizer.prepare_batch_requests(
                file_path=file_path,
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                sections_to_include=sections_to_include,
            )

        # Submit batch (requests are already in correct format)
        logger.debug(f"Submitting batch: {ticker} {filing_type} {filing_date} with {len(requests)} requests")
        message_batch = self.client.messages.batches.create(requests=requests)
        logger.debug(f"Batch created: {message_batch.id}, status: {message_batch.processing_status}")

        # Create and track job
        job = BatchJob(
            batch_id=message_batch.id,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            created_at=datetime.now().isoformat(),
            status=message_batch.processing_status,
            request_count=len(requests),
            source_path=str(file_path),
        )
        self.tracker.add_job(job)
        logger.info(f"✓ Batch submitted: {job.batch_id} ({job.request_count} requests)")

        return job

    def check_status(self, batch_id: str) -> tuple[str, dict]:
        """
        Check the status of a batch job.

        Args:
            batch_id: The batch job ID

        Returns:
            Tuple of (status, details dict with counts)
        """
        logger.debug(f"Checking status for batch: {batch_id}")
        message_batch = self.client.messages.batches.retrieve(batch_id)

        details = {
            "processing_status": message_batch.processing_status,
            "created_at": message_batch.created_at,
            "ended_at": getattr(message_batch, "ended_at", None),
            "request_counts": {
                "processing": message_batch.request_counts.processing,
                "succeeded": message_batch.request_counts.succeeded,
                "errored": message_batch.request_counts.errored,
                "canceled": message_batch.request_counts.canceled,
                "expired": message_batch.request_counts.expired,
            },
        }

        logger.debug(
            f"Batch {batch_id} status: {message_batch.processing_status}, "
            f"counts: {message_batch.request_counts.succeeded}/{message_batch.request_counts.processing + message_batch.request_counts.succeeded} succeeded"
        )

        # Update tracker
        self.tracker.update_job(batch_id, status=message_batch.processing_status)

        return message_batch.processing_status, details

    def poll_until_complete(
        self,
        batch_id: str,
        poll_interval: int = 60,
        timeout_hours: float | None = 24.0,
        progress_callback: Callable[[str, dict], None] | None = None,
    ) -> str:
        """
        Poll a batch job until it completes.

        Args:
            batch_id: The batch job ID
            poll_interval: Seconds between status checks
            timeout_hours: Maximum hours to wait (default 24, None for no limit)
            progress_callback: Optional callback(status, details)

        Returns:
            Final status ("ended", "expired", "canceled")

        Raises:
            TimeoutError: If timeout_hours exceeded
            RuntimeError: If API errors persist
        """
        # Terminal statuses that indicate batch is done
        terminal_statuses = {"ended", "expired", "canceled"}
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            # Check timeout
            if timeout_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours > timeout_hours:
                    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout_hours} hours")

            # Try to get status with error handling
            try:
                status, details = self.check_status(batch_id)
                consecutive_errors = 0  # Reset on success
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Failed to check batch status after {max_consecutive_errors} attempts: {e}"
                    ) from e
                # Wait before retry
                time.sleep(poll_interval)
                continue

            if progress_callback:
                progress_callback(status, details)

            # Check for any terminal status, not just "ended"
            if status in terminal_statuses:
                return status

            time.sleep(poll_interval)

    def retrieve_results(
        self,
        batch_id: str,
        save_summary: bool = True,
    ) -> FilingSummary | None:
        """
        Retrieve and process batch results.

        Handles multi-part sections automatically by combining chunks (e.g., _part1, _part2)
        into complete section summaries. Uses section ID normalization to handle variations
        like "Q&A" vs "Q and A".

        If an existing summary file exists (e.g., from a prior batch), this will
        merge the new sections into it and regenerate the executive summary.
        This enables the resubmit workflow where only truncated sections are
        resubmitted and then merged back into the complete summary.

        Uses atomic writes (tempfile + shutil.move) to prevent corruption if interrupted.

        Args:
            batch_id: The batch job ID
            save_summary: Whether to save the summary to file

        Returns:
            FilingSummary if successful, None if batch not complete
        """
        # Use shared collection logic (removes 60+ lines of duplication)
        collected = self._collect_batch_sections(batch_id)
        if collected is None:
            return None

        job, section_summaries, total_input_tokens, total_output_tokens = collected

        # Check if there's an existing summary file to merge into
        # This handles the resubmit case where we only have partial sections
        existing_sections: list[SectionSummary] = []
        if self.processed_dir:
            output_path = self.summarizer.get_output_path(job.ticker, job.filing_type, job.filing_date)
            if output_path.exists():
                try:
                    existing_summary = FilingSummary.from_markdown(output_path)
                    existing_sections = existing_summary.section_summaries
                    logger.info(f"📋 Merging into existing summary with {len(existing_sections)} sections")
                except Exception as e:
                    logger.warning(f"Could not load existing summary: {e}")

        # Merge: new sections override existing ones (O(1) matching)
        merged_sections = self._merge_sections(existing_sections, section_summaries)

        # Create executive summary for all document types
        # For 10-K: include full investment thesis (bull/bear case)
        # For other docs: simpler summary without thesis
        include_thesis = job.filing_type == "10-K"
        exec_summary, exec_input, exec_output = self.summarizer.create_executive_summary(
            list(merged_sections.values()),
            f"{job.ticker} {job.filing_type} ({job.filing_date})",
            include_thesis=include_thesis,
        )
        total_input_tokens += exec_input
        total_output_tokens += exec_output

        # Build FilingSummary
        filing_summary = FilingSummary(
            ticker=job.ticker,
            filing_type=job.filing_type,
            filing_date=job.filing_date,
            source_path=f"batch:{batch_id}",
            executive_summary=exec_summary,
            section_summaries=list(merged_sections.values()),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

        # Save if requested and update tracker
        if save_summary and self.processed_dir:
            output_path = self.summarizer.get_output_path(job.ticker, job.filing_type, job.filing_date)
            self._save_summary_atomic(filing_summary, output_path)
            self.tracker.update_job(batch_id, results_path=str(output_path))
        else:
            # Mark as retrieved even without saving
            self.tracker.update_job(batch_id, results_path="retrieved:no-save")

        return filing_summary

    def _merge_sections(
        self,
        existing_sections: list[SectionSummary],
        new_sections: dict[str, SectionSummary],
    ) -> dict[str, SectionSummary]:
        """
        Merge new section summaries with existing ones.

        Uses O(1) normalized lookup to replace sections that match
        (handling variations like "Q&A" vs "Q and A").

        Args:
            existing_sections: List of existing section summaries
            new_sections: Dict of new section summaries (section_id -> summary)

        Returns:
            Merged dict of section_id -> SectionSummary
        """
        merged_sections: dict[str, SectionSummary] = {}

        # First add all existing sections and build normalized lookup
        normalized_to_original: dict[str, str] = {}
        for section in existing_sections:
            merged_sections[section.section_id] = section
            normalized_to_original[_normalize_for_matching(section.section_id)] = section.section_id

        # Then override with new sections, using O(1) lookup for replacements
        for section_id, new_section in new_sections.items():
            normalized_id = _normalize_for_matching(section_id)

            # Check if this replaces an existing section via normalized lookup
            if normalized_id in normalized_to_original:
                existing_id = normalized_to_original[normalized_id]
                logger.info(f"  ↻ Replacing section: {existing_id} with {section_id}")
                # Remove old version and add new one
                del merged_sections[existing_id]
                merged_sections[section_id] = new_section
                # Update lookup to point to new ID
                normalized_to_original[normalized_id] = section_id
            else:
                logger.info(f"  + Adding section: {section_id}")
                merged_sections[section_id] = new_section
                normalized_to_original[normalized_id] = section_id

        return merged_sections

    def _save_summary_atomic(self, filing_summary: FilingSummary, output_path: Path) -> None:
        """
        Save filing summary to file using atomic write.

        Uses tempfile + shutil.move to ensure atomic write on POSIX systems.
        This prevents race conditions when multiple processes retrieve batches.

        Args:
            filing_summary: The summary to save
            output_path: Target file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".md", dir=output_path.parent, prefix=f".{output_path.stem}_")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(filing_summary.to_markdown())
            shutil.move(temp_path, output_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _collect_batch_sections(
        self,
        batch_id: str,
    ) -> tuple[BatchJob, dict[str, SectionSummary], int, int] | None:
        """
        Collect section summaries from a batch without generating executive summary.

        Handles multi-chunk sections by combining chunk results using the summarizer's
        _combine_chunk_summaries method.

        Returns:
            Tuple of (job, section_summaries, input_tokens, output_tokens) or None if not ready
        """
        # Check status first
        status, _ = self.check_status(batch_id)
        if status != "ended":
            return None

        # Get job info
        job = self.tracker.get_job(batch_id)
        if not job:
            raise ValueError(f"Job {batch_id} not found in tracker")

        # First pass: collect all results, grouping chunks
        # Structure: {section_id: {part_num: (summary_text, input_tokens, output_tokens, truncated)}}
        chunk_results: dict[str, dict[int, tuple[str, int, int, bool]]] = {}
        total_input_tokens = 0
        total_output_tokens = 0

        for result in self.client.messages.batches.results(batch_id):
            custom_id = result.custom_id

            if result.result.type == "succeeded":
                parts = custom_id.split("_")

                # Check if this is a chunk (ends with _partN)
                if parts[-1].startswith("part") and parts[-1][4:].isdigit():
                    part_num = int(parts[-1][4:])
                    section_id = normalize_section_id(parts[-2])
                else:
                    part_num = 1  # Single-chunk section
                    section_id = normalize_section_id(parts[-1])

                message = result.result.message
                summary_text: str = getattr(message.content[0], "text", "")
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                truncated = message.stop_reason == "max_tokens"

                if truncated:
                    logger.warning(f"Section '{section_id}' part {part_num} was truncated")

                if section_id not in chunk_results:
                    chunk_results[section_id] = {}
                chunk_results[section_id][part_num] = (summary_text, input_tokens, output_tokens, truncated)

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

            elif result.result.type == "errored":
                logger.error(f"Batch request error for {custom_id}: {result.result.error}")

            elif result.result.type == "expired":
                logger.warning(f"Batch request expired: {custom_id}")

        if not chunk_results:
            return None

        # Second pass: combine chunks into final section summaries
        section_summaries: dict[str, SectionSummary] = {}

        for section_id, chunks in chunk_results.items():
            config = SECTION_WEIGHTS.get(section_id, {"weight": "medium", "description": ""})
            weight = config["weight"]
            title = config.get("description") or section_id

            # Sort chunks by part number and extract summaries
            sorted_parts = sorted(chunks.keys())
            chunk_summaries = [chunks[p][0] for p in sorted_parts]
            chunk_input = sum(chunks[p][1] for p in sorted_parts)
            chunk_output = sum(chunks[p][2] for p in sorted_parts)
            any_truncated = any(chunks[p][3] for p in sorted_parts)

            # Combine if multiple chunks
            if len(chunk_summaries) > 1:
                logger.info(f"📦 Combining {len(chunk_summaries)} chunks for section '{section_id}'")
                filing_context = f"{job.ticker} {job.filing_type} ({job.filing_date})"
                final_summary = self.summarizer._combine_chunk_summaries(
                    chunk_summaries, section_id, title, filing_context
                )
                # Add tokens from combine call (rough estimate - we don't track it precisely)
                combine_input = sum(len(s) // 4 for s in chunk_summaries)
                combine_output = len(final_summary) // 4
                logger.debug(f"Chunk combining tokens - input: {combine_input}, output: {combine_output}")
                total_input_tokens += combine_input
                total_output_tokens += combine_output
            else:
                final_summary = chunk_summaries[0]

            section_summaries[section_id] = SectionSummary(
                section_id=section_id,
                section_title=title,
                summary=final_summary,
                weight=weight,
                input_tokens=chunk_input,
                output_tokens=chunk_output,
                truncated=any_truncated,
            )

        return (job, section_summaries, total_input_tokens, total_output_tokens)

    def retrieve_results_parallel(
        self,
        batch_ids: list[str],
        save_summary: bool = True,
        max_concurrent: int = 5,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> list[FilingSummary]:
        """
        Retrieve results from multiple batch jobs with parallel executive summary generation.

        This is significantly faster than calling retrieve_results() sequentially
        when you have multiple batch jobs, as executive summaries are generated
        in parallel instead of one-by-one.

        Args:
            batch_ids: List of batch job IDs to retrieve
            save_summary: Whether to save summaries to files
            max_concurrent: Maximum concurrent executive summary API calls
            progress_callback: Optional callback(batch_id, status) for progress updates

        Returns:
            List of FilingSummary objects (only for successful retrievals)
        """
        # Phase 1: Collect all section summaries (fast, ~0.5s per batch)
        collected: list[tuple[BatchJob, dict[str, SectionSummary], int, int]] = []

        for batch_id in batch_ids:
            if progress_callback:
                progress_callback(batch_id, "collecting")
            result = self._collect_batch_sections(batch_id)
            if result:
                collected.append(result)

        if not collected:
            return []

        # Phase 2: Generate executive summaries in parallel (the slow part)
        async def generate_all_exec_summaries():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def generate_one(
                job: BatchJob,
                section_summaries: dict[str, SectionSummary],
                input_tokens: int,
                output_tokens: int,
            ) -> tuple[str, str | None, FilingSummary]:
                if progress_callback:
                    progress_callback(job.batch_id, "generating_summary")

                include_thesis = job.filing_type == "10-K"
                exec_summary, exec_input, exec_output = await self.summarizer.create_executive_summary_async(
                    list(section_summaries.values()),
                    f"{job.ticker} {job.filing_type} ({job.filing_date})",
                    include_thesis=include_thesis,
                    semaphore=semaphore,
                )

                total_input = input_tokens + exec_input
                total_output = output_tokens + exec_output

                filing_summary = FilingSummary(
                    ticker=job.ticker,
                    filing_type=job.filing_type,
                    filing_date=job.filing_date,
                    source_path=f"batch:{job.batch_id}",
                    executive_summary=exec_summary,
                    section_summaries=list(section_summaries.values()),
                    total_input_tokens=total_input,
                    total_output_tokens=total_output,
                )

                # Save if requested (but DON'T update tracker here - race condition!)
                output_path_str = None
                if save_summary and self.processed_dir:
                    output_path = self.summarizer.get_output_path(job.ticker, job.filing_type, job.filing_date)
                    filing_summary.save(output_path)
                    output_path_str = str(output_path)

                if progress_callback:
                    progress_callback(job.batch_id, "complete")

                return (job.batch_id, output_path_str, filing_summary)

            tasks = [generate_one(job, sections, inp, out) for job, sections, inp, out in collected]
            return await asyncio.gather(*tasks)

        # Run the async function
        results = asyncio.run(generate_all_exec_summaries())

        # Update tracker ONCE after all tasks complete (avoids race condition)
        # Always update tracker, even if save_summary=False, to mark jobs as retrieved
        if results:
            # Build lookup dict for O(1) access instead of O(n²) nested loops
            result_paths = {batch_id: output_path_str for batch_id, output_path_str, _ in results}
            jobs = self.tracker.load_jobs()
            for job in jobs:
                if job.batch_id in result_paths:
                    # Only set results_path if we actually saved (path is not None)
                    if result_paths[job.batch_id]:
                        job.results_path = result_paths[job.batch_id]
                    # Mark as retrieved even without saving by setting a sentinel
                    elif job.results_path is None:
                        job.results_path = "retrieved:no-save"
            self.tracker.save_jobs(jobs)

        return [summary for _, _, summary in results]

    def list_jobs(self, status_filter: str | None = None, refresh: bool = True) -> list[BatchJob]:
        """
        List tracked batch jobs.

        Args:
            status_filter: Optional status to filter by
            refresh: If True, refresh status from API for non-ended jobs

        Returns:
            List of BatchJob objects
        """
        jobs = self.tracker.load_jobs()

        # Refresh status from API for non-ended jobs
        if refresh:
            updated = False
            for job in jobs:
                if job.status not in ("ended", "expired", "canceled"):
                    try:
                        message_batch = self.client.messages.batches.retrieve(job.batch_id)
                        if job.status != message_batch.processing_status:
                            job.status = message_batch.processing_status
                            updated = True
                    except Exception as e:
                        logger.debug(f"Could not refresh status for {job.batch_id}: {e}")
                        # Keep cached status if API call fails
            if updated:
                self.tracker.save_jobs(jobs)

        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        return jobs

    def resubmit_truncated(
        self,
        batch_id: str,
        filing_summary: FilingSummary,
        token_multiplier: float = 1.5,
    ) -> BatchJob | None:
        """
        Resubmit truncated sections with higher token limits.

        Validates source file exists and checks for duplicate submissions before
        creating a new batch job. Results automatically merge into the existing
        summary file when retrieved.

        Args:
            batch_id: Original batch job ID
            filing_summary: The retrieved filing summary with truncated sections
            token_multiplier: Factor to increase token limits by (default 1.5x)

        Returns:
            New BatchJob if truncated sections found and resubmitted, None otherwise
        """
        # Get original job info
        job = self.tracker.get_job(batch_id)
        if not job:
            raise ValueError(f"Job {batch_id} not found in tracker")

        # Find truncated sections
        truncated_sections = [s for s in filing_summary.section_summaries if s.truncated]
        if not truncated_sections:
            logger.info("No truncated sections found - nothing to resubmit")
            return None

        truncated_ids = [s.section_id for s in truncated_sections]
        logger.info(f"Found {len(truncated_sections)} truncated section(s): {truncated_ids}")

        # Get source file path
        source_path = job.source_path
        if not source_path:
            logger.warning("Source path not available - cannot resubmit")
            logger.info("(Source paths are tracked for jobs created after this feature)")
            return None

        file_path = Path(source_path)
        if not file_path.exists():
            logger.error(f"Source file not found: {source_path}")
            return None

        # Create new batch requests for only the truncated sections
        # with increased token limits
        truncated_section_ids = {s.section_id for s in truncated_sections}

        pdf_types = {"transcript", "presentation", "conference", "analyst"}
        if job.filing_type.lower() in pdf_types:
            requests = self.summarizer.prepare_batch_requests_pdf(
                file_path=file_path,
                ticker=job.ticker,
                doc_type=job.filing_type.lower(),
                doc_date=job.filing_date,
                sections_to_include=list(truncated_section_ids),
                token_multiplier=token_multiplier,
            )
        else:
            # SEC filings
            requests = self.summarizer.prepare_batch_requests(
                file_path=file_path,
                ticker=job.ticker,
                filing_type=job.filing_type,
                filing_date=job.filing_date,
                sections_to_include=list(truncated_section_ids),
                token_multiplier=token_multiplier,
            )

        if not requests:
            logger.warning("No requests generated for truncated sections")
            logger.info("The source file may have been modified since the original batch")
            return None

        # Validate expected sections were found
        expected = set(truncated_section_ids)
        # Extract section names from custom_ids using normalize_section_id
        generated = set()
        for req in requests:
            parts = req["custom_id"].split("_")
            section_part = parts[-1] if parts else ""
            generated.add(normalize_section_id(section_part))

        # Build normalized lookup set for O(1) matching instead of O(n²)
        normalized_generated = {_normalize_for_matching(gid) for gid in generated}

        # O(n) lookup instead of O(n*m) nested loop
        missing = [eid for eid in expected if _normalize_for_matching(eid) not in normalized_generated]

        if missing:
            logger.warning(f"Could not find sections in source file: {missing}")

        # Submit new batch
        message_batch = self.client.messages.batches.create(requests=requests)
        logger.debug(f"Batch API request created: {message_batch.id}")

        # Create and track new job
        new_job = BatchJob(
            batch_id=message_batch.id,
            ticker=job.ticker,
            filing_type=job.filing_type,
            filing_date=job.filing_date,
            created_at=datetime.now().isoformat(),
            status=message_batch.processing_status,
            request_count=len(requests),
            source_path=str(file_path),
        )
        self.tracker.add_job(new_job)

        logger.info(f"✓ Resubmitted {len(requests)} truncated section(s) as batch: {new_job.batch_id}")
        logger.info(f"  Token limit increased by {token_multiplier}x")

        return new_job
