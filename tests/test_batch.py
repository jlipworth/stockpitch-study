"""Tests for the batch processing module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.summarizer import SECTION_WEIGHTS
from src.summarizer.batch import (
    BatchJob,
    BatchJobTracker,
    BatchProcessor,
)


class TestBatchJob:
    """Tests for BatchJob dataclass."""

    def test_batch_job_creation(self):
        job = BatchJob(
            batch_id="msgbatch_123",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            created_at="2024-01-15T12:00:00",
            status="in_progress",
            request_count=10,
        )

        assert job.batch_id == "msgbatch_123"
        assert job.ticker == "AAPL"
        assert job.status == "in_progress"
        assert job.request_count == 10
        assert job.results_path is None

    def test_to_dict(self):
        job = BatchJob(
            batch_id="msgbatch_123",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            created_at="2024-01-15T12:00:00",
            status="in_progress",
            request_count=10,
        )

        data = job.to_dict()

        assert data["batch_id"] == "msgbatch_123"
        assert data["ticker"] == "AAPL"
        assert "results_path" in data

    def test_from_dict(self):
        data = {
            "batch_id": "msgbatch_456",
            "ticker": "MSFT",
            "filing_type": "10-Q",
            "filing_date": "2024-03-31",
            "created_at": "2024-04-01T10:00:00",
            "status": "ended",
            "request_count": 5,
            "results_path": "/path/to/results.md",
        }

        job = BatchJob.from_dict(data)

        assert job.batch_id == "msgbatch_456"
        assert job.ticker == "MSFT"
        assert job.status == "ended"
        assert job.results_path == "/path/to/results.md"


class TestBatchJobTracker:
    """Tests for BatchJobTracker."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_tracker_initialization(self, temp_dir):
        manifest_path = temp_dir / "batch_jobs.json"
        tracker = BatchJobTracker(manifest_path)

        assert tracker.manifest_path == manifest_path

    def test_load_empty(self, temp_dir):
        tracker = BatchJobTracker(temp_dir / "batch_jobs.json")
        jobs = tracker.load_jobs()

        assert jobs == []

    def test_add_and_load_job(self, temp_dir):
        tracker = BatchJobTracker(temp_dir / "batch_jobs.json")

        job = BatchJob(
            batch_id="msgbatch_123",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            created_at="2024-01-15T12:00:00",
            status="in_progress",
            request_count=10,
        )

        tracker.add_job(job)
        loaded_jobs = tracker.load_jobs()

        assert len(loaded_jobs) == 1
        assert loaded_jobs[0].batch_id == "msgbatch_123"

    def test_update_job(self, temp_dir):
        tracker = BatchJobTracker(temp_dir / "batch_jobs.json")

        job = BatchJob(
            batch_id="msgbatch_123",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            created_at="2024-01-15T12:00:00",
            status="in_progress",
            request_count=10,
        )
        tracker.add_job(job)

        updated = tracker.update_job("msgbatch_123", status="ended")

        assert updated is not None
        assert updated.status == "ended"

        # Verify persisted
        loaded = tracker.load_jobs()
        assert loaded[0].status == "ended"

    def test_update_nonexistent_job(self, temp_dir):
        tracker = BatchJobTracker(temp_dir / "batch_jobs.json")

        result = tracker.update_job("nonexistent", status="ended")
        assert result is None

    def test_get_job(self, temp_dir):
        tracker = BatchJobTracker(temp_dir / "batch_jobs.json")

        job = BatchJob(
            batch_id="msgbatch_123",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
            created_at="2024-01-15T12:00:00",
            status="in_progress",
            request_count=10,
        )
        tracker.add_job(job)

        found = tracker.get_job("msgbatch_123")
        assert found is not None
        assert found.ticker == "AAPL"

        not_found = tracker.get_job("nonexistent")
        assert not_found is None

    def test_get_pending_jobs(self, temp_dir):
        tracker = BatchJobTracker(temp_dir / "batch_jobs.json")

        jobs = [
            BatchJob("batch_1", "AAPL", "10-K", "2024-01-15", "2024-01-15T12:00:00", "in_progress", 5),
            BatchJob("batch_2", "MSFT", "10-K", "2024-01-15", "2024-01-15T12:00:00", "ended", 5),
            BatchJob("batch_3", "GOOGL", "10-K", "2024-01-15", "2024-01-15T12:00:00", "in_progress", 5),
        ]

        for job in jobs:
            tracker.add_job(job)

        pending = tracker.get_pending_jobs()

        assert len(pending) == 2
        batch_ids = [j.batch_id for j in pending]
        assert "batch_1" in batch_ids
        assert "batch_3" in batch_ids


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_requires_api_key(self, temp_dir):
        """Should fail without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                BatchProcessor(processed_dir=temp_dir)

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_init_with_api_key(self, mock_summarizer, mock_anthropic, temp_dir):
        """Should initialize with API key."""
        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        assert processor.processed_dir == temp_dir
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_list_jobs_empty(self, mock_summarizer, mock_anthropic, temp_dir):
        """Should return empty list when no jobs."""
        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        jobs = processor.list_jobs()
        assert jobs == []

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_submit_batch(self, mock_summarizer, mock_anthropic, temp_dir):
        """Should submit batch and track job."""
        # Setup mocks
        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.prepare_batch_requests.return_value = [
            {
                "custom_id": "AAPL_10-K_2024-01-15_Item 1",
                "params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 2000,
                    "system": "prompt",
                    "messages": [{"role": "user", "content": "test"}],
                },
            }
        ]
        mock_summarizer.return_value = mock_summarizer_instance

        mock_client = MagicMock()
        mock_batch_response = MagicMock()
        mock_batch_response.id = "msgbatch_test123"
        mock_batch_response.processing_status = "in_progress"
        mock_client.messages.batches.create.return_value = mock_batch_response
        # Also mock retrieve for refresh logic in list_jobs
        mock_client.messages.batches.retrieve.return_value = mock_batch_response
        mock_anthropic.return_value = mock_client

        # Create a dummy file
        dummy_file = temp_dir / "test.html"
        dummy_file.write_text("<html>Test</html>")

        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        job = processor.submit_batch(
            file_path=dummy_file,
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-01-15",
        )

        assert job.batch_id == "msgbatch_test123"
        assert job.ticker == "AAPL"
        assert job.status == "in_progress"

        # Verify job was tracked
        tracked_jobs = processor.list_jobs()
        assert len(tracked_jobs) == 1

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_check_status(self, mock_summarizer, mock_anthropic, temp_dir):
        """Should check batch status."""
        mock_client = MagicMock()
        mock_batch_response = MagicMock()
        mock_batch_response.processing_status = "ended"
        mock_batch_response.created_at = "2024-01-15T12:00:00Z"
        mock_batch_response.ended_at = "2024-01-15T13:00:00Z"
        mock_batch_response.request_counts.processing = 0
        mock_batch_response.request_counts.succeeded = 10
        mock_batch_response.request_counts.errored = 0
        mock_batch_response.request_counts.canceled = 0
        mock_batch_response.request_counts.expired = 0
        mock_client.messages.batches.retrieve.return_value = mock_batch_response
        mock_anthropic.return_value = mock_client

        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        status, details = processor.check_status("msgbatch_test123")

        assert status == "ended"
        assert details["request_counts"]["succeeded"] == 10


class TestSectionWeights:
    """Tests for section weight configuration in batch context."""

    def test_high_weight_sections_defined(self):
        """High priority sections should have higher max_tokens."""
        for section_id in ["Item 1", "Item 1A", "Item 7"]:
            assert section_id in SECTION_WEIGHTS
            assert SECTION_WEIGHTS[section_id]["weight"] == "high"
            assert SECTION_WEIGHTS[section_id]["max_tokens"] >= 2000

    def test_low_weight_sections_defined(self):
        """Low priority sections should have lower max_tokens."""
        for section_id in ["Item 15", "Item 16"]:
            assert section_id in SECTION_WEIGHTS
            assert SECTION_WEIGHTS[section_id]["weight"] == "low"
            assert SECTION_WEIGHTS[section_id]["max_tokens"] <= 1000


class TestFilingSummaryMarkdownParsing:
    """Tests for FilingSummary.from_markdown() parsing."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_from_markdown_basic(self, temp_dir):
        """Test parsing a basic summary file."""
        from src.summarizer import FilingSummary

        # Create a sample markdown file
        content = """# AAPL 10-K Analysis
**Filing Date:** 2024-09-30
**Generated:** 2024-10-15T10:30:00
**Tokens Used:** 12,345 in / 2,500 out

---

## Executive Summary

This is the executive summary with key findings.

---

# Section Analysis

## Key Sections

### Item 1: Business Description

This section describes the business.

### Item 1A: Risk Factors

Risk factors are outlined here.

## Supporting Sections

### Item 7: Management's Discussion and Analysis

MD&A content here.
"""
        summary_path = temp_dir / "test_summary.md"
        summary_path.write_text(content, encoding="utf-8")

        # Parse it
        filing = FilingSummary.from_markdown(summary_path)

        # Verify fields
        assert filing.ticker == "AAPL"
        assert filing.filing_type == "10-K"
        assert filing.filing_date == "2024-09-30"
        assert filing.total_input_tokens == 12345
        assert filing.total_output_tokens == 2500
        assert "executive summary" in filing.executive_summary.lower()
        assert len(filing.section_summaries) == 3

        # Check sections
        section_ids = {s.section_id for s in filing.section_summaries}
        assert "Item 1" in section_ids
        assert "Item 1A" in section_ids
        assert "Item 7" in section_ids

    def test_from_markdown_with_truncated_sections(self, temp_dir):
        """Test parsing markdown with TRUNCATED markers."""
        from src.summarizer import FilingSummary

        content = """# MSFT 10-Q Analysis
**Filing Date:** 2024-06-30
**Generated:** 2024-07-15T14:00:00
**Tokens Used:** 8,000 in / 1,500 out

---

## Executive Summary

Summary text.

---

# Section Analysis

## Key Sections

### Item 1: Financial Statements

Normal section content.

### Item 7: MD&A ⚠️ TRUNCATED

This section was truncated due to hitting max_tokens limit.

## Supporting Sections

### Item 8: Other Info

Another section.
"""
        summary_path = temp_dir / "test_truncated.md"
        summary_path.write_text(content, encoding="utf-8")

        filing = FilingSummary.from_markdown(summary_path)

        # Find the truncated section
        truncated_sections = [s for s in filing.section_summaries if s.truncated]
        assert len(truncated_sections) == 1
        assert truncated_sections[0].section_id == "Item 7"

        # Verify non-truncated sections
        normal_sections = [s for s in filing.section_summaries if not s.truncated]
        assert len(normal_sections) == 2

    def test_from_markdown_ticker_with_hyphen(self, temp_dir):
        """Test parsing with ticker like BRK-B."""
        from src.summarizer import FilingSummary

        content = """# BRK 10-K Analysis
**Filing Date:** 2024-12-31
**Generated:** 2025-01-15T10:00:00
**Tokens Used:** 5,000 in / 1,000 out

---

## Executive Summary

Berkshire Hathaway summary.

---

# Section Analysis

## Key Sections

### Item 1: Business Description

Content here.
"""
        summary_path = temp_dir / "test_hyphen.md"
        summary_path.write_text(content, encoding="utf-8")

        filing = FilingSummary.from_markdown(summary_path)
        assert filing.ticker == "BRK"

    def test_from_markdown_with_bom(self, temp_dir):
        """Test parsing file with UTF-8 BOM."""
        from src.summarizer import FilingSummary

        content = """# GOOGL 10-K Analysis
**Filing Date:** 2024-12-31
**Generated:** 2025-01-15T10:00:00
**Tokens Used:** 10,000 in / 2,000 out

---

## Executive Summary

Google summary.

---

# Section Analysis

## Key Sections

### Item 1: Business Description

Content.
"""
        summary_path = temp_dir / "test_bom.md"
        # Write with BOM
        with open(summary_path, "w", encoding="utf-8-sig") as f:
            f.write(content)

        # Note: Python's read_text with utf-8 encoding automatically strips BOM
        # This test verifies the from_markdown parser is compatible with files
        # that may have been saved with BOM (common in Windows editors)
        filing = FilingSummary.from_markdown(summary_path)
        assert filing.ticker == "GOOGL"


class TestMergeWorkflow:
    """Tests for section merging in retrieve_results()."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_retrieve_results_merges_existing_summary(self, mock_summarizer_class, mock_anthropic, temp_dir):
        """Test that retrieve_results merges new sections into existing summary."""

        # 1. Create an existing summary file with sections A, B (truncated), C
        existing_content = """# AAPL 10-K Analysis
**Filing Date:** 2024-09-30
**Generated:** 2024-10-15T10:00:00
**Tokens Used:** 10,000 in / 2,000 out

---

## Executive Summary

Old executive summary.

---

# Section Analysis

## Key Sections

### Item 1: Business Description

Original Item 1 content.

### Item 1A: Risk Factors ⚠️ TRUNCATED

Truncated risk factors section.

### Item 7: Management's Discussion and Analysis

Original Item 7 content.
"""
        output_path = temp_dir / "AAPL_10-K_2024-09-30.md"
        output_path.write_text(existing_content, encoding="utf-8")

        # 2. Setup mocks for batch retrieval (only returning Item 1A - the resubmit)
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock batch status
        mock_status_response = MagicMock()
        mock_status_response.processing_status = "ended"
        mock_client.messages.batches.retrieve.return_value = mock_status_response

        # Mock batch results - only Item 1A (resubmitted section)
        mock_result = MagicMock()
        mock_result.custom_id = "AAPL_10-K_2024-09-30_Item-1A"
        mock_result.result.type = "succeeded"
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="NEW Item 1A content - no longer truncated.")]
        mock_message.usage.input_tokens = 2000
        mock_message.usage.output_tokens = 500
        mock_message.stop_reason = "end_turn"  # Not truncated anymore
        mock_result.result.message = mock_message
        mock_client.messages.batches.results.return_value = [mock_result]

        # Mock summarizer for executive summary generation
        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.get_output_path.return_value = output_path
        mock_summarizer_instance.create_executive_summary.return_value = (
            "NEW executive summary with all sections.",
            1000,
            200,
        )
        mock_summarizer_class.return_value = mock_summarizer_instance

        # Create processor
        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        # Add a job to tracker
        job = BatchJob(
            batch_id="msgbatch_resubmit",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-09-30",
            created_at="2024-10-15T12:00:00",
            status="ended",
            request_count=1,
            source_path=str(temp_dir / "test.html"),
        )
        processor.tracker.add_job(job)

        # 3. Call retrieve_results
        result = processor.retrieve_results(batch_id="msgbatch_resubmit", save_summary=True)

        # 4. Assert result has sections Item 1, Item 1A (new version), Item 7
        assert result is not None
        assert len(result.section_summaries) == 3

        section_map = {s.section_id: s for s in result.section_summaries}
        assert "Item 1" in section_map
        assert "Item 1A" in section_map
        assert "Item 7" in section_map

        # Item 1A should have new content and not be truncated
        assert "NEW Item 1A content" in section_map["Item 1A"].summary
        assert not section_map["Item 1A"].truncated

        # Item 1 and Item 7 should have original content
        assert "Original Item 1 content" in section_map["Item 1"].summary
        assert "Original Item 7 content" in section_map["Item 7"].summary

        # 5. Assert executive summary was regenerated
        assert result.executive_summary == "NEW executive summary with all sections."

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_retrieve_results_no_existing_file(self, mock_summarizer_class, mock_anthropic, temp_dir):
        """Test retrieve_results when no existing summary exists."""
        # Setup mocks
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_status_response = MagicMock()
        mock_status_response.processing_status = "ended"
        mock_client.messages.batches.retrieve.return_value = mock_status_response

        # Mock batch results - two sections
        results = []
        for section_id in ["Item 1", "Item 1A"]:
            mock_result = MagicMock()
            mock_result.custom_id = f"MSFT_10-Q_2024-06-30_{section_id.replace(' ', '-')}"
            mock_result.result.type = "succeeded"
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=f"{section_id} content.")]
            mock_message.usage.input_tokens = 1000
            mock_message.usage.output_tokens = 200
            mock_message.stop_reason = "end_turn"
            mock_result.result.message = mock_message
            results.append(mock_result)
        mock_client.messages.batches.results.return_value = results

        # Mock summarizer
        mock_summarizer_instance = MagicMock()
        output_path = temp_dir / "MSFT_10-Q_2024-06-30.md"
        mock_summarizer_instance.get_output_path.return_value = output_path
        mock_summarizer_instance.create_executive_summary.return_value = (
            "Executive summary.",
            500,
            100,
        )
        mock_summarizer_class.return_value = mock_summarizer_instance

        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        # Add job
        job = BatchJob(
            batch_id="msgbatch_new",
            ticker="MSFT",
            filing_type="10-Q",
            filing_date="2024-06-30",
            created_at="2024-07-01T10:00:00",
            status="ended",
            request_count=2,
        )
        processor.tracker.add_job(job)

        # Retrieve results
        result = processor.retrieve_results(batch_id="msgbatch_new", save_summary=True)

        # Should create new summary with all sections from batch
        assert result is not None
        assert len(result.section_summaries) == 2

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_retrieve_results_handles_section_id_variations(self, mock_summarizer_class, mock_anthropic, temp_dir):
        """Test that Q&A and Q and A are treated as same section."""

        # Existing file has "Q and A"
        existing_content = """# DT transcript Analysis
**Filing Date:** 2025-11-06
**Generated:** 2025-11-07T10:00:00
**Tokens Used:** 5,000 in / 1,000 out

---

## Executive Summary

Old summary.

---

# Section Analysis

## Key Sections

### Prepared Remarks: Management's prepared comments

Prepared remarks content.

### Q and A: Q and A ⚠️ TRUNCATED

Old truncated Q&A.
"""
        output_path = temp_dir / "DT_transcript_2025-11-06.md"
        output_path.write_text(existing_content, encoding="utf-8")

        # Setup mocks - batch returns "Q and A"
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_status_response = MagicMock()
        mock_status_response.processing_status = "ended"
        mock_client.messages.batches.retrieve.return_value = mock_status_response

        # Batch result with "Q and A" section
        mock_result = MagicMock()
        mock_result.custom_id = "DT_transcript_2025-11-06_Q-and-A"
        mock_result.result.type = "succeeded"
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="NEW Q&A content - complete now.")]
        mock_message.usage.input_tokens = 3000
        mock_message.usage.output_tokens = 800
        mock_message.stop_reason = "end_turn"
        mock_result.result.message = mock_message
        mock_client.messages.batches.results.return_value = [mock_result]

        # Mock summarizer
        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.get_output_path.return_value = output_path
        mock_summarizer_instance.create_executive_summary.return_value = (
            "New summary.",
            1000,
            200,
        )
        mock_summarizer_class.return_value = mock_summarizer_instance

        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        # Add job
        job = BatchJob(
            batch_id="msgbatch_qa",
            ticker="DT",
            filing_type="transcript",
            filing_date="2025-11-06",
            created_at="2025-11-07T12:00:00",
            status="ended",
            request_count=1,
        )
        processor.tracker.add_job(job)

        # Retrieve
        result = processor.retrieve_results(batch_id="msgbatch_qa", save_summary=True)

        # Should have 2 sections, with Q&A replaced (not duplicated)
        assert result is not None
        assert len(result.section_summaries) == 2

        section_map = {s.section_id: s for s in result.section_summaries}
        assert "Prepared Remarks" in section_map
        # After replacement, the section ID is "Q and A" (from batch result parsing)
        # The old "Q and A" was replaced by fuzzy matching
        assert "Q&A" in section_map  # Batch results normalize hyphens to &

        # Q&A should have new content
        qa_section = section_map["Q&A"]
        assert "NEW Q&A content" in qa_section.summary
        assert not qa_section.truncated


class TestResubmitWorkflow:
    """Tests for resubmit_truncated workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_resubmit_truncated_creates_correct_batch(self, mock_summarizer_class, mock_anthropic, temp_dir):
        """Test that resubmit only includes truncated sections."""
        from src.summarizer import FilingSummary, SectionSummary

        # Create filing summary with truncated and non-truncated sections
        sections = [
            SectionSummary(
                section_id="Item 1",
                section_title="Business",
                summary="Complete content",
                weight="high",
                input_tokens=1000,
                output_tokens=200,
                truncated=False,
            ),
            SectionSummary(
                section_id="Item 1A",
                section_title="Risk Factors",
                summary="Truncated content",
                weight="high",
                input_tokens=2000,
                output_tokens=500,
                truncated=True,
            ),
            SectionSummary(
                section_id="Item 7",
                section_title="MD&A",
                summary="Truncated MD&A",
                weight="high",
                input_tokens=3000,
                output_tokens=800,
                truncated=True,
            ),
        ]
        filing = FilingSummary(
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-09-30",
            source_path="test.html",
            executive_summary="Summary",
            section_summaries=sections,
            total_input_tokens=6000,
            total_output_tokens=1500,
        )

        # Setup mocks
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock batch submission
        mock_batch_response = MagicMock()
        mock_batch_response.id = "msgbatch_resubmit"
        mock_batch_response.processing_status = "in_progress"
        mock_client.messages.batches.create.return_value = mock_batch_response
        mock_client.messages.batches.retrieve.return_value = mock_batch_response

        # Mock summarizer prepare_batch_requests
        mock_summarizer_instance = MagicMock()
        # Should be called with only truncated section IDs
        mock_summarizer_instance.prepare_batch_requests.return_value = [
            {"custom_id": "AAPL_10-K_2024-09-30_Item-1A", "params": {"max_tokens": 3000}},
            {"custom_id": "AAPL_10-K_2024-09-30_Item-7", "params": {"max_tokens": 4500}},
        ]
        mock_summarizer_class.return_value = mock_summarizer_instance

        # Create source file
        source_file = temp_dir / "test.html"
        source_file.write_text("<html>Test</html>")

        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        # Add original job with source_path
        original_job = BatchJob(
            batch_id="msgbatch_original",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-09-30",
            created_at="2024-10-15T10:00:00",
            status="ended",
            request_count=3,
            source_path=str(source_file),
        )
        processor.tracker.add_job(original_job)

        # Call resubmit_truncated
        new_job = processor.resubmit_truncated("msgbatch_original", filing, token_multiplier=1.5)

        # Verify new job was created
        assert new_job is not None
        assert new_job.batch_id == "msgbatch_resubmit"

        # Verify prepare_batch_requests was called with only truncated sections
        mock_summarizer_instance.prepare_batch_requests.assert_called_once()
        call_args = mock_summarizer_instance.prepare_batch_requests.call_args
        sections_included = call_args.kwargs.get("sections_to_include")
        assert set(sections_included) == {"Item 1A", "Item 7"}
        assert call_args.kwargs.get("token_multiplier") == 1.5

    @patch("src.summarizer.batch.Anthropic")
    @patch("src.summarizer.batch.Summarizer")
    def test_resubmit_truncated_missing_source_path(self, mock_summarizer_class, mock_anthropic, temp_dir):
        """Test error handling when source_path is None."""
        from src.summarizer import FilingSummary, SectionSummary

        # Create filing with truncated section
        sections = [
            SectionSummary(
                section_id="Item 1A",
                section_title="Risk Factors",
                summary="Truncated",
                weight="high",
                input_tokens=2000,
                output_tokens=500,
                truncated=True,
            )
        ]
        filing = FilingSummary(
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-09-30",
            source_path="test.html",
            executive_summary="Summary",
            section_summaries=sections,
            total_input_tokens=6000,
            total_output_tokens=1500,
        )

        # Setup mocks
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        processor = BatchProcessor(api_key="test-key", processed_dir=temp_dir)

        # Add job WITHOUT source_path (legacy job)
        job = BatchJob(
            batch_id="msgbatch_legacy",
            ticker="AAPL",
            filing_type="10-K",
            filing_date="2024-09-30",
            created_at="2024-10-15T10:00:00",
            status="ended",
            request_count=1,
            source_path=None,  # Missing!
        )
        processor.tracker.add_job(job)

        # Should return None and print warning
        result = processor.resubmit_truncated("msgbatch_legacy", filing)
        assert result is None
