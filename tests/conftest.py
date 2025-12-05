"""Shared pytest fixtures."""

import json
import tempfile
from pathlib import Path

import pytest

# Standard test ticker - use a clearly fictional company for template genericity
TEST_TICKER = "ACME"
TEST_CIK = "0001234567"
TEST_COMPANY_NAME = "Acme Corporation"


@pytest.fixture
def test_ticker():
    """Provide the standard test ticker."""
    return TEST_TICKER


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for test indexes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_manifest_data():
    """Sample manifest data for testing."""
    return {
        "ticker": TEST_TICKER,
        "cik": TEST_CIK,
        "company_name": TEST_COMPANY_NAME,
        "last_updated": "2024-01-15T10:30:00",
        "filings": {
            "0001234567-24-000001": {
                "accession_number": "0001234567-24-000001",
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "company_name": TEST_COMPANY_NAME,
                "ticker": TEST_TICKER,
                "cik": TEST_CIK,
                "file_path": f"{TEST_TICKER}/10-K_2024-01-15_0001234567-24-000001.html",
                "file_hash": "abc123def456",
                "file_format": "html",
                "downloaded_at": "2024-01-15T10:30:00",
            }
        },
    }


@pytest.fixture
def sample_manifest_file(temp_data_dir, sample_manifest_data):
    """Create a sample manifest file."""
    company_dir = temp_data_dir / TEST_TICKER
    company_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = company_dir / "manifest.json"
    manifest_path.write_text(json.dumps(sample_manifest_data, indent=2))
    return manifest_path
