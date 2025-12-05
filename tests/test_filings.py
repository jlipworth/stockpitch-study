"""Tests for the SEC filings fetcher module."""

from unittest.mock import patch

import pytest

from src.filings import (
    SUPPORTED_FILINGS,
    FilingMetadata,
    Manifest,
    SECFetcher,
    compute_file_hash,
)

from .conftest import TEST_CIK, TEST_COMPANY_NAME, TEST_TICKER


class TestFilingMetadata:
    """Tests for FilingMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating filing metadata."""
        accession = f"{TEST_CIK}-24-000001"
        metadata = FilingMetadata(
            accession_number=accession,
            filing_type="10-K",
            filing_date="2024-01-15",
            company_name=TEST_COMPANY_NAME,
            ticker=TEST_TICKER,
            cik=TEST_CIK,
            file_path=f"{TEST_TICKER}/10-K_2024-01-15.html",
            file_hash="abc123",
        )
        assert metadata.accession_number == accession
        assert metadata.filing_type == "10-K"
        assert metadata.ticker == TEST_TICKER
        assert metadata.downloaded_at is not None


class TestManifest:
    """Tests for Manifest dataclass."""

    def test_create_empty_manifest(self):
        """Test creating an empty manifest."""
        manifest = Manifest(ticker=TEST_TICKER, cik=TEST_CIK, company_name=TEST_COMPANY_NAME)
        assert manifest.ticker == TEST_TICKER
        assert manifest.filings == {}

    def test_save_and_load_manifest(self, temp_data_dir, sample_manifest_data):
        """Test saving and loading a manifest."""
        # Create manifest with filing - use key from sample_manifest_data
        accession_key = list(sample_manifest_data["filings"].keys())[0]
        metadata = FilingMetadata(**sample_manifest_data["filings"][accession_key])
        manifest = Manifest(
            ticker=TEST_TICKER,
            cik=TEST_CIK,
            company_name=TEST_COMPANY_NAME,
            filings={accession_key: metadata},
        )

        # Save
        manifest_path = temp_data_dir / "manifest.json"
        manifest.save(manifest_path)
        assert manifest_path.exists()

        # Load
        loaded = Manifest.load(manifest_path)
        assert loaded.ticker == TEST_TICKER
        assert loaded.cik == TEST_CIK
        assert accession_key in loaded.filings
        assert loaded.filings[accession_key].filing_type == "10-K"

    def test_load_existing_manifest(self, sample_manifest_file):
        """Test loading an existing manifest file."""
        manifest = Manifest.load(sample_manifest_file)
        assert manifest.ticker == TEST_TICKER
        assert manifest.company_name == TEST_COMPANY_NAME
        assert len(manifest.filings) == 1


class TestComputeFileHash:
    """Tests for file hashing."""

    def test_hash_bytes(self):
        """Test hashing byte content."""
        content = b"Hello, World!"
        hash1 = compute_file_hash(content)
        hash2 = compute_file_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex digest length

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        hash1 = compute_file_hash(b"Content A")
        hash2 = compute_file_hash(b"Content B")
        assert hash1 != hash2


class TestSECFetcher:
    """Tests for SECFetcher class."""

    def test_init_without_identity(self, temp_data_dir):
        """Test initializing fetcher without user agent."""
        fetcher = SECFetcher(data_dir=temp_data_dir)
        assert fetcher.data_dir == temp_data_dir

    @patch("src.filings.fetcher.set_identity")
    def test_init_with_identity(self, mock_set_identity, temp_data_dir):
        """Test initializing fetcher with user agent."""
        SECFetcher(data_dir=temp_data_dir, user_agent="Test test@example.com")
        mock_set_identity.assert_called_once_with("Test test@example.com")

    def test_get_company_dir_creates_directory(self, temp_data_dir):
        """Test that company directory is created."""
        fetcher = SECFetcher(data_dir=temp_data_dir)
        company_dir = fetcher._get_company_dir(TEST_TICKER)
        assert company_dir.exists()
        assert company_dir.name == TEST_TICKER

    def test_get_company_dir_uppercase(self, temp_data_dir):
        """Test that ticker is uppercased."""
        fetcher = SECFetcher(data_dir=temp_data_dir)
        company_dir = fetcher._get_company_dir(TEST_TICKER.lower())
        assert company_dir.name == TEST_TICKER

    def test_estimate_filing_count(self, temp_data_dir):
        """Test filing count estimation."""
        fetcher = SECFetcher(data_dir=temp_data_dir)
        assert fetcher._estimate_filing_count("10-K", 3) == 3  # 1 per year
        assert fetcher._estimate_filing_count("10-Q", 3) == 9  # 3 per year
        assert fetcher._estimate_filing_count("8-K", 2) == 20  # 10 per year

    def test_get_manifest_new(self, temp_data_dir):
        """Test getting manifest for new company."""
        fetcher = SECFetcher(data_dir=temp_data_dir)
        manifest = fetcher._get_manifest(TEST_TICKER)
        assert manifest.ticker == TEST_TICKER
        assert manifest.filings == {}

    def test_get_manifest_existing(self, temp_data_dir, sample_manifest_data):
        """Test getting existing manifest."""
        # Create manifest file
        company_dir = temp_data_dir / TEST_TICKER
        company_dir.mkdir(parents=True)
        manifest_path = company_dir / "manifest.json"
        import json

        manifest_path.write_text(json.dumps(sample_manifest_data))

        fetcher = SECFetcher(data_dir=temp_data_dir)
        manifest = fetcher._get_manifest(TEST_TICKER)
        assert manifest.ticker == TEST_TICKER
        assert len(manifest.filings) == 1

    def test_fetch_validates_filing_types(self, temp_data_dir):
        """Test that fetch validates filing types."""
        fetcher = SECFetcher(data_dir=temp_data_dir)
        with pytest.raises(ValueError, match="Unsupported filing type"):
            fetcher.fetch(TEST_TICKER, filing_types=["INVALID"])  # type: ignore[arg-type]


class TestSupportedFilings:
    """Tests for supported filing types."""

    def test_all_types_present(self):
        """Test all expected filing types are supported."""
        assert "10-K" in SUPPORTED_FILINGS
        assert "10-Q" in SUPPORTED_FILINGS
        assert "8-K" in SUPPORTED_FILINGS
        assert "DEF 14A" in SUPPORTED_FILINGS
        assert "4" in SUPPORTED_FILINGS

    def test_count(self):
        """Test correct number of supported types."""
        assert len(SUPPORTED_FILINGS) == 5


# Integration tests (require network access)
# Run with: pytest -m integration
@pytest.mark.integration
class TestSECFetcherIntegration:
    """Integration tests that hit the real SEC API."""

    @pytest.fixture
    def fetcher(self, temp_data_dir):
        """Create fetcher with test identity."""
        return SECFetcher(
            data_dir=temp_data_dir,
            user_agent="StockPitchTest test@example.com",
        )

    def test_fetch_single_10k(self, fetcher, temp_data_dir):
        """Test fetching a single 10-K filing."""
        # Fetch just 1 year of 10-K for a well-known company
        downloaded = fetcher.fetch("AAPL", filing_types=["10-K"], years=1)

        assert len(downloaded) >= 1
        assert downloaded[0].filing_type == "10-K"
        assert downloaded[0].ticker == "AAPL"

        # Check file was created
        file_path = temp_data_dir / downloaded[0].file_path
        assert file_path.exists()

        # Check manifest was created
        manifest_path = temp_data_dir / "AAPL" / "manifest.json"
        assert manifest_path.exists()
