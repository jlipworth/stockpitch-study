"""Tests for CLI commands."""

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


class TestFetchSince:
    """Tests for the --since flag on the fetch command."""

    def test_since_with_existing_manifest(self, temp_data_dir, sample_manifest_data):
        """Test --since uses manifest's last_updated date."""
        # Create manifest
        company_dir = temp_data_dir / "AAPL"
        company_dir.mkdir(parents=True)
        manifest_path = company_dir / "manifest.json"
        manifest_path.write_text(json.dumps(sample_manifest_data))

        with patch("src.cli.fetch.DATA_DIR", temp_data_dir):
            with patch("src.cli.fetch.SECFetcher") as mock_fetcher_class:
                mock_fetcher = MagicMock()
                mock_fetcher.fetch.return_value = []
                mock_fetcher_class.return_value = mock_fetcher

                result = runner.invoke(app, ["fetch", "AAPL", "--since"])

                assert result.exit_code == 0
                assert "Fetching filings since: 2024-01-15" in result.stdout

                # Verify fetcher was called with the correct start_date
                mock_fetcher.fetch.assert_called_once()
                call_kwargs = mock_fetcher.fetch.call_args.kwargs
                assert call_kwargs["start_date"].strftime("%Y-%m-%d") == "2024-01-15"

    def test_since_without_manifest(self, temp_data_dir):
        """Test --since falls back to defaults when no manifest exists."""
        with patch("src.cli.fetch.DATA_DIR", temp_data_dir):
            with patch("src.cli.fetch.SECFetcher") as mock_fetcher_class:
                mock_fetcher = MagicMock()
                mock_fetcher.fetch.return_value = []
                mock_fetcher_class.return_value = mock_fetcher

                result = runner.invoke(app, ["fetch", "NEWCO", "--since"])

                assert result.exit_code == 0
                assert "No manifest found" in result.stdout

                # Verify fetcher was called with no start_date
                mock_fetcher.fetch.assert_called_once()
                call_kwargs = mock_fetcher.fetch.call_args.kwargs
                assert call_kwargs["start_date"] is None

    def test_since_conflicts_with_start_date(self, temp_data_dir):
        """Test --since cannot be used with --start-date."""
        with patch("src.cli.fetch.DATA_DIR", temp_data_dir):
            result = runner.invoke(app, ["fetch", "AAPL", "--since", "--start-date", "2024-01-01"])

            assert result.exit_code == 1
            assert "cannot be used with --start-date" in result.stdout

    def test_since_conflicts_with_end_date(self, temp_data_dir):
        """Test --since cannot be used with --end-date."""
        with patch("src.cli.fetch.DATA_DIR", temp_data_dir):
            result = runner.invoke(app, ["fetch", "AAPL", "--since", "--end-date", "2024-12-31"])

            assert result.exit_code == 1
            assert "cannot be used with" in result.stdout

    def test_since_with_empty_last_updated(self, temp_data_dir):
        """Test --since handles manifest with empty last_updated."""
        company_dir = temp_data_dir / "AAPL"
        company_dir.mkdir(parents=True)
        manifest_path = company_dir / "manifest.json"
        manifest_data = {
            "ticker": "AAPL",
            "cik": "320193",
            "company_name": "Apple Inc.",
            "last_updated": "",
            "filings": {},
        }
        manifest_path.write_text(json.dumps(manifest_data))

        with patch("src.cli.fetch.DATA_DIR", temp_data_dir):
            with patch("src.cli.fetch.SECFetcher") as mock_fetcher_class:
                mock_fetcher = MagicMock()
                mock_fetcher.fetch.return_value = []
                mock_fetcher_class.return_value = mock_fetcher

                result = runner.invoke(app, ["fetch", "AAPL", "--since"])

                assert result.exit_code == 0
                assert "no last_updated" in result.stdout
