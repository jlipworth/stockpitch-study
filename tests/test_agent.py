# Tests for temporal/comparative query detection (added Dec 21, 2025)


class TestTemporalDetection:
    """Tests for detect_temporal_refs function."""

    def test_detect_explicit_years(self):
        """Should detect explicit year references."""
        from src.rag.agent import detect_temporal_refs

        refs = detect_temporal_refs("What was revenue in 2023?")
        assert len(refs) == 1
        assert refs[0]["year"] == 2023

    def test_detect_multiple_years(self):
        """Should detect multiple years for comparisons."""
        from src.rag.agent import detect_temporal_refs

        refs = detect_temporal_refs("Compare 2023 vs 2024 performance")
        assert len(refs) == 2
        years = {r["year"] for r in refs}
        assert years == {2023, 2024}

    def test_detect_fiscal_year(self):
        """Should detect fiscal year references."""
        from src.rag.agent import detect_temporal_refs

        refs = detect_temporal_refs("What happened in FY24?")
        assert len(refs) == 1
        assert refs[0]["year"] == 2024
        assert refs[0]["type"] == "fiscal"

    def test_detect_quarter_with_year(self):
        """Should detect quarter + year references."""
        from src.rag.agent import detect_temporal_refs

        refs = detect_temporal_refs("Q2 2024 earnings")
        # Finds both the bare year and the quarter reference
        assert len(refs) >= 1
        # At least one should have both year and quarter
        quarter_refs = [r for r in refs if r.get("quarter")]
        assert len(quarter_refs) >= 1
        assert quarter_refs[0]["year"] == 2024
        assert quarter_refs[0]["quarter"] == "Q2"


class TestComparativeDetection:
    """Tests for is_comparative_query function."""

    def test_detect_vs(self):
        """Should detect 'vs' as comparative."""
        from src.rag.agent import is_comparative_query

        assert is_comparative_query("2023 vs 2024 revenue")
        assert is_comparative_query("compare margins vs last year")

    def test_detect_change(self):
        """Should detect 'change' as comparative."""
        from src.rag.agent import is_comparative_query

        assert is_comparative_query("How has strategy changed?")
        assert is_comparative_query("What changes occurred in margins?")

    def test_detect_evolution(self):
        """Should detect 'evolution/evolved' as comparative."""
        from src.rag.agent import is_comparative_query

        assert is_comparative_query("How has the business evolved?")
        assert is_comparative_query("Revenue evolution over time")

    def test_detect_yoy(self):
        """Should detect 'YoY' as comparative."""
        from src.rag.agent import is_comparative_query

        assert is_comparative_query("YoY revenue growth")
        assert is_comparative_query("year-over-year comparison")

    def test_non_comparative(self):
        """Should not flag non-comparative queries."""
        from src.rag.agent import is_comparative_query

        assert not is_comparative_query("What is the current revenue?")
        assert not is_comparative_query("Describe the business model")


class TestStructuredQuery:
    """Tests for StructuredQuery dataclass."""

    def test_date_filter_year_only(self):
        """Year-only filter should span full year."""
        from src.rag.agent import StructuredQuery

        sq = StructuredQuery(query="test", year=2024)
        start, end = sq.get_date_filter()
        assert start == "2024-01-01"
        assert end == "2024-12-31"

    def test_date_filter_with_quarter(self):
        """Quarter filter should span that quarter."""
        from src.rag.agent import StructuredQuery

        sq = StructuredQuery(query="test", year=2024, quarter="Q2")
        start, end = sq.get_date_filter()
        assert start == "2024-04-01"
        assert end == "2024-06-30"

    def test_date_filter_no_year(self):
        """No year should return None, None."""
        from src.rag.agent import StructuredQuery

        sq = StructuredQuery(query="test")
        start, end = sq.get_date_filter()
        assert start is None
        assert end is None
