"""Tests for query expansion module."""

import pytest

from src.rag.query_expansion import ExpandedQuery, QueryExpander, expand_query, get_expander


class TestExpandedQuery:
    """Tests for ExpandedQuery dataclass."""

    def test_all_terms_includes_original(self):
        """all_terms should include original query."""
        eq = ExpandedQuery(original="NRR trends", expansions=["net revenue retention"])
        assert "NRR trends" in eq.all_terms
        assert "net revenue retention" in eq.all_terms

    def test_all_terms_empty_expansions(self):
        """all_terms with no expansions returns just original."""
        eq = ExpandedQuery(original="test query")
        assert eq.all_terms == ["test query"]

    def test_fts_query_no_expansions(self):
        """fts_query with no expansions returns original."""
        eq = ExpandedQuery(original="test query")
        assert eq.fts_query == "test query"

    def test_fts_query_with_expansions(self):
        """fts_query builds OR clause with expansions."""
        eq = ExpandedQuery(
            original="NRR",
            expansions=["net revenue retention"],
        )
        assert "OR" in eq.fts_query
        assert "NRR" in eq.fts_query
        assert '"net revenue retention"' in eq.fts_query  # Multi-word quoted

    def test_fts_query_single_word_expansion(self):
        """Single word expansions are not quoted."""
        eq = ExpandedQuery(
            original="revenue",
            expansions=["sales"],
        )
        fts = eq.fts_query
        assert "revenue OR sales" == fts


class TestQueryExpander:
    """Tests for QueryExpander class."""

    @pytest.fixture
    def expander(self):
        """Create a QueryExpander instance."""
        return QueryExpander()

    def test_expand_acronym_to_definition(self, expander):
        """Acronym should expand to full definition."""
        result = expander.expand("NRR trends")
        assert len(result.expansions) > 0
        assert any("net revenue retention" in exp.lower() for exp in result.expansions)
        assert len(result.acronyms_found) > 0

    def test_expand_multiple_acronyms(self, expander):
        """Multiple acronyms in query should all expand."""
        result = expander.expand("ARR and NRR metrics")
        assert len(result.acronyms_found) >= 2
        expansions_lower = [e.lower() for e in result.expansions]
        assert any("annual recurring revenue" in e for e in expansions_lower)
        assert any("net revenue retention" in e for e in expansions_lower)

    def test_expand_definition_to_acronym(self, expander):
        """Full term should expand to acronym."""
        result = expander.expand("net revenue retention growth")
        assert "NRR" in result.expansions

    def test_expand_synonym(self, expander):
        """Synonym should expand to related terms."""
        result = expander.expand("revenue growth")
        assert len(result.synonyms_found) > 0
        expansions_lower = [e.lower() for e in result.expansions]
        assert any("sales" in e for e in expansions_lower)

    def test_expand_churn_synonym(self, expander):
        """Churn should expand to attrition."""
        result = expander.expand("customer churn rate")
        expansions_lower = [e.lower() for e in result.expansions]
        assert any("attrition" in e for e in expansions_lower)

    def test_no_duplicate_expansions(self, expander):
        """Should not add duplicate expansions."""
        result = expander.expand("revenue sales top line")
        # Even though "revenue" and "sales" both have synonyms,
        # we shouldn't see duplicates
        assert len(result.expansions) == len(set(result.expansions))

    def test_no_expansion_when_term_in_query(self, expander):
        """Should not expand to a term already in the query."""
        result = expander.expand("revenue and sales trends")
        # "sales" is already in query, shouldn't appear in expansions
        assert "sales" not in result.expansions

    def test_case_insensitive_acronym(self, expander):
        """Acronym matching should be case-insensitive."""
        result1 = expander.expand("NRR")
        result2 = expander.expand("nrr")
        # Both should find the expansion
        assert len(result1.expansions) > 0
        assert len(result2.expansions) > 0

    def test_disable_acronyms(self):
        """Can disable acronym expansion."""
        expander = QueryExpander(enable_acronyms=False, enable_synonyms=True)
        result = expander.expand("NRR trends")
        # Should not expand NRR
        assert not any("net revenue retention" in e.lower() for e in result.expansions)

    def test_disable_synonyms(self):
        """Can disable synonym expansion."""
        expander = QueryExpander(enable_acronyms=True, enable_synonyms=False)
        result = expander.expand("revenue growth")
        # Should not expand synonyms
        assert not any("sales" in e.lower() for e in result.expansions)

    def test_get_acronym_definition(self, expander):
        """get_acronym_definition returns correct definition."""
        assert expander.get_acronym_definition("NRR") == "net revenue retention"
        assert expander.get_acronym_definition("ARR") == "annual recurring revenue"
        assert expander.get_acronym_definition("UNKNOWN") is None

    def test_get_synonyms(self, expander):
        """get_synonyms returns related terms."""
        synonyms = expander.get_synonyms("revenue")
        assert "sales" in synonyms
        assert "top line" in synonyms

    def test_get_synonyms_not_found(self, expander):
        """get_synonyms returns empty list for unknown term."""
        assert expander.get_synonyms("xyzabc") == []

    def test_multi_word_phrase_synonym(self, expander):
        """Multi-word phrases should expand to synonyms."""
        result = expander.expand("top line growth")
        assert len(result.synonyms_found) > 0
        expansions_lower = [e.lower() for e in result.expansions]
        assert any("revenue" in e for e in expansions_lower)

    def test_ebitda_expansion(self, expander):
        """EBITDA should expand to full form."""
        result = expander.expand("EBITDA margin")
        expansions_lower = [e.lower() for e in result.expansions]
        assert any("earnings before" in e for e in expansions_lower)

    def test_common_financial_acronyms(self, expander):
        """Test a variety of common financial acronyms."""
        acronyms = ["FCF", "ROE", "COGS", "SG&A", "CAPEX", "EPS", "P/E"]
        for acronym in acronyms:
            result = expander.expand(acronym)
            assert len(result.expansions) > 0, f"No expansion for {acronym}"


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_expander_returns_singleton(self):
        """get_expander should return same instance."""
        exp1 = get_expander()
        exp2 = get_expander()
        assert exp1 is exp2

    def test_expand_query_function(self):
        """expand_query convenience function works."""
        result = expand_query("NRR trends")
        assert isinstance(result, ExpandedQuery)
        assert len(result.expansions) > 0


class TestFTSQueryBuilding:
    """Tests for FTS query string building."""

    @pytest.fixture
    def expander(self):
        return QueryExpander()

    def test_fts_query_structure(self, expander):
        """FTS query should use OR and quote multi-word terms."""
        result = expander.expand("NRR")
        fts = result.fts_query
        # Should have original term
        assert "NRR" in fts
        # Should have OR
        assert " OR " in fts
        # Multi-word expansion should be quoted
        assert '"net revenue retention"' in fts

    def test_fts_query_multiple_expansions(self, expander):
        """Multiple expansions should all be in FTS query."""
        result = expander.expand("revenue")
        fts = result.fts_query
        # Should have multiple OR clauses
        assert fts.count(" OR ") >= 1
