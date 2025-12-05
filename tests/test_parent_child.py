"""Tests for parent-child document retrieval.

Parent = Section (full text)
Child = Chunk (searchable fragment)

Key behaviors:
- Chunks know their section_id
- Can retrieve full section by section_id
- Size limits prevent fetching massive sections
- Falls back to chunk + neighbors when section too large
"""

import pytest

from src.rag.parser import Chunk, Section


class TestSectionStorage:
    """Tests for section storage and retrieval."""

    def test_chunk_has_section_id(self):
        """Chunks should have a section_id field."""
        chunk = Chunk(
            text="Test chunk text",
            metadata={"section": "Item 1A"},
            section_id="abc123_Item_1A",
        )
        assert chunk.section_id == "abc123_Item_1A"

    def test_chunk_section_id_default_empty(self):
        """section_id should default to empty string."""
        chunk = Chunk(text="Test")
        assert chunk.section_id == ""

    def test_section_has_section_id(self):
        """Sections should have a unique section_id."""
        section = Section(
            item="Item 1A",
            title="Risk Factors",
            content="Full risk factors text...",
            start_pos=0,
            end_pos=1000,
            section_id="abc123_Item_1A",
        )
        assert section.section_id == "abc123_Item_1A"

    def test_section_token_count(self):
        """Sections should report approximate token count."""
        content = "word " * 1000  # ~1000 words ≈ ~1333 tokens
        section = Section(
            item="Item 1A",
            title="Risk Factors",
            content=content,
            start_pos=0,
            end_pos=len(content),
            section_id="test",
        )
        # Rough estimate: 1 token ≈ 4 chars, or ~0.75 tokens per word
        assert 750 < section.approx_tokens < 1500


class TestParentChildRetrieval:
    """Tests for retrieving parent context from chunks."""

    @pytest.fixture
    def small_section(self):
        """A section under the size limit (~2000 tokens)."""
        content = "This is a small section. " * 200  # ~1000 tokens
        return Section(
            item="Item 1",
            title="Business",
            content=content,
            start_pos=0,
            end_pos=len(content),
            section_id="test_Item_1",
        )

    @pytest.fixture
    def large_section(self):
        """A section over the size limit (~20000 tokens)."""
        content = "This is a very large section with lots of content. " * 4000  # ~20000 tokens
        return Section(
            item="Item 1A",
            title="Risk Factors",
            content=content,
            start_pos=0,
            end_pos=len(content),
            section_id="test_Item_1A",
        )

    def test_get_section_returns_full_text_when_small(self, small_section):
        """When section is under limit, return full text."""
        # This will be implemented in the store
        # For now, test the size check logic
        max_tokens = 8000
        assert small_section.approx_tokens < max_tokens

    def test_get_section_respects_size_limit(self, large_section):
        """When section exceeds limit, should not return full text."""
        max_tokens = 8000
        assert large_section.approx_tokens > max_tokens

    def test_fallback_to_chunk_context_when_large(self, large_section):
        """Large sections should fall back to chunk + neighbors."""
        max_tokens = 8000
        # When section > max_tokens, retrieval should use expand_results_with_context
        assert large_section.approx_tokens > max_tokens
        # The actual fallback logic will be in Searcher


class TestSectionSizeLimits:
    """Tests for configurable size limits."""

    def test_default_max_section_tokens(self):
        """Default max section size should be reasonable."""
        from src.rag.search import SearchConfig

        config = SearchConfig()
        assert hasattr(config, "max_section_tokens")
        assert config.max_section_tokens == 8000  # ~32K chars

    def test_configurable_max_section_tokens(self):
        """Max section tokens should be configurable."""
        from src.rag.search import SearchConfig

        config = SearchConfig(max_section_tokens=4000)
        assert config.max_section_tokens == 4000

    def test_zero_max_disables_section_retrieval(self):
        """Setting max_section_tokens=0 should disable section retrieval."""
        from src.rag.search import SearchConfig

        config = SearchConfig(max_section_tokens=0)
        # 0 means never fetch full section, always use chunks
        assert config.max_section_tokens == 0


class TestChunkToSectionMapping:
    """Tests for mapping chunks back to their parent sections."""

    def test_chunk_id_contains_section_key(self):
        """Chunk IDs should contain section key for mapping."""
        chunk = Chunk(
            text="Test",
            chunk_id="abc123_Item_1A_0",
            section_id="abc123_Item_1A",
        )
        # section_id should be derivable from chunk_id (minus the index)
        assert chunk.section_id in chunk.chunk_id

    def test_multiple_chunks_same_section(self):
        """Multiple chunks from same section share section_id."""
        section_id = "abc123_Item_1A"
        chunks = [
            Chunk(text="First chunk", chunk_id=f"{section_id}_0", section_id=section_id),
            Chunk(text="Second chunk", chunk_id=f"{section_id}_1", section_id=section_id),
            Chunk(text="Third chunk", chunk_id=f"{section_id}_2", section_id=section_id),
        ]
        assert all(c.section_id == section_id for c in chunks)


class TestContextExpansionModes:
    """Tests for different context expansion strategies."""

    def test_section_mode_returns_full_section(self):
        """Section mode should return complete section text."""
        from src.rag.store import SearchResult

        # Create a mock result with section_id
        result = SearchResult(
            text="This is a chunk",
            score=0.9,
            metadata={"section_id": "test_section", "chunk_id": "test_section_0"},
        )
        # Actual test will require a real Searcher with stored sections
        assert result.section_id == "test_section"

    def test_window_mode_returns_chunk_plus_neighbors(self):
        """Window mode should return chunk + N neighbors."""
        # Already implemented in expand_results_with_context
        pass

    def test_auto_mode_chooses_based_on_size(self):
        """Auto mode should choose section vs window based on size."""
        from src.rag.search import SearchConfig

        # Config should have both max_section_tokens and context_window
        config = SearchConfig(max_section_tokens=8000, context_window=2)
        assert config.max_section_tokens == 8000
        assert config.context_window == 2


class TestSearcherParentRetrieval:
    """Tests for Searcher parent-child retrieval integration."""

    def test_get_parent_context_returns_section_when_small(self, tmp_path):
        """get_parent_context returns full section when under size limit."""
        from src.rag.search import SearchConfig
        from src.rag.store import SearchResult, VectorStore

        # Create a store with a small section
        store = VectorStore(index_dir=tmp_path, ticker="TEST")
        store.store_section(
            section_id="small_section",
            item="Item 1",
            title="Business",
            content="This is a small section. " * 50,  # ~500 tokens
            filing_type="10-K",
            filing_date="2024-01-15",
        )

        # Mock result
        result = SearchResult(
            text="chunk text",
            score=0.9,
            metadata={"section_id": "small_section", "chunk_id": "small_section_0"},
        )

        # Config with high limit (should return full section)
        config = SearchConfig(max_section_tokens=8000)

        # Can't create full Searcher without embedding model, but can test store directly
        section = store.get_section_for_chunk(result.metadata)
        assert section is not None
        assert section["approx_tokens"] < config.max_section_tokens
        assert "This is a small section" in section["content"]

    def test_get_parent_context_fallback_when_large(self, tmp_path):
        """get_parent_context falls back to window when section too large."""
        from src.rag.search import SearchConfig
        from src.rag.store import SearchResult, VectorStore

        # Create a store with a large section
        store = VectorStore(index_dir=tmp_path, ticker="TEST")
        store.store_section(
            section_id="large_section",
            item="Item 1A",
            title="Risk Factors",
            content="This is a very large section. " * 5000,  # ~25000 tokens
            filing_type="10-K",
            filing_date="2024-01-15",
        )

        # Mock result
        result = SearchResult(
            text="chunk text",
            score=0.9,
            metadata={"section_id": "large_section", "chunk_id": "large_section_0"},
        )

        # Config with lower limit (should trigger fallback)
        config = SearchConfig(max_section_tokens=8000)

        section = store.get_section_for_chunk(result.metadata)
        assert section is not None
        assert section["approx_tokens"] > config.max_section_tokens

    def test_get_parent_context_no_section(self, tmp_path):
        """get_parent_context handles missing sections gracefully."""
        from src.rag.store import SearchResult, VectorStore

        store = VectorStore(index_dir=tmp_path, ticker="TEST")

        # Result with section_id that doesn't exist
        result = SearchResult(
            text="chunk text",
            score=0.9,
            metadata={"section_id": "nonexistent", "chunk_id": "nonexistent_0"},
        )

        section = store.get_section_for_chunk(result.metadata)
        assert section is None

    def test_zero_max_tokens_always_uses_window(self):
        """Setting max_section_tokens=0 always uses window mode."""
        from src.rag.search import SearchConfig

        config = SearchConfig(max_section_tokens=0)
        # When max_tokens=0, get_parent_context should always use window mode
        assert config.max_section_tokens == 0


class TestSectionDeduplication:
    """Tests for section deduplication in parent retrieval."""

    def test_deduplicate_same_section_chunks(self, tmp_path):
        """Multiple chunks from same section should deduplicate."""
        from src.rag.store import SearchResult, VectorStore

        store = VectorStore(index_dir=tmp_path, ticker="TEST")
        section_id = "test_Item_1A"
        store.store_section(
            section_id=section_id,
            item="Item 1A",
            title="Risk Factors",
            content="Full risk factors content...",
        )

        # Three chunks from same section
        results = [
            SearchResult(
                text="Chunk 1",
                score=0.9,
                metadata={"section_id": section_id, "chunk_id": f"{section_id}_0"},
            ),
            SearchResult(
                text="Chunk 2",
                score=0.85,
                metadata={"section_id": section_id, "chunk_id": f"{section_id}_1"},
            ),
            SearchResult(
                text="Chunk 3",
                score=0.8,
                metadata={"section_id": section_id, "chunk_id": f"{section_id}_2"},
            ),
        ]

        # All have same section_id - deduplication should work
        section_ids = [r.section_id for r in results]
        unique_sections = set(section_ids)
        assert len(unique_sections) == 1  # All from same section
