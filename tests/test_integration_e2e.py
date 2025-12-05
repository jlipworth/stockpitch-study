"""End-to-end integration tests using real data.

These tests verify the complete pipeline works with real SEC filings,
real embeddings, and real search. They are slower but provide confidence
that the system works end-to-end.

Run with: pytest tests/test_integration_e2e.py -v
Skip with: pytest -m "not integration"
"""

import tempfile
from pathlib import Path

import pytest

from src.filings import SECFetcher
from src.rag import EmbeddingModel, Searcher, SECFilingParser, VectorStore


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end tests that fetch, parse, embed, index, and search real data."""

    @pytest.fixture(scope="class")
    def temp_dirs(self):
        """Create temporary directories for the test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            index_dir = root / "index"
            data_dir.mkdir()
            index_dir.mkdir()
            yield {"root": root, "data": data_dir, "index": index_dir}

    @pytest.fixture(scope="class")
    def fetched_filing(self, temp_dirs):
        """Fetch a single real 10-K from SEC EDGAR."""
        fetcher = SECFetcher(
            data_dir=temp_dirs["data"],
            user_agent="StockPitchTest test@example.com",
        )

        # Fetch just 1 10-K for AAPL (most recent)
        filings = fetcher.fetch("AAPL", filing_types=["10-K"], years=1)

        assert len(filings) >= 1, "Should fetch at least one 10-K"
        filing = filings[0]

        # Verify file exists
        file_path = temp_dirs["data"] / filing.file_path
        assert file_path.exists(), f"Filing file should exist: {file_path}"

        return {"metadata": filing, "path": file_path, "dirs": temp_dirs}

    @pytest.fixture(scope="class")
    def parsed_chunks(self, fetched_filing):
        """Parse the fetched filing into chunks."""
        parser = SECFilingParser()
        metadata = fetched_filing["metadata"]

        # Parse the filing
        parsed = parser.parse_file(
            fetched_filing["path"],
            ticker="AAPL",
            filing_type="10-K",
            filing_date=metadata.filing_date,
        )
        assert len(parsed.sections) > 0, "Should parse at least one section"

        # Chunk the sections
        filing_metadata = {
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": metadata.filing_date,
        }
        all_chunks = []
        for section in parsed.sections:
            chunks = parser.chunk_section(section, filing_metadata=filing_metadata)
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0, "Should create at least one chunk"
        return {
            "chunks": all_chunks,
            "sections": parsed.sections,
            **fetched_filing,
        }

    @pytest.fixture(scope="class")
    def embedding_model(self):
        """Load the real embedding model."""
        model = EmbeddingModel()
        # Trigger model load
        _ = model.embedding_dim
        assert model._model is not None, "Model should be loaded"
        return model

    @pytest.fixture(scope="class")
    def indexed_store(self, parsed_chunks, embedding_model):
        """Create embeddings and store in vector DB."""
        chunks = parsed_chunks["chunks"]
        dirs = parsed_chunks["dirs"]

        # Create embeddings for all chunks
        texts = [c.text for c in chunks]
        embeddings = embedding_model.encode(texts)

        assert embeddings.shape[0] == len(chunks), "Should have one embedding per chunk"
        assert embeddings.shape[1] == 1024, "BGE-M3 should produce 1024-dim embeddings"

        # Create vector store
        store = VectorStore(
            index_dir=dirs["index"],
            ticker="AAPL",
            embedding_dim=1024,
        )

        # Create EmbeddedChunks and add to store
        from src.rag import EmbeddedChunk

        embedded_chunks = [
            EmbeddedChunk(
                text=chunk.text,
                embedding=embeddings[i],
                metadata={
                    "ticker": "AAPL",
                    "filing_type": "10-K",
                    "filing_date": parsed_chunks["metadata"].filing_date,
                    "section": chunk.metadata.get("section", ""),
                    "section_title": chunk.metadata.get("section_title", ""),
                    "source": str(parsed_chunks["path"].name),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        store.add(
            embedded_chunks,
            source_file=str(parsed_chunks["path"].name),
            file_hash="test_integration",
        )

        return {
            "store": store,
            "chunk_count": len(chunks),
            **parsed_chunks,
        }

    @pytest.fixture(scope="class")
    def searcher(self, indexed_store, embedding_model):
        """Create a searcher for the indexed store."""
        searcher = Searcher(
            index_dir=indexed_store["dirs"]["index"],
            ticker="AAPL",
            embedding_model=embedding_model,
        )
        return {"searcher": searcher, **indexed_store}

    # =========================================================================
    # Actual Tests
    # =========================================================================

    def test_fetch_downloads_real_filing(self, fetched_filing):
        """Verify we fetched a real 10-K from SEC."""
        metadata = fetched_filing["metadata"]

        assert metadata.ticker == "AAPL"
        assert metadata.filing_type == "10-K"
        assert metadata.filing_date is not None

        # Check file has substantial content (real 10-K is large)
        file_size = fetched_filing["path"].stat().st_size
        assert file_size > 100_000, f"10-K should be >100KB, got {file_size}"

    def test_parser_extracts_sections(self, parsed_chunks):
        """Verify parser extracts expected sections from 10-K."""
        sections = parsed_chunks["sections"]
        section_items = [s.item for s in sections]

        # A real 10-K should have these sections
        expected_sections = ["Item 1", "Item 1A", "Item 7"]
        for expected in expected_sections:
            assert any(expected in item for item in section_items), f"Should have {expected} section"

    def test_chunking_produces_reasonable_chunks(self, parsed_chunks):
        """Verify chunking produces reasonable-sized chunks."""
        chunks = parsed_chunks["chunks"]

        # Should have multiple chunks
        assert len(chunks) > 10, "A 10-K should produce many chunks"

        # Chunks should have reasonable text length
        for chunk in chunks[:10]:  # Check first 10
            assert len(chunk.text) > 100, "Chunks should have substantial text"
            # Some chunks (e.g., tables, special sections) can be larger
            assert len(chunk.text) < 15000, "Chunks should not be excessively large"

    def test_embeddings_are_valid(self, indexed_store):
        """Verify embeddings were created correctly."""
        chunk_count = indexed_store["chunk_count"]

        # Verify chunks were indexed
        assert chunk_count > 0, "Should have indexed chunks"
        assert chunk_count > 10, "A 10-K should produce many chunks"

    def test_fts_search_returns_results(self, searcher):
        """Verify full-text search returns relevant results."""
        results = searcher["searcher"].search(
            query="iPhone revenue",
            mode="fts",
            top_k=5,
        )

        assert len(results) > 0, "FTS search should return results"

        # Results should mention iPhone or revenue
        found_relevant = False
        for r in results:
            text_lower = r.text.lower()
            if "iphone" in text_lower or "revenue" in text_lower:
                found_relevant = True
                break

        assert found_relevant, "Should find results mentioning iPhone or revenue"

    def test_vector_search_returns_results(self, searcher):
        """Verify vector/semantic search returns relevant results."""
        results = searcher["searcher"].search(
            query="What are the main risk factors?",
            mode="vector",
            top_k=5,
        )

        assert len(results) > 0, "Vector search should return results"

    def test_hybrid_search_returns_results(self, searcher):
        """Verify hybrid search combines FTS and vector."""
        results = searcher["searcher"].search(
            query="services gross margin",
            mode="hybrid",
            top_k=5,
        )

        assert len(results) > 0, "Hybrid search should return results"

    def test_search_with_section_filter(self, searcher):
        """Verify search can filter by section."""
        results = searcher["searcher"].search(
            query="competition",
            mode="fts",
            top_k=10,
            section="Item 1A",
        )

        # All results should be from Item 1A (Risk Factors)
        for r in results:
            assert "1A" in r.metadata.get("section", ""), "Results should be from Item 1A"

    def test_search_scores_are_reasonable(self, searcher):
        """Verify search scores are in expected range."""
        results = searcher["searcher"].search(
            query="Apple Inc annual report",
            mode="hybrid",
            top_k=5,
        )

        for r in results:
            # Scores should be between 0 and 1 for normalized scores
            # or positive for raw scores
            assert r.score >= 0, "Scores should be non-negative"


@pytest.mark.integration
class TestMinimalPipeline:
    """Minimal integration test - faster, tests core flow only."""

    def test_fetch_parse_embed_search(self):
        """Single test that verifies the complete pipeline works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            index_dir = root / "index"
            data_dir.mkdir()
            index_dir.mkdir()

            # 1. Fetch one 10-K
            fetcher = SECFetcher(
                data_dir=data_dir,
                user_agent="StockPitchTest test@example.com",
            )
            filings = fetcher.fetch("AAPL", filing_types=["10-K"], years=1)
            assert len(filings) >= 1, "Step 1 failed: No filings fetched"

            filing = filings[0]
            file_path = data_dir / filing.file_path
            assert file_path.exists(), "Step 1 failed: File not created"

            # 2. Parse and chunk
            parser = SECFilingParser()
            parsed = parser.parse_file(
                file_path,
                ticker="AAPL",
                filing_type="10-K",
                filing_date=filing.filing_date,
            )
            assert len(parsed.sections) > 0, "Step 2 failed: No sections parsed"

            filing_metadata = {
                "ticker": "AAPL",
                "filing_type": "10-K",
                "filing_date": filing.filing_date,
            }
            chunks = []
            for section in parsed.sections[:3]:  # Only first 3 sections for speed
                chunks.extend(parser.chunk_section(section, filing_metadata=filing_metadata))
            assert len(chunks) > 0, "Step 2 failed: No chunks created"

            # 3. Create embeddings
            model = EmbeddingModel()
            texts = [c.text for c in chunks]
            embeddings = model.encode(texts)
            assert embeddings.shape[0] == len(chunks), "Step 3 failed: Embedding count mismatch"

            # 4. Store in vector DB
            store = VectorStore(
                index_dir=index_dir,
                ticker="AAPL",
                embedding_dim=1024,
            )

            from src.rag import EmbeddedChunk

            embedded_chunks = [
                EmbeddedChunk(
                    text=chunk.text,
                    embedding=embeddings[i],
                    metadata={
                        "ticker": "AAPL",
                        "filing_type": "10-K",
                        "filing_date": filing.filing_date,
                        "section": chunk.metadata.get("section", ""),
                        "section_title": chunk.metadata.get("section_title", ""),
                        "source": file_path.name,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]
            store.add(embedded_chunks, source_file=file_path.name, file_hash="test")

            # 5. Search
            searcher = Searcher(
                index_dir=index_dir,
                ticker="AAPL",
                embedding_model=model,
            )

            # FTS search
            fts_results = searcher.search("Apple", mode="fts", top_k=3)
            assert len(fts_results) > 0, "Step 5 failed: FTS search returned no results"

            # Vector search
            vector_results = searcher.search("business overview", mode="vector", top_k=3)
            assert len(vector_results) > 0, "Step 5 failed: Vector search returned no results"

            # Hybrid search
            hybrid_results = searcher.search("iPhone sales", mode="hybrid", top_k=3)
            assert len(hybrid_results) > 0, "Step 5 failed: Hybrid search returned no results"
