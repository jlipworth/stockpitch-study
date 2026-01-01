"""LanceDB vector store with hybrid search support."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import numpy as np

from src.utils.file_locks import file_lock

from .embeddings import EmbeddedChunk

logger = logging.getLogger(__name__)

# Allowlist of columns that can be filtered (prevents SQL injection via key names)
FILTERABLE_COLUMNS = frozenset(
    {
        "ticker",
        "filing_type",
        "filing_date",
        "section",
        "section_title",
        "source",
        "chunk_index",
        # Phase 1.1: Chunk position and linking fields
        "chunk_id",
        "prev_chunk_id",
        "next_chunk_id",
        "position_in_section",
        "is_section_start",
        "is_section_end",
        # Parent-child retrieval
        "section_id",
        # Phase 2: Table Intelligence
        "content_type",
        "has_table",
        "table_type",
        "is_table_continuation",
        # Fiscal period filtering (extracted from iXBRL metadata)
        "fiscal_year",
        "fiscal_quarter",
        # Source type filtering (transcript, conference, analyst, etc.)
        "source_type",
    }
)


def _escape_sql_string(value: str) -> str:
    """Escape single quotes for SQL string literals."""
    return value.replace("'", "''")


def _build_where_clause(filter_dict: dict[str, Any]) -> str | None:
    """
    Build a safe SQL WHERE clause from a filter dictionary.

    Args:
        filter_dict: Dictionary of column -> value filters

    Returns:
        SQL WHERE clause string, or None if empty

    Raises:
        ValueError: If an invalid column name is provided
    """
    if not filter_dict:
        return None

    where_clauses = []
    for key, value in filter_dict.items():
        # Validate column name against allowlist
        if key not in FILTERABLE_COLUMNS:
            raise ValueError(
                f"Invalid filter column '{key}'. " f"Allowed columns: {', '.join(sorted(FILTERABLE_COLUMNS))}"
            )

        if isinstance(value, list):
            if not value:
                # Empty list - skip this filter
                continue
            # Handle list values with IN clause
            if all(isinstance(v, str) for v in value):
                escaped_values = [_escape_sql_string(v) for v in value]
                values_str = ", ".join(f"'{v}'" for v in escaped_values)
                where_clauses.append(f"{key} IN ({values_str})")
            else:
                values_str = ", ".join(str(v) for v in value)
                where_clauses.append(f"{key} IN ({values_str})")
        elif isinstance(value, str):
            escaped_value = _escape_sql_string(value)
            where_clauses.append(f"{key} = '{escaped_value}'")
        else:
            where_clauses.append(f"{key} = {value}")

    if not where_clauses:
        return None

    return " AND ".join(where_clauses)


@dataclass
class IndexManifest:
    """Tracks indexed files for incremental updates."""

    ticker: str
    indexed_files: dict[str, str] = field(default_factory=dict)  # file_path -> file_hash
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    chunk_count: int = 0
    embedding_model: str = ""
    embedding_dim: int = 0

    def save(self, path: Path) -> None:
        """Save manifest to JSON file with locking."""
        with file_lock(path):
            data = {
                "ticker": self.ticker,
                "indexed_files": self.indexed_files,
                "last_updated": datetime.now().isoformat(),
                "chunk_count": self.chunk_count,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
            }
            path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        """Load manifest from JSON file."""
        data = json.loads(path.read_text())
        return cls(
            ticker=data["ticker"],
            indexed_files=data.get("indexed_files", {}),
            last_updated=data.get("last_updated", ""),
            chunk_count=data.get("chunk_count", 0),
            embedding_model=data.get("embedding_model", ""),
            embedding_dim=data.get("embedding_dim", 0),
        )


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def ticker(self) -> str:
        return self.metadata.get("ticker", "")

    @property
    def filing_type(self) -> str:
        return self.metadata.get("filing_type", "")

    @property
    def filing_date(self) -> str:
        return self.metadata.get("filing_date", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    # Phase 1.1: Chunk position and linking properties
    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "")

    @property
    def prev_chunk_id(self) -> str | None:
        return self.metadata.get("prev_chunk_id")

    @property
    def next_chunk_id(self) -> str | None:
        return self.metadata.get("next_chunk_id")

    @property
    def position_in_section(self) -> int:
        return self.metadata.get("position_in_section", 0)

    @property
    def total_chunks_in_section(self) -> int:
        return self.metadata.get("total_chunks_in_section", 0)

    @property
    def is_section_start(self) -> bool:
        return self.metadata.get("is_section_start", False)

    @property
    def is_section_end(self) -> bool:
        return self.metadata.get("is_section_end", False)

    # Parent-child retrieval
    @property
    def section_id(self) -> str:
        return self.metadata.get("section_id", "")

    # Phase 2: Table Intelligence
    @property
    def content_type(self) -> str:
        return self.metadata.get("content_type", "text")

    @property
    def has_table(self) -> bool:
        return self.metadata.get("has_table", False)

    @property
    def table_html(self) -> str:
        return self.metadata.get("table_html", "")

    @property
    def table_caption(self) -> str:
        return self.metadata.get("table_caption", "")

    @property
    def table_type(self) -> str:
        return self.metadata.get("table_type", "")

    @property
    def is_table_continuation(self) -> bool:
        return self.metadata.get("is_table_continuation", False)


class VectorStore:
    """LanceDB-based vector store with per-company isolation."""

    TABLE_NAME = "chunks"

    @staticmethod
    def _row_to_metadata(r: dict) -> dict[str, Any]:
        """Extract metadata dict from a LanceDB row.

        Centralizes metadata extraction to ensure all SearchResults include
        the same fields (including Phase 1.1 chunk linking fields).
        """
        return {
            "ticker": r.get("ticker", ""),
            "filing_type": r.get("filing_type", ""),
            "filing_date": r.get("filing_date", ""),
            "section": r.get("section", ""),
            "section_title": r.get("section_title", ""),
            "source": r.get("source", ""),
            "chunk_index": r.get("chunk_index", 0),
            # Phase 1.1: Chunk position and linking fields
            "chunk_id": r.get("chunk_id", ""),
            "prev_chunk_id": r.get("prev_chunk_id"),
            "next_chunk_id": r.get("next_chunk_id"),
            "position_in_section": r.get("position_in_section", 0),
            "total_chunks_in_section": r.get("total_chunks_in_section", 0),
            "is_section_start": r.get("is_section_start", False),
            "is_section_end": r.get("is_section_end", False),
            # Parent-child retrieval
            "section_id": r.get("section_id", ""),
            # Phase 2: Table Intelligence
            "content_type": r.get("content_type", "text"),
            "has_table": r.get("has_table", False),
            "table_html": r.get("table_html", ""),
            "table_caption": r.get("table_caption", ""),
            "table_type": r.get("table_type", ""),
            "is_table_continuation": r.get("is_table_continuation", False),
        }

    def __init__(
        self,
        index_dir: Path,
        ticker: str,
        embedding_dim: int = 1024,
        embedding_model: str = "",
    ):
        """
        Initialize vector store for a company.

        Args:
            index_dir: Base directory for indexes (e.g., ./index)
            ticker: Stock ticker (used for isolation)
            embedding_dim: Dimension of embeddings
            embedding_model: Name of embedding model (for tracking)
        """
        self.index_dir = index_dir
        self.ticker = ticker.upper()
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        # Per-company directory
        self.company_dir = index_dir / self.ticker
        self.company_dir.mkdir(parents=True, exist_ok=True)

        # LanceDB database path
        self.db_path = self.company_dir / "lancedb"

        # Manifest for tracking indexed files
        self.manifest_path = self.company_dir / "index_manifest.json"
        self._manifest: IndexManifest | None = None

        # Parent-child: Section storage for full section retrieval
        self.sections_path = self.company_dir / "sections.json"
        self._sections: dict[str, dict] | None = None

        # LanceDB connection (lazy loaded)
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    @property
    def db(self) -> lancedb.DBConnection:
        """Lazy load database connection."""
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    @property
    def manifest(self) -> IndexManifest:
        """Load or create manifest."""
        if self._manifest is None:
            if self.manifest_path.exists():
                self._manifest = IndexManifest.load(self.manifest_path)
            else:
                self._manifest = IndexManifest(
                    ticker=self.ticker,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dim,
                )
        return self._manifest

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        self.manifest.save(self.manifest_path)

    # =========================================================================
    # Parent-child retrieval: Section storage
    # =========================================================================

    @property
    def sections(self) -> dict[str, dict]:
        """Load or create sections store.

        Returns dict mapping section_id -> section data:
        {
            "section_id": {
                "item": "Item 1A",
                "title": "Risk Factors",
                "content": "Full section text...",
                "approx_tokens": 1500,
                "filing_type": "10-K",
                "filing_date": "2024-01-15",
                "source": "abc123..."
            }
        }
        """
        if self._sections is None:
            if self.sections_path.exists():
                with open(self.sections_path, encoding="utf-8") as f:
                    self._sections = json.load(f)
            else:
                self._sections = {}
        # Assert to satisfy type checker - we know it's not None after above logic
        assert self._sections is not None
        return self._sections

    def _save_sections(self) -> None:
        """Save sections to disk."""
        with file_lock(self.sections_path):
            with open(self.sections_path, "w", encoding="utf-8") as f:
                json.dump(self.sections, f, indent=2)

    def store_section(
        self,
        section_id: str,
        item: str,
        title: str,
        content: str,
        filing_type: str = "",
        filing_date: str = "",
        source: str = "",
    ) -> None:
        """Store a section for parent-child retrieval.

        Args:
            section_id: Unique section identifier
            item: Section item (e.g., "Item 1A")
            title: Section title (e.g., "Risk Factors")
            content: Full section text
            filing_type: Filing type (e.g., "10-K")
            filing_date: Filing date
            source: Source file hash
        """
        approx_tokens = int(len(content.split()) * 1.3)
        self.sections[section_id] = {
            "item": item,
            "title": title,
            "content": content,
            "approx_tokens": approx_tokens,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "source": source,
        }
        self._save_sections()

    def get_section(self, section_id: str) -> dict | None:
        """Retrieve a section by ID.

        Args:
            section_id: Section identifier

        Returns:
            Section data dict or None if not found
        """
        return self.sections.get(section_id)

    def get_section_for_chunk(self, chunk_metadata: dict) -> dict | None:
        """Get parent section for a chunk.

        Args:
            chunk_metadata: Chunk metadata containing section_id

        Returns:
            Section data dict or None if not found
        """
        section_id = chunk_metadata.get("section_id")
        if not section_id:
            return None
        return self.get_section(section_id)

    def clear_sections(self) -> None:
        """Clear all stored sections."""
        self._sections = {}
        if self.sections_path.exists():
            self.sections_path.unlink()

    def _get_table(self) -> lancedb.table.Table | None:
        """Get or return None if table doesn't exist."""
        if self._table is not None:
            return self._table

        if self.TABLE_NAME in self.db.table_names():
            self._table = self.db.open_table(self.TABLE_NAME)
            return self._table

        return None

    def _create_table(self, chunks: list[EmbeddedChunk]) -> lancedb.table.Table:
        """Create table with initial data."""
        data = self._chunks_to_records(chunks)
        self._table = self.db.create_table(
            self.TABLE_NAME,
            data=data,
            mode="overwrite",
        )
        # Create FTS index for hybrid search
        # Use use_tantivy=False for proper filtering support (see lancedb/lancedb#1636)
        assert self._table is not None
        self._table.create_fts_index("text", replace=True, use_tantivy=False)
        return self._table

    def _chunks_to_records(self, chunks: list[EmbeddedChunk]) -> list[dict]:
        """Convert EmbeddedChunk objects to LanceDB records."""
        records = []
        for chunk in chunks:
            record = {
                "text": chunk.text,
                "vector": chunk.embedding.tolist(),
                # Flatten common metadata fields for filtering
                "ticker": chunk.metadata.get("ticker", self.ticker),
                "filing_type": chunk.metadata.get("filing_type", chunk.metadata.get("doc_type", "")),
                "filing_date": chunk.metadata.get("filing_date", ""),
                "section": chunk.metadata.get("section", ""),
                "section_title": chunk.metadata.get("section_title", ""),
                "source": chunk.metadata.get("source", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                # Phase 1.1: Chunk position and linking fields
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "prev_chunk_id": chunk.metadata.get("prev_chunk_id"),  # Can be None
                "next_chunk_id": chunk.metadata.get("next_chunk_id"),  # Can be None
                "position_in_section": chunk.metadata.get("position_in_section", 0),
                "total_chunks_in_section": chunk.metadata.get("total_chunks_in_section", 0),
                "is_section_start": chunk.metadata.get("is_section_start", False),
                "is_section_end": chunk.metadata.get("is_section_end", False),
                # Parent-child retrieval
                "section_id": chunk.metadata.get("section_id", ""),
                # Phase 2: Table Intelligence
                "content_type": chunk.metadata.get("content_type", "text"),
                "has_table": chunk.metadata.get("has_table", False),
                "table_html": chunk.metadata.get("table_html", ""),
                "table_caption": chunk.metadata.get("table_caption", ""),
                "table_type": chunk.metadata.get("table_type", ""),
                "is_table_continuation": chunk.metadata.get("is_table_continuation", False),
                # Fiscal period filtering (extracted from iXBRL metadata)
                # Use 0 as "unknown" to maintain consistent int64 schema
                "fiscal_year": chunk.metadata.get("fiscal_year") or 0,
                "fiscal_quarter": chunk.metadata.get("fiscal_quarter") or 0,
                # Store full metadata as JSON for flexibility
                "metadata_json": json.dumps(chunk.metadata),
            }
            records.append(record)
        return records

    def add(self, chunks: list[EmbeddedChunk], source_file: str = "", file_hash: str = "") -> int:
        """
        Add embedded chunks to the store.

        Args:
            chunks: List of EmbeddedChunk objects
            source_file: Path to source file (for incremental tracking)
            file_hash: Hash of source file (for change detection)

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        logger.debug(f"Adding {len(chunks)} chunks to {self.ticker} index (source: {source_file or 'unknown'})")

        table = self._get_table()
        records = self._chunks_to_records(chunks)

        if table is None:
            # Create new table
            logger.info(f"Creating new index table for {self.ticker}")
            self._create_table(chunks)
        else:
            # Add to existing table
            table.add(records)
            # Rebuild FTS index after adding
            logger.debug("Rebuilding FTS index")
            table.create_fts_index("text", replace=True)

        # Update manifest
        if source_file:
            self.manifest.indexed_files[source_file] = file_hash
        self.manifest.chunk_count += len(chunks)
        self._save_manifest()

        logger.info(f"✓ Added {len(chunks)} chunks to {self.ticker} index (total: {self.manifest.chunk_count})")

        return len(chunks)

    def is_file_indexed(self, file_path: str, file_hash: str) -> bool:
        """
        Check if a file has already been indexed with the same hash.

        Args:
            file_path: Path to the file
            file_hash: Current hash of the file

        Returns:
            True if file is indexed with same hash, False otherwise
        """
        return self.manifest.indexed_files.get(file_path) == file_hash

    def search_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filters (e.g., {"filing_type": "10-K"})

        Returns:
            List of SearchResult objects
        """
        table = self._get_table()
        if table is None:
            logger.warning(f"No index table found for {self.ticker}")
            return []

        logger.debug(f"Vector search: top_k={top_k}, filters={filter_dict}")

        query = table.search(query_embedding.tolist()).limit(top_k)

        # Apply filters using helper (handles escaping and validation)
        where_clause = _build_where_clause(filter_dict) if filter_dict else None
        if where_clause:
            query = query.where(where_clause)

        results = query.to_list()
        logger.debug(f"Vector search returned {len(results)} results")

        return [
            SearchResult(
                text=r["text"],
                score=1 - r.get("_distance", 0),  # Convert distance to similarity
                metadata=self._row_to_metadata(r),
            )
            for r in results
        ]

    def search_fts(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search using full-text search (BM25).

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional filters

        Returns:
            List of SearchResult objects
        """
        table = self._get_table()
        if table is None:
            return []

        fts_query = table.search(query, query_type="fts").limit(top_k)

        # Apply post-filters using helper (prefilter=False required for FTS filtering)
        where_clause = _build_where_clause(filter_dict) if filter_dict else None
        if where_clause:
            fts_query = fts_query.where(where_clause, prefilter=False)

        results = fts_query.to_list()

        return [
            SearchResult(
                text=r["text"],
                score=r.get("_score", 0),  # FTS returns a score
                metadata=self._row_to_metadata(r),
            )
            for r in results
        ]

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search combining vector and FTS.

        Args:
            query_embedding: Query embedding vector
            query_text: Query text for FTS
            top_k: Number of results to return
            vector_weight: Weight for vector results (0-1), FTS gets 1-weight
            filter_dict: Optional filters

        Returns:
            List of SearchResult objects, reranked by combined score
        """
        table = self._get_table()
        if table is None:
            return []

        # Use LanceDB's hybrid search with explicit vector and text
        query = table.search(query_type="hybrid").vector(query_embedding.tolist()).text(query_text).limit(top_k)

        # Apply post-filters using helper (prefilter=False required for hybrid filtering)
        where_clause = _build_where_clause(filter_dict) if filter_dict else None
        if where_clause:
            query = query.where(where_clause, prefilter=False)

        results = query.to_list()

        return [
            SearchResult(
                text=r["text"],
                score=r.get("_relevance_score", 0),
                metadata=self._row_to_metadata(r),
            )
            for r in results
        ]

    def count(self) -> int:
        """Return total number of chunks in the store."""
        table = self._get_table()
        if table is None:
            return 0
        return table.count_rows()

    def get_chunk_by_id(self, chunk_id: str) -> SearchResult | None:
        """
        Retrieve a specific chunk by its ID.

        Phase 1.1: Enables context expansion by fetching adjacent chunks.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            SearchResult or None if not found
        """
        table = self._get_table()
        if table is None:
            return None

        escaped_id = _escape_sql_string(chunk_id)
        try:
            results = table.search().where(f"chunk_id = '{escaped_id}'").limit(1).to_list()
        except Exception as e:
            logger.warning(f"Error fetching chunk by ID '{chunk_id}': {e}")
            return None

        if not results:
            return None

        r = results[0]
        return SearchResult(
            text=r["text"],
            score=1.0,  # Direct lookup, perfect match
            metadata=self._row_to_metadata(r),
        )

    def clear(self) -> None:
        """Clear all data from the store.

        Removes both the LanceDB table and physical directory to handle
        corrupt states from interrupted operations.
        """
        import shutil

        # Clear sections storage (parent-child)
        self.clear_sections()

        # Try to drop table gracefully
        try:
            if self.TABLE_NAME in self.db.table_names():
                self.db.drop_table(self.TABLE_NAME)
        except Exception as e:
            logger.debug("Table drop failed (will clean up with physical delete): %s", e)

        # Close connection and remove physical directory for clean slate
        self._db = None
        self._table = None

        if self.db_path.exists():
            shutil.rmtree(self.db_path)

        self._manifest = IndexManifest(
            ticker=self.ticker,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
        )
        self._save_manifest()

    def delete_by_source(self, source_file: str) -> int:
        """
        Delete all chunks from a specific source file.

        Args:
            source_file: Path to source file

        Returns:
            Number of chunks deleted
        """
        table = self._get_table()
        if table is None:
            return 0

        # Count before delete
        before_count = table.count_rows()

        # Delete matching rows (escape SQL string to prevent injection)
        escaped_source = _escape_sql_string(source_file)
        table.delete(f"source = '{escaped_source}'")

        # Count after delete
        after_count = table.count_rows()
        deleted = before_count - after_count

        # Update manifest
        if source_file in self.manifest.indexed_files:
            del self.manifest.indexed_files[source_file]
            self.manifest.chunk_count -= deleted
            self._save_manifest()

        return deleted
