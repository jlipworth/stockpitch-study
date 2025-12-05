# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-12-26

Initial public release of the Stock Pitch Case Template.

### Added

#### Core Features

- **SEC Filing Fetcher** - Download 10-K, 10-Q, 8-K, DEF 14A, and Form 4 filings from SEC EDGAR
- **Form 4 Parser** - Structured parsing of insider trading XML (transactions, holdings, insider info)
- **Document Parser** - Section-aware chunking with fiscal period extraction (Q1-Q4/FY)
- **Vector Search** - Hybrid search (semantic + BM25) powered by LanceDB and BGE-M3 embeddings
- **RAG Q&A** - Question answering with Claude-powered responses and source citations
- **Document Summarizer** - Hierarchical summarization with section-specific guidance
- **Batch Processing** - Async batch API for 50% cost savings on large jobs
- **Handwriting Converter** - Convert handwritten PDF notes to markdown via Claude vision
- **Materials Inventory** - Generate quick reference of all project materials

#### CLI Commands

- `pitch process` - Full pipeline: fetch → index → summarize
- `pitch fetch` - Download SEC filings with date range and type filters
- `pitch index` - Build vector index for multiple source types
- `pitch search` - Hybrid, vector, or FTS search with reranking
- `pitch ask` - RAG-powered question answering
- `pitch summarize` - Generate document summaries
- `pitch batch-submit/status/results/resubmit` - Batch processing workflow
- `pitch notes` - Convert handwritten PDFs to markdown
- `pitch wrap` - Format markdown for narrow displays
- `pitch inventory` - Generate materials inventory

#### Infrastructure

- Pre-commit hooks (ruff, mypy, pytest)
- GitHub Actions CI with Docker caching
- Comprehensive test coverage
- BaseParser ABC for consistent parser interfaces
- Configurable section weights via YAML

### Architecture

- **Embeddings**: BGE-M3 (1024 dimensions, 8192 token context)
- **Vector Store**: LanceDB with hybrid search support
- **LLM**: Claude Sonnet 4.5 via Anthropic API
- **PDF Processing**: PyMuPDF for page extraction
- **SEC Access**: edgartools for EDGAR API

### Documentation

- Comprehensive README with CLI examples
- CLAUDE.md for AI assistant context
- Architecture documentation
- RAG improvements roadmap
- Research questions workflow guide

______________________________________________________________________

## Version History Notes

### Versioning Strategy

- **Major (X.0.0)**: Breaking changes to CLI or API
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, documentation updates

### Pre-release Development

The following improvements were made during pre-release development:

#### Code Quality (H-series)

- H1: Extracted SECTION_WEIGHTS to YAML configuration
- H2: Added atomic file writes for crash safety
- H3: Created BaseParser abstract base class
- H4: Extracted table utilities to dedicated module
- H5: Decoupled Searcher from QueryClassifier
- H6: Optimized O(n²) → O(n) section matching
- H7: Refactored retrieve_results() with helper methods
- H8: Added comprehensive query_classifier tests
- H9: Added comprehensive feedback module tests
- H10: Added pytest to pre-commit hooks

#### Medium Priority (M-series)

- M1: Extracted magic numbers to constants module
- M3: Refactored large functions (chunk_section, ask)
- M4: Improved exception handling with specific types
- M7-M9: Added logging to Form4Parser and NotesConverter
- M10-M11: Parametrized test fixtures and consolidated duplicates
- M13: Split cli.py into submodules
- M14: Extracted section patterns to dedicated module

#### Infrastructure

- CI: CPU-only PyTorch for smaller Docker images
- Renovate configuration for dependency updates
