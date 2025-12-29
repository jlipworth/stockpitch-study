# Contributing to Stock Pitch Case Template

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (for dependency management)
- Git

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/jlipworth/stock-pitch-template.git
cd stock-pitch-template

# Install dependencies (creates .venv automatically)
uv sync

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# Copy environment template
cp .env.template .env
# Add your API keys to .env
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_parser.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests (requires network)
pytest -m integration
```

## Code Style

This project uses automated formatting and linting via pre-commit hooks:

| Tool            | Purpose                            |
| --------------- | ---------------------------------- |
| **ruff**        | Linting and auto-fixes             |
| **ruff-format** | Code formatting (Black-compatible) |
| **mypy**        | Static type checking               |
| **mdformat**    | Markdown formatting                |

### Running Pre-commit Manually

```bash
# Run all hooks on all files
pre-commit run --all-files

# Skip specific hooks if needed
SKIP=mypy,pytest git commit -m "message"
```

### Code Guidelines

- **Type hints**: All public functions should have type annotations
- **Docstrings**: Use Google-style docstrings for public functions
- **Imports**: Let ruff organize imports automatically
- **Line length**: 120 characters max (configured in pyproject.toml)

Example docstring:

```python
def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
    """Search the vector index for relevant documents.

    Args:
        query: The search query string.
        top_k: Maximum number of results to return.

    Returns:
        List of SearchResult objects sorted by relevance.

    Raises:
        ValueError: If the index is empty.
    """
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Keep commits focused and atomic
- Write descriptive commit messages
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Message Format

```
<type>: <short description>

<optional body with more details>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, CI, etc.)

Examples:

```
feat: Add support for 8-K filing parsing
fix: Handle empty sections in transcript parser
docs: Add architecture diagrams to ARCHITECTURE.md
refactor: Extract table utilities to dedicated module
```

### 4. Run Checks

Before submitting:

```bash
# Format and lint
pre-commit run --all-files

# Run tests
pytest -v

# Type check
mypy src/
```

### 5. Submit Pull Request

- Fill out the PR template
- Link any related issues
- Request review from maintainers

## What Gets Accepted

### Good Contributions

- Bug fixes with test coverage
- New features aligned with project goals
- Documentation improvements
- Performance optimizations with benchmarks
- Test coverage improvements

### Template vs Fork Changes

This is a **template repository**. Keep changes generic:

| Accept                               | Reject                             |
| ------------------------------------ | ---------------------------------- |
| Generic features useful to all users | Company-specific customizations    |
| Industry-agnostic improvements       | Hardcoded tickers or company names |
| Reusable utilities                   | One-off scripts                    |

Company-specific code belongs in your **fork**, not the template.

## Project Structure

```
src/
├── cli/              # CLI commands (split into submodules)
├── filings/          # SEC fetcher and Form 4 parser
├── notes/            # Handwriting conversion
├── rag/              # Parser, embeddings, search, query engine
├── summarizer/       # Document summarization
└── questions/        # Research question workflows

tests/
├── conftest.py       # Shared fixtures
├── fixtures/         # Test data files
└── test_*.py         # Test modules

docs/                 # Documentation
config/               # Configuration files (section weights YAML)
```

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open an Issue to discuss before implementing

## License

By contributing, you agree that your contributions will be licensed under the project's GPLv3 license.
