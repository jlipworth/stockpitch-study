# ci/Dockerfile
# Base image for stock-pitch CI testing
# Contains Python 3.12, Poetry, and all project dependencies pre-installed
#
# Build:  ./ci/build-image.sh
# Push:   ./ci/build-image.sh --push
#
# The image includes a hash of pyproject.toml to detect when dependencies change.
# If pyproject.toml changes, rebuild and push a new image.

FROM python:3.12-slim

# =============================================================================
# Metadata
# =============================================================================
LABEL maintainer="Jonathan Lipworth <jonathan.lipworth@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/jlipworth/stock-pitch"
LABEL org.opencontainers.image.description="CI base image for stock-pitch project"

# =============================================================================
# System dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials (needed for some Python packages)
    build-essential \
    # Git (needed for pip VCS dependencies and pre-commit)
    git \
    # Certificates for HTTPS
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Poetry installation
# =============================================================================
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# =============================================================================
# Working directory
# =============================================================================
WORKDIR /workspace

# =============================================================================
# Copy dependency files first (better layer caching)
# =============================================================================
COPY pyproject.toml ./

# Generate hash of pyproject.toml for dependency change detection
# This allows CI to detect when the image needs rebuilding
RUN sha256sum pyproject.toml > /workspace/.deps-hash

# =============================================================================
# Install CPU-only PyTorch first (saves ~2-3GB per architecture)
# =============================================================================
# sentence-transformers pulls in full PyTorch by default (~2.5GB with CUDA)
# For CI testing (mocked embeddings), CPU-only is sufficient (~500MB)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# =============================================================================
# Install dependencies
# =============================================================================
# Install without dev dependencies first (smaller layer if only prod changes)
# Then install with dev dependencies for full test environment
RUN poetry install --no-root --only main && \
    poetry install --no-root

# =============================================================================
# Verification
# =============================================================================
RUN poetry run python --version && \
    poetry run pip list | head -20

# =============================================================================
# Entry point
# =============================================================================
# Default command shows installed packages
CMD ["poetry", "run", "python", "--version"]
