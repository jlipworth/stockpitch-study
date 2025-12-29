# ci/Dockerfile
# Base image for stock-pitch CI testing
# Contains Python 3.12, uv, and all project dependencies pre-installed
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
    # curl for uv installer
    curl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# uv installation
# =============================================================================
ENV UV_VERSION=0.5
ENV UV_INSTALL_DIR=/opt/uv
ENV PATH="$UV_INSTALL_DIR/bin:$PATH"

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh

# =============================================================================
# Working directory
# =============================================================================
WORKDIR /workspace

# =============================================================================
# Copy dependency files first (better layer caching)
# =============================================================================
COPY pyproject.toml uv.lock ./

# Generate hash of pyproject.toml for dependency change detection
# This allows CI to detect when the image needs rebuilding
RUN sha256sum pyproject.toml > /workspace/.deps-hash

# =============================================================================
# Install dependencies with CPU-only PyTorch
# =============================================================================
# Strategy: Export requirements from uv, filter CUDA deps, install with pip
# 1. Export requirements.txt from pyproject.toml
# 2. Filter out torch and nvidia-* packages (we'll install CPU torch separately)
# 3. Install CPU-only torch first
# 4. pip install remaining deps
# Result: ~2GB image, not 10GB
RUN uv export --no-hashes --all-groups > requirements.txt && \
    grep -v -E "^(torch|nvidia-|triton)" requirements.txt > requirements-cpu.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-cpu.txt && \
    rm requirements.txt requirements-cpu.txt

# =============================================================================
# Verification
# =============================================================================
RUN python --version && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" && \
    pip list | grep -i torch

# =============================================================================
# Entry point
# =============================================================================
CMD ["python", "--version"]
