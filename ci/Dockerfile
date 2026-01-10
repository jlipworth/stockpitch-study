# ci/Dockerfile
# Base image for stock-pitch CI testing
# Contains Python 3.12, uv, and pre-cached wheels for fast installs
#
# Build and push using: ./ci/build-image.sh --push

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official image (faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Pre-cache dev dependency wheels (for lint/typecheck)
WORKDIR /tmp/cache-build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group dev --no-install-project && rm -rf .venv

# Pre-cache main deps (filtering CUDA/torch) and install CPU-only torch
# This creates /opt/venv for the test step to use directly
# Install CPU torch FIRST so transitive deps (sentence-transformers) use it
RUN uv export --frozen --all-groups --no-emit-project --no-hashes | \
    grep -v "^nvidia-" | grep -v "^triton==" | grep -v "^torch==" > /tmp/reqs.txt && \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --python /opt/venv/bin/python -r /tmp/reqs.txt && \
    rm -f /tmp/reqs.txt

# Clean up temp dir
WORKDIR /
RUN rm -rf /tmp/cache-build

CMD ["bash"]
