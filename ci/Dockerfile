# CI base image for stock-pitch
# Build: ./ci/build-image.sh
# Push:  ./ci/build-image.sh --push

FROM python:3.12-slim

LABEL maintainer="Jonathan Lipworth <jonathan.lipworth@gmail.com>"

# Copy uv from official image (recommended approach)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Export deps excluding torch and CUDA packages, then add CPU-only torch
# Install to /opt/venv so it survives CI workspace mount at /workspace
RUN uv export --frozen --all-groups --no-emit-project --no-hashes | \
    grep -v "^torch==" | grep -v "^nvidia-" | grep -v "^triton==" > /tmp/reqs.txt && \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r /tmp/reqs.txt && \
    uv pip install --python /opt/venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

# Add venv to PATH so we don't need "uv run" everywhere
ENV PATH="/opt/venv/bin:$PATH"

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')"

CMD ["bash"]
