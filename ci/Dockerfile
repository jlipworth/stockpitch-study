# CI base image for stock-pitch
# Build: ./ci/build-image.sh
# Push:  ./ci/build-image.sh --push

FROM python:3.12-slim

LABEL maintainer="Jonathan Lipworth <jonathan.lipworth@gmail.com>"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Copy and install dependencies with CPU-only PyTorch
COPY pyproject.toml uv.lock ./

# Install CPU-only PyTorch first (avoids 10GB CUDA packages)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch already satisfied)
RUN uv sync --frozen --all-groups

# Add venv to PATH so we don't need "uv run" everywhere
ENV PATH="/workspace/.venv/bin:$PATH"

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')"

CMD ["bash"]
