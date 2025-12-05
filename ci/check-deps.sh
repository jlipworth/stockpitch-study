#!/bin/bash
# ci/check-deps.sh
# Ensure all dependencies are available for CI, with multiple fallback layers:
#
# 1. If custom image has fresh deps → use them (fastest)
# 2. If custom image has stale deps → run poetry install (medium)
# 3. If poetry not installed → install poetry + deps (slowest, full fallback)
#
# This ensures CI always works, even if:
# - The custom image hasn't been pushed yet
# - pyproject.toml changed since the image was built
# - Using a base python image instead of custom image

set -e

echo "=== Checking CI environment ==="

# ---------------------------------------------------------------------------
# Step 1: Check if Poetry is available
# ---------------------------------------------------------------------------
if ! command -v poetry &> /dev/null; then
    echo "⚠️  Poetry not found - installing..."
    pip install --quiet poetry
    echo "✓ Poetry installed"
fi

# ---------------------------------------------------------------------------
# Step 2: Check dependency freshness
# ---------------------------------------------------------------------------
CURRENT_HASH=$(sha256sum pyproject.toml | awk '{print $1}')
IMAGE_HASH=""
DEPS_FRESH=false

if [[ -f /workspace/.deps-hash ]]; then
    IMAGE_HASH=$(cat /workspace/.deps-hash | awk '{print $1}')
    if [[ "$CURRENT_HASH" == "$IMAGE_HASH" ]]; then
        DEPS_FRESH=true
    fi
fi

if [[ "$DEPS_FRESH" == "true" ]]; then
    echo "✓ Dependencies are up to date with CI image"
else
    echo ""
    if [[ -z "$IMAGE_HASH" ]]; then
        echo "ℹ️  No pre-built dependencies found (using base image or first run)"
    else
        echo "⚠️  pyproject.toml has changed since the CI image was built!"
        echo "   Current hash: ${CURRENT_HASH:0:16}..."
        echo "   Image hash:   ${IMAGE_HASH:0:16}..."
    fi
    echo ""
    echo "   Installing dependencies now (this will be slower)..."
    echo ""

    # Configure poetry for CI
    poetry config virtualenvs.create true
    poetry config virtualenvs.in-project true

    # Install dependencies
    poetry install --no-root --no-interaction

    echo ""
    echo "✓ Dependencies installed successfully"

    if [[ -n "$IMAGE_HASH" ]]; then
        echo ""
        echo "   TIP: Rebuild the CI image for faster builds:"
        echo "   ./ci/build-image.sh --push"
    fi
fi

echo ""
echo "=== Environment ready ==="
