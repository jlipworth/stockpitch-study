#!/bin/bash
# ci/build-image.sh
# Build and optionally push the CI base image to Docker Hub
#
# Usage:
#   ./ci/build-image.sh              # Build only (local arch)
#   ./ci/build-image.sh --push       # Build multi-arch and push

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
DOCKERHUB_USER="${DOCKERHUB_USER:-jlipworth}"
IMAGE_NAME="stock-pitch-ci"
TAG="${TAG:-latest}"

FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

cd "$(dirname "$0")/.." # Go to repo root

# ============================================================================
# Build
# ============================================================================
if [[ "$1" == "--push" ]]; then
    # Multi-architecture build (amd64 + arm64) - requires buildx
    DATE_TAG=$(date +%Y.%m.%d)
    echo "Building multi-arch image: ${FULL_IMAGE}"
    echo "Also tagging as: ${DOCKERHUB_USER}/${IMAGE_NAME}:${DATE_TAG}"
    echo "Platforms: linux/amd64, linux/arm64"

    # Create buildx builder if it doesn't exist
    docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch

    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -t "${FULL_IMAGE}" \
        -t "${DOCKERHUB_USER}/${IMAGE_NAME}:${DATE_TAG}" \
        -f ci/Dockerfile \
        --push \
        .

    echo ""
    echo "Multi-arch build and push complete!"
    echo "Tags pushed: latest, ${DATE_TAG}"
    echo "Image available at: https://hub.docker.com/r/${DOCKERHUB_USER}/${IMAGE_NAME}"
    echo ""
    echo "Update .woodpecker.yml to use: ${DOCKERHUB_USER}/${IMAGE_NAME}:${DATE_TAG}"
else
    # Local build only (current architecture)
    echo "Building image for local arch: ${FULL_IMAGE}"

    docker build \
        -t "${FULL_IMAGE}" \
        -f ci/Dockerfile \
        .

    echo "Build complete: ${FULL_IMAGE}"
    echo ""
    echo "To build multi-arch and push to Docker Hub, run:"
    echo "  ./ci/build-image.sh --push"
fi
