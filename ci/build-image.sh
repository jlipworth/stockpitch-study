#!/bin/bash
# Build and optionally push the CI base image
# Usage:
#   ./ci/build-image.sh              # Build locally
#   ./ci/build-image.sh --push       # Build multi-arch and push
#   ./ci/build-image.sh --no-cache   # Build without cache

set -e

IMAGE="jlipworth/stock-pitch-ci"
cd "$(dirname "$0")/.."

# Parse args
PUSH=false
NO_CACHE=""
for arg in "$@"; do
    case $arg in
        --push) PUSH=true ;;
        --no-cache) NO_CACHE="--no-cache" ;;
    esac
done

if [[ "$PUSH" == "true" ]]; then
    DATE_TAG=$(date +%Y.%m.%d)
    echo "Building and pushing: ${IMAGE}:latest and ${IMAGE}:${DATE_TAG}"

    docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -t "${IMAGE}:latest" \
        -t "${IMAGE}:${DATE_TAG}" \
        -f ci/Dockerfile \
        $NO_CACHE \
        --push .

    echo "Done: ${IMAGE}:latest, ${IMAGE}:${DATE_TAG}"
else
    echo "Building locally: ${IMAGE}:latest"
    docker build -t "${IMAGE}:latest" -f ci/Dockerfile $NO_CACHE .
    echo "To push: $0 --push"
fi
