#!/bin/bash
# setup_company.sh - Clone Case Template for a new company analysis
#
# Usage: ./scripts/setup_company.sh TICKER [DEST_DIR]
#
# Examples:
#   ./scripts/setup_company.sh AAPL                    # Creates ../Long/AAPL
#   ./scripts/setup_company.sh AAPL /path/to/AAPL-Case # Custom destination

set -e

TICKER="${1:?Usage: $0 TICKER [DEST_DIR]}"
TICKER_UPPER=$(echo "$TICKER" | tr '[:lower:]' '[:upper:]')
TICKER_LOWER=$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')

# Default destination: ../Long/TICKER (relative to Case Template)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_DEST="$(dirname "$TEMPLATE_DIR")/Long/$TICKER_UPPER"
DEST_DIR="${2:-$DEFAULT_DEST}"

echo "=== Setting up $TICKER_UPPER case study ==="
echo "Template: $TEMPLATE_DIR"
echo "Destination: $DEST_DIR"
echo

# Check if destination already exists
if [ -d "$DEST_DIR" ]; then
    echo "ERROR: Destination already exists: $DEST_DIR"
    exit 1
fi

# Clone the template
echo "1. Cloning template..."
git clone "$TEMPLATE_DIR" "$DEST_DIR"

cd "$DEST_DIR"

# Set up git remotes and branch
echo "2. Setting up git..."
git remote rename origin template
git checkout -b "${TICKER_LOWER}-analysis"

# Create additional directories
echo "3. Creating directory structure..."
mkdir -p conferences data index output processed model
for dir in conferences data index output processed model; do
    touch "$dir/.gitkeep"
done

# Copy .env if it exists in a sibling directory
echo "4. Checking for .env file..."
for sibling in "$TEMPLATE_DIR"/../Long/*/; do
    if [ -f "${sibling}.env" ]; then
        cp "${sibling}.env" .env
        echo "   Copied .env from $(basename "$sibling")"
        break
    fi
done

if [ ! -f .env ]; then
    cp .env.template .env
    echo "   Created .env from template (edit to add API keys)"
fi

# Update COMPANY.md placeholder
echo "5. Updating COMPANY.md..."
sed -i "s/{TICKER}/$TICKER_UPPER/g" COMPANY.md
sed -i "s/{Company Name}/[Company Name - update this]/g" COMPANY.md

echo
echo "=== Setup complete! ==="
echo
echo "Next steps:"
echo "  cd \"$DEST_DIR\""
echo "  # Edit COMPANY.md with company-specific details"
echo "  # Edit .env with API keys if needed"
echo
echo "  # Fetch SEC filings"
echo "  poetry run pitch fetch $TICKER_UPPER -t 10-K -y 3"
echo "  poetry run pitch fetch $TICKER_UPPER -t 10-Q -y 2"
echo
echo "  # Build index and summarize"
echo "  poetry run pitch index $TICKER_UPPER --source all"
echo "  poetry run pitch summarize $TICKER_UPPER --latest"
echo
echo "  # Commit initial setup"
echo "  git add . && git commit -m \"Initial $TICKER_UPPER setup\""
