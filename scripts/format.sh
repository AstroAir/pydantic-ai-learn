#!/usr/bin/env bash
# Format code for pydantic-ai-learn project
#
# Usage:
#   ./scripts/format.sh [--check] [PATH]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
CHECK=false
PATH_ARG="."

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK=true
            shift
            ;;
        *)
            PATH_ARG="$1"
            shift
            ;;
    esac
done

echo -e "${CYAN}=== Code Formatting ===${NC}"
echo ""

# Check if ruff is installed
if ! python3 -m ruff --version &> /dev/null; then
    echo -e "${RED}✗ Ruff not found. Please run setup.sh first.${NC}"
    exit 1
fi

RUFF_VERSION=$(python3 -m ruff --version)
echo -e "${CYAN}Using: $RUFF_VERSION${NC}"

# Verify path exists
if [ ! -e "$PATH_ARG" ]; then
    echo -e "${RED}✗ Path not found: $PATH_ARG${NC}"
    exit 1
fi

echo ""

# Run formatter
if [ "$CHECK" = true ]; then
    echo -e "${CYAN}Checking code formatting (no changes will be made)...${NC}"
    echo ""

    python3 -m ruff format --check "$PATH_ARG"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ All files are properly formatted${NC}"
        exit 0
    else
        echo -e "${RED}✗ Some files need formatting${NC}"
        echo ""
        echo -e "${CYAN}Run without --check to format files automatically:${NC}"
        echo "  ./scripts/format.sh"
        exit 1
    fi
else
    echo -e "${CYAN}Formatting code...${NC}"
    echo ""

    python3 -m ruff format "$PATH_ARG"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Code formatted successfully${NC}"
        echo ""
        echo -e "${CYAN}Files have been formatted according to project style.${NC}"
        echo "Review changes with: git diff"
        exit 0
    else
        echo -e "${RED}✗ Error formatting code${NC}"
        exit 1
    fi
fi
