#!/usr/bin/env bash
# Clean build artifacts and cache files
#
# Usage:
#   ./scripts/clean.sh [--all] [--dry-run]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
ALL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}=== Cleaning Project ===${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}! DRY RUN MODE - No files will be deleted${NC}"
    echo ""
fi

ITEMS_REMOVED=0

# Function to remove items
remove_items() {
    local pattern=$1
    local description=$2

    echo -e "${CYAN}Cleaning: $description${NC}"

    while IFS= read -r -d '' item; do
        if [ "$DRY_RUN" = true ]; then
            echo -e "${CYAN}  Would remove: $item${NC}"
        else
            rm -rf "$item"
            echo -e "${CYAN}  Removed: $item${NC}"
            ITEMS_REMOVED=$((ITEMS_REMOVED + 1))
        fi
    done < <(find . -name "$pattern" -print0 2>/dev/null)
}

# Python cache files
remove_items "__pycache__" "Python cache directories"
remove_items "*.pyc" "Python compiled files (.pyc)"
remove_items "*.pyo" "Python optimized files (.pyo)"
remove_items "*.pyd" "Python DLL files (.pyd)"

# Test cache
remove_items ".pytest_cache" "Pytest cache"

# Coverage reports
remove_items "htmlcov" "HTML coverage reports"
remove_items ".coverage" "Coverage data files"
remove_items "coverage.xml" "Coverage XML reports"
remove_items ".coverage.*" "Coverage data files (parallel)"

# Build directories
remove_items "build" "Build directories"
remove_items "dist" "Distribution directories"
remove_items "*.egg-info" "Egg info directories"
remove_items "*.egg" "Egg files"

# Type checker cache
remove_items ".mypy_cache" "MyPy cache"
remove_items ".pytype" "Pytype cache"
remove_items ".pyre" "Pyre cache"

# Linter cache
remove_items ".ruff_cache" "Ruff cache"

# Temporary files
remove_items "*.tmp" "Temporary files (.tmp)"
remove_items "*.temp" "Temporary files (.temp)"
remove_items "*.log" "Log files"
remove_items "*.swp" "Vim swap files"
remove_items "*.swo" "Vim swap files"
remove_items "*~" "Backup files"

# OS-specific
remove_items ".DS_Store" "macOS metadata files"
remove_items "Thumbs.db" "Windows thumbnail cache"
remove_items "Desktop.ini" "Windows desktop config"

# Virtual environment (if --all specified)
if [ "$ALL" = true ]; then
    echo -e "${YELLOW}! Removing virtual environment${NC}"
    remove_items "venv" "Virtual environment"
    remove_items ".venv" "Virtual environment (.venv)"
    remove_items "env" "Virtual environment (env)"
fi

# Summary
echo ""
echo -e "${CYAN}=== Cleanup Summary ===${NC}"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}! DRY RUN - No files were actually deleted${NC}"
else
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    echo -e "${CYAN}Items removed: $ITEMS_REMOVED${NC}"
fi

echo ""

if [ "$ALL" = true ]; then
    echo -e "${CYAN}Virtual environment removed. Run setup.sh to recreate it.${NC}"
else
    echo -e "${CYAN}To also remove virtual environment, run with --all flag:${NC}"
    echo "  ./scripts/clean.sh --all"
fi

exit 0
