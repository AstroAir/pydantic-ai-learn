#!/usr/bin/env bash
# Run examples for pydantic-ai-learn project
#
# Usage:
#   ./scripts/run_examples.sh [--category CATEGORY] [--file FILE] [--list]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
CATEGORY=""
FILE=""
LIST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --file)
            FILE="$2"
            shift 2
            ;;
        --list)
            LIST=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# List examples
if [ "$LIST" = true ]; then
    echo -e "${CYAN}=== Available Examples ===${NC}"
    echo ""
    echo -e "${CYAN}Category: simple${NC}"
    echo "  - examples/simple_demo.py"
    echo ""
    for cat in basic messages output tools graph mcp multi-agent; do
        echo -e "${CYAN}Category: $cat${NC}"
        find "examples/$cat" -name "*.py" ! -name "*test*" 2>/dev/null | sed 's/^/  - /' || true
        echo ""
    done
    exit 0
fi

# Run specific file
if [ -n "$FILE" ]; then
    echo -e "${CYAN}=== Running Example: $FILE ===${NC}"
    echo ""

    if [ ! -f "$FILE" ]; then
        echo -e "${RED}✗ File not found: $FILE${NC}"
        exit 1
    fi

    python3 "$FILE"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Example completed successfully${NC}"
    else
        echo -e "${RED}✗ Example failed with exit code: $EXIT_CODE${NC}"
    fi

    exit $EXIT_CODE
fi

# Determine category
if [ -z "$CATEGORY" ]; then
    CATEGORY="simple"
    echo -e "${YELLOW}! Running only 'simple' category by default${NC}"
    echo -e "${CYAN}  Use --category to run specific categories or --list to see all${NC}"
    echo ""
fi

# Run examples
TOTAL=0
SUCCESS=0
FAILED=0

echo -e "${CYAN}=== Running Examples ===${NC}"
echo ""
echo -e "${CYAN}Category: $CATEGORY${NC}"
echo ""

if [ "$CATEGORY" = "simple" ]; then
    EXAMPLES=("examples/simple_demo.py")
else
    mapfile -t EXAMPLES < <(find "examples/$CATEGORY" -name "*.py" ! -name "*test*" 2>/dev/null || true)
fi

for example in "${EXAMPLES[@]}"; do
    if [ ! -f "$example" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    EXAMPLE_NAME=$(basename "$example")

    echo -e "${CYAN}Running: $EXAMPLE_NAME${NC}"

    if timeout 30s python3 "$example" &> /dev/null; then
        echo -e "${GREEN}  ✓ Passed${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}  ✗ Failed${NC}"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# Summary
echo -e "${CYAN}=== Summary ===${NC}"
echo -e "${CYAN}Total: $TOTAL${NC}"
echo -e "${GREEN}Passed: $SUCCESS${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
fi

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All examples completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some examples failed${NC}"
    exit 1
fi
