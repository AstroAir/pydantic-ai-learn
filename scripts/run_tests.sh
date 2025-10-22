#!/usr/bin/env bash
# Run tests for pydantic-ai-learn project
#
# Usage:
#   ./scripts/run_tests.sh [OPTIONS] [PATH]
#
# Options:
#   --coverage    Generate coverage report
#   --verbose     Run with verbose output
#   --fast        Run tests in parallel
#   --fail-fast   Stop on first failure
#   --markers     Run tests with specific markers

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
PATH_ARG="tests"
COVERAGE=false
VERBOSE=false
FAST=false
FAIL_FAST=false
MARKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --fast)
            FAST=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --markers)
            MARKERS="$2"
            shift 2
            ;;
        *)
            PATH_ARG="$1"
            shift
            ;;
    esac
done

echo -e "${CYAN}=== Running Tests ===${NC}"
echo ""

# Check if pytest is installed
if ! python3 -m pytest --version &> /dev/null; then
    echo -e "${RED}✗ pytest not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Build pytest command
PYTEST_ARGS=("$PATH_ARG")

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v")
else
    PYTEST_ARGS+=("-q")
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    echo -e "${CYAN}Generating coverage report...${NC}"
    PYTEST_ARGS+=("--cov=code_agent" "--cov=tools" "--cov=utils")
    PYTEST_ARGS+=("--cov-report=term" "--cov-report=html" "--cov-report=xml")
fi

# Add parallel execution
if [ "$FAST" = true ]; then
    echo -e "${CYAN}Running tests in parallel...${NC}"
    PYTEST_ARGS+=("-n" "auto")
fi

# Add fail fast
if [ "$FAIL_FAST" = true ]; then
    PYTEST_ARGS+=("--maxfail=1")
fi

# Add markers
if [ -n "$MARKERS" ]; then
    echo -e "${CYAN}Running tests with markers: $MARKERS${NC}"
    PYTEST_ARGS+=("-m" "$MARKERS")
fi

# Add standard options
PYTEST_ARGS+=("--tb=short" "--strict-markers")

# Run tests
echo -e "${CYAN}Running: pytest ${PYTEST_ARGS[*]}${NC}"
echo ""

START_TIME=$(date +%s)
python3 -m pytest "${PYTEST_ARGS[@]}"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}=== Test Results ===${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
elif [ $EXIT_CODE -eq 1 ]; then
    echo -e "${RED}✗ Some tests failed${NC}"
elif [ $EXIT_CODE -eq 5 ]; then
    echo -e "${YELLOW}! No tests collected${NC}"
else
    echo -e "${RED}✗ Test execution error (code: $EXIT_CODE)${NC}"
fi

echo -e "${CYAN}Duration: ${DURATION}s${NC}"

if [ "$COVERAGE" = true ]; then
    echo ""
    echo -e "${GREEN}✓ Coverage report generated:${NC}"
    echo "  - HTML: htmlcov/index.html"
    echo "  - XML: coverage.xml"
fi

exit $EXIT_CODE
