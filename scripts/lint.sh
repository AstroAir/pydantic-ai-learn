#!/usr/bin/env bash
# Run code quality checks for pydantic-ai-learn project
#
# Usage:
#   ./scripts/lint.sh [--tool TOOL] [--fix] [--strict]
#
# Tools: ruff, mypy, bandit, pip-audit, all

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
TOOL="all"
FIX=false
STRICT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            TOOL="$2"
            shift 2
            ;;
        --fix)
            FIX=true
            shift
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}=== Code Quality Checks ===${NC}"
echo ""

ALL_PASSED=true

# Ruff linting
run_ruff() {
    echo -e "${CYAN}Running Ruff linting...${NC}"

    if ! python3 -m ruff --version &> /dev/null; then
        echo -e "${YELLOW}! Ruff not installed, skipping${NC}"
        return 0
    fi

    if [ "$FIX" = true ]; then
        echo -e "${CYAN}Auto-fixing issues...${NC}"
        python3 -m ruff check . --fix
    else
        python3 -m ruff check .
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Ruff: No issues found${NC}"
        return 0
    else
        echo -e "${RED}✗ Ruff: Issues found${NC}"
        return 1
    fi
}

# MyPy type checking
run_mypy() {
    echo -e "${CYAN}Running MyPy type checking...${NC}"

    if ! python3 -m mypy --version &> /dev/null; then
        echo -e "${YELLOW}! MyPy not installed, skipping${NC}"
        return 0
    fi

    MYPY_ARGS=("code_agent" "tools" "utils" "--show-error-codes" "--pretty")

    if [ "$STRICT" = true ]; then
        MYPY_ARGS+=("--strict")
    fi

    python3 -m mypy "${MYPY_ARGS[@]}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ MyPy: No type errors${NC}"
        return 0
    else
        echo -e "${YELLOW}! MyPy: Type errors found${NC}"
        return 1
    fi
}

# Bandit security scan
run_bandit() {
    echo -e "${CYAN}Running Bandit security scan...${NC}"

    if ! python3 -m bandit --version &> /dev/null; then
        echo -e "${YELLOW}! Bandit not installed, skipping${NC}"
        return 0
    fi

    python3 -m bandit -r code_agent tools utils -f screen

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Bandit: No security issues${NC}"
        return 0
    else
        echo -e "${YELLOW}! Bandit: Security issues found${NC}"
        return 1
    fi
}

# pip-audit dependency check
run_pip_audit() {
    echo -e "${CYAN}Running pip-audit dependency check...${NC}"

    if ! python3 -m pip_audit --version &> /dev/null; then
        echo -e "${YELLOW}! pip-audit not installed, skipping${NC}"
        return 0
    fi

    python3 -m pip_audit --requirement requirements.txt --requirement requirements-dev.txt

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ pip-audit: No vulnerable dependencies${NC}"
        return 0
    else
        echo -e "${YELLOW}! pip-audit: Vulnerable dependencies found${NC}"
        return 1
    fi
}

# Run selected tools
case $TOOL in
    ruff)
        run_ruff || ALL_PASSED=false
        ;;
    mypy)
        run_mypy || ALL_PASSED=false
        ;;
    bandit)
        run_bandit || ALL_PASSED=false
        ;;
    pip-audit)
        run_pip_audit || ALL_PASSED=false
        ;;
    all)
        run_ruff || ALL_PASSED=false
        echo ""
        run_mypy || ALL_PASSED=false
        echo ""

        if [ "$STRICT" = true ]; then
            run_bandit || ALL_PASSED=false
            echo ""
            run_pip_audit || ALL_PASSED=false
            echo ""
        fi
        ;;
    *)
        echo -e "${RED}Unknown tool: $TOOL${NC}"
        exit 1
        ;;
esac

# Summary
echo -e "${CYAN}=== Summary ===${NC}"
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo ""
    echo -e "${CYAN}Tips:${NC}"
    echo "  - Run with --fix to auto-fix Ruff issues"
    echo "  - Check output above for specific issues"
    echo "  - See CONTRIBUTING.md for coding standards"
    exit 1
fi
