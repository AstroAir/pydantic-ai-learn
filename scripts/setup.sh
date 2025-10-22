#!/usr/bin/env bash
# Setup script for pydantic-ai-learn project
#
# Usage:
#   ./scripts/setup.sh [--skip-venv] [--dev-only]
#
# Options:
#   --skip-venv   Skip virtual environment creation
#   --dev-only    Install only development dependencies

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_VENV=false
DEV_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --dev-only)
            DEV_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}=== pydantic-ai-learn Setup Script ===${NC}"
echo ""

# Check Python version
echo -e "${CYAN}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.12 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Found: Python $PYTHON_VERSION${NC}"

# Check version is 3.12+
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 12 ]); then
    echo -e "${RED}✗ Python 3.12+ required, found $MAJOR.$MINOR${NC}"
    exit 1
fi

# Create virtual environment
if [ "$SKIP_VENV" = false ]; then
    echo ""
    echo -e "${CYAN}Creating virtual environment...${NC}"

    if [ -d "venv" ]; then
        echo -e "${YELLOW}! Virtual environment already exists${NC}"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${CYAN}Removing existing virtual environment...${NC}"
            rm -rf venv
            python3 -m venv venv
            echo -e "${GREEN}✓ Virtual environment recreated${NC}"
        else
            echo -e "${CYAN}Using existing virtual environment${NC}"
        fi
    else
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    fi

    # Activate virtual environment
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Upgrade pip
echo ""
echo -e "${CYAN}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
if [ "$DEV_ONLY" = true ]; then
    echo -e "${CYAN}Installing development dependencies only...${NC}"
    python3 -m pip install -r requirements-dev.txt --quiet
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
else
    echo -e "${CYAN}Installing project dependencies...${NC}"
    python3 -m pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Project dependencies installed${NC}"

    echo -e "${CYAN}Installing development dependencies...${NC}"
    python3 -m pip install -r requirements-dev.txt --quiet
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Verify installation
echo ""
echo -e "${CYAN}Verifying installation...${NC}"

PACKAGES=("pydantic-ai" "pytest" "mypy" "ruff")
ALL_INSTALLED=true

for package in "${PACKAGES[@]}"; do
    if python3 -m pip show "$package" &> /dev/null; then
        echo -e "${GREEN}✓ $package installed${NC}"
    else
        echo -e "${RED}✗ $package not found${NC}"
        ALL_INSTALLED=false
    fi
done

# Run quick test
echo ""
echo -e "${CYAN}Running quick verification test...${NC}"
if python3 -c "import code_agent; import tools; import utils; print('OK')" 2>&1 | grep -q "OK"; then
    echo -e "${GREEN}✓ Package imports successful${NC}"
else
    echo -e "${YELLOW}! Could not verify package imports (this is OK if running outside project root)${NC}"
fi

# Summary
echo ""
echo -e "${CYAN}=== Setup Complete ===${NC}"
echo -e "${GREEN}✓ Environment ready for development${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Activate virtual environment (if not already active):"
echo "     source venv/bin/activate"
echo "  2. Run tests: ./scripts/run_tests.sh"
echo "  3. Run examples: ./scripts/run_examples.sh"
echo "  4. Check code quality: ./scripts/lint.sh"
echo ""

if [ "$ALL_INSTALLED" = false ]; then
    echo -e "${YELLOW}! Some packages failed to install. Please check the output above.${NC}"
    exit 1
fi

exit 0
