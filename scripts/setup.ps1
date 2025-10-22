#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup script for pydantic-ai-learn project

.DESCRIPTION
    This script sets up the development environment by:
    - Checking Python version
    - Creating virtual environment
    - Installing dependencies
    - Verifying installation

.PARAMETER SkipVenv
    Skip virtual environment creation

.PARAMETER DevOnly
    Install only development dependencies

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -SkipVenv
#>

param(
    [switch]$SkipVenv,
    [switch]$DevOnly
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "=== pydantic-ai-learn Setup Script ==="
Write-Info ""

# Check Python version
Write-Info "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "✓ Found: $pythonVersion"

    # Extract version number
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]

        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 12)) {
            Write-Error "✗ Python 3.12+ required, found $major.$minor"
            exit 1
        }
    }
} catch {
    Write-Error "✗ Python not found. Please install Python 3.12 or higher."
    exit 1
}

# Create virtual environment
if (-not $SkipVenv) {
    Write-Info ""
    Write-Info "Creating virtual environment..."

    if (Test-Path "venv") {
        Write-Warning "! Virtual environment already exists"
        $response = Read-Host "Do you want to recreate it? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Write-Info "Removing existing virtual environment..."
            Remove-Item -Recurse -Force venv
            python -m venv venv
            Write-Success "✓ Virtual environment recreated"
        } else {
            Write-Info "Using existing virtual environment"
        }
    } else {
        python -m venv venv
        Write-Success "✓ Virtual environment created"
    }

    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    if ($IsWindows -or $env:OS -match "Windows") {
        & ".\venv\Scripts\Activate.ps1"
    } else {
        Write-Warning "! Please activate manually: source venv/bin/activate"
    }
}

# Upgrade pip
Write-Info ""
Write-Info "Upgrading pip..."
python -m pip install --upgrade pip --quiet
Write-Success "✓ pip upgraded"

# Install dependencies
Write-Info ""
if ($DevOnly) {
    Write-Info "Installing development dependencies only..."
    python -m pip install -r requirements-dev.txt --quiet
    Write-Success "✓ Development dependencies installed"
} else {
    Write-Info "Installing project dependencies..."
    python -m pip install -r requirements.txt --quiet
    Write-Success "✓ Project dependencies installed"

    Write-Info "Installing development dependencies..."
    python -m pip install -r requirements-dev.txt --quiet
    Write-Success "✓ Development dependencies installed"
}

# Verify installation
Write-Info ""
Write-Info "Verifying installation..."

$packages = @("pydantic-ai", "pytest", "mypy", "ruff")
$allInstalled = $true

foreach ($package in $packages) {
    try {
        $null = python -m pip show $package 2>&1
        Write-Success "✓ $package installed"
    } catch {
        Write-Error "✗ $package not found"
        $allInstalled = $false
    }
}

# Run a quick test
Write-Info ""
Write-Info "Running quick verification test..."
try {
    $testResult = python -c "import code_agent; import tools; import utils; print('OK')" 2>&1
    if ($testResult -match "OK") {
        Write-Success "✓ Package imports successful"
    } else {
        Write-Warning "! Package import test returned: $testResult"
    }
} catch {
    Write-Warning "! Could not verify package imports (this is OK if running outside project root)"
}

# Summary
Write-Info ""
Write-Info "=== Setup Complete ==="
Write-Success "✓ Environment ready for development"
Write-Info ""
Write-Info "Next steps:"
Write-Info "  1. Activate virtual environment (if not already active):"
if ($IsWindows -or $env:OS -match "Windows") {
    Write-Info "     .\venv\Scripts\Activate.ps1"
} else {
    Write-Info "     source venv/bin/activate"
}
Write-Info "  2. Run tests: .\scripts\run_tests.ps1"
Write-Info "  3. Run examples: .\scripts\run_examples.ps1"
Write-Info "  4. Check code quality: .\scripts\lint.ps1"
Write-Info ""

if (-not $allInstalled) {
    Write-Warning "! Some packages failed to install. Please check the output above."
    exit 1
}

exit 0
