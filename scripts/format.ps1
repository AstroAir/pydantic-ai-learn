#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Format code for pydantic-ai-learn project

.DESCRIPTION
    This script formats Python code using Ruff formatter.
    Can check formatting or apply formatting.

.PARAMETER Check
    Check formatting without making changes

.PARAMETER Path
    Specific path to format (default: all files)

.EXAMPLE
    .\scripts\format.ps1
    .\scripts\format.ps1 -Check
    .\scripts\format.ps1 -Path code_agent/
#>

param(
    [switch]$Check,
    [string]$Path = "."
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "=== Code Formatting ==="
Write-Info ""

# Check if ruff is installed
try {
    $ruffVersion = python -m ruff --version 2>&1
    Write-Info "Using: $ruffVersion"
} catch {
    Write-Error "✗ Ruff not found. Please run setup.ps1 first."
    exit 1
}

# Verify path exists
if (-not (Test-Path $Path)) {
    Write-Error "✗ Path not found: $Path"
    exit 1
}

Write-Info ""

# Run formatter
if ($Check) {
    Write-Info "Checking code formatting (no changes will be made)..."
    Write-Info ""

    try {
        python -m ruff format --check $Path
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Success "✓ All files are properly formatted"
            exit 0
        } else {
            Write-Error "✗ Some files need formatting"
            Write-Info ""
            Write-Info "Run without -Check to format files automatically:"
            Write-Info "  .\scripts\format.ps1"
            exit 1
        }
    } catch {
        Write-Error "✗ Error checking formatting: $_"
        exit 1
    }
} else {
    Write-Info "Formatting code..."
    Write-Info ""

    try {
        python -m ruff format $Path
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Success "✓ Code formatted successfully"
            Write-Info ""
            Write-Info "Files have been formatted according to project style."
            Write-Info "Review changes with: git diff"
            exit 0
        } else {
            Write-Error "✗ Error formatting code"
            exit 1
        }
    } catch {
        Write-Error "✗ Error running formatter: $_"
        exit 1
    }
}
