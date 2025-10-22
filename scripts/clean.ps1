#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean build artifacts and cache files

.DESCRIPTION
    This script removes various build artifacts, cache files, and temporary files:
    - Python cache files (__pycache__, *.pyc, *.pyo)
    - Test cache (.pytest_cache)
    - Coverage reports (htmlcov/, .coverage)
    - Build directories (build/, dist/, *.egg-info)
    - MyPy cache (.mypy_cache)
    - Ruff cache (.ruff_cache)
    - Temporary files (*.tmp, *.log)

.PARAMETER All
    Remove everything including virtual environment

.PARAMETER DryRun
    Show what would be deleted without actually deleting

.EXAMPLE
    .\scripts\clean.ps1
    .\scripts\clean.ps1 -All
    .\scripts\clean.ps1 -DryRun
#>

param(
    [switch]$All,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "=== Cleaning Project ==="
Write-Info ""

if ($DryRun) {
    Write-Warning "! DRY RUN MODE - No files will be deleted"
    Write-Info ""
}

$itemsRemoved = 0
$bytesFreed = 0

# Function to remove items
function Remove-Items {
    param(
        [string]$Pattern,
        [string]$Description
    )

    Write-Info "Cleaning: $Description"

    $items = Get-ChildItem -Path . -Recurse -Force -Include $Pattern -ErrorAction SilentlyContinue

    foreach ($item in $items) {
        $size = 0
        if ($item.PSIsContainer) {
            $size = (Get-ChildItem -Path $item.FullName -Recurse -File -ErrorAction SilentlyContinue |
                     Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        } else {
            $size = $item.Length
        }

        if ($size) {
            $script:bytesFreed += $size
        }

        if ($DryRun) {
            Write-Info "  Would remove: $($item.FullName)"
        } else {
            try {
                Remove-Item -Path $item.FullName -Recurse -Force -ErrorAction Stop
                Write-Info "  Removed: $($item.FullName)"
                $script:itemsRemoved++
            } catch {
                Write-Warning "  Failed to remove: $($item.FullName)"
            }
        }
    }
}

# Python cache files
Remove-Items -Pattern "__pycache__" -Description "Python cache directories"
Remove-Items -Pattern "*.pyc" -Description "Python compiled files (.pyc)"
Remove-Items -Pattern "*.pyo" -Description "Python optimized files (.pyo)"
Remove-Items -Pattern "*.pyd" -Description "Python DLL files (.pyd)"

# Test cache
Remove-Items -Pattern ".pytest_cache" -Description "Pytest cache"

# Coverage reports
Remove-Items -Pattern "htmlcov" -Description "HTML coverage reports"
Remove-Items -Pattern ".coverage" -Description "Coverage data files"
Remove-Items -Pattern "coverage.xml" -Description "Coverage XML reports"
Remove-Items -Pattern ".coverage.*" -Description "Coverage data files (parallel)"

# Build directories
Remove-Items -Pattern "build" -Description "Build directories"
Remove-Items -Pattern "dist" -Description "Distribution directories"
Remove-Items -Pattern "*.egg-info" -Description "Egg info directories"
Remove-Items -Pattern "*.egg" -Description "Egg files"

# Type checker cache
Remove-Items -Pattern ".mypy_cache" -Description "MyPy cache"
Remove-Items -Pattern ".pytype" -Description "Pytype cache"
Remove-Items -Pattern ".pyre" -Description "Pyre cache"

# Linter cache
Remove-Items -Pattern ".ruff_cache" -Description "Ruff cache"

# Temporary files
Remove-Items -Pattern "*.tmp" -Description "Temporary files (.tmp)"
Remove-Items -Pattern "*.temp" -Description "Temporary files (.temp)"
Remove-Items -Pattern "*.log" -Description "Log files"
Remove-Items -Pattern "*.swp" -Description "Vim swap files"
Remove-Items -Pattern "*.swo" -Description "Vim swap files"
Remove-Items -Pattern "*~" -Description "Backup files"

# OS-specific
Remove-Items -Pattern ".DS_Store" -Description "macOS metadata files"
Remove-Items -Pattern "Thumbs.db" -Description "Windows thumbnail cache"
Remove-Items -Pattern "Desktop.ini" -Description "Windows desktop config"

# Virtual environment (if -All specified)
if ($All) {
    Write-Warning "! Removing virtual environment"
    Remove-Items -Pattern "venv" -Description "Virtual environment"
    Remove-Items -Pattern ".venv" -Description "Virtual environment (.venv)"
    Remove-Items -Pattern "env" -Description "Virtual environment (env)"
}

# Summary
Write-Info ""
Write-Info "=== Cleanup Summary ==="

if ($DryRun) {
    Write-Warning "! DRY RUN - No files were actually deleted"
} else {
    Write-Success "✓ Cleanup complete"
    Write-Info "Items removed: $itemsRemoved"

    $mbFreed = [math]::Round($bytesFreed / 1MB, 2)
    if ($mbFreed -gt 0) {
        Write-Info "Space freed: $mbFreed MB"
    }
}

Write-Info ""

if ($All) {
    Write-Info "Virtual environment removed. Run setup.ps1 to recreate it."
} else {
    Write-Info "To also remove virtual environment, run with -All flag:"
    Write-Info "  .\scripts\clean.ps1 -All"
}

exit 0
