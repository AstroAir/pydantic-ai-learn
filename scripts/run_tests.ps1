#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run tests for pydantic-ai-learn project

.DESCRIPTION
    This script runs the test suite with various options:
    - Run all tests
    - Run specific test files or directories
    - Generate coverage reports
    - Run with different verbosity levels

.PARAMETER Path
    Specific test path to run (file or directory)

.PARAMETER Coverage
    Generate coverage report

.PARAMETER Verbose
    Run with verbose output

.PARAMETER Fast
    Run tests in parallel (faster)

.PARAMETER FailFast
    Stop on first failure

.PARAMETER Markers
    Run tests with specific markers (e.g., "unit", "integration")

.EXAMPLE
    .\scripts\run_tests.ps1
    .\scripts\run_tests.ps1 -Coverage
    .\scripts\run_tests.ps1 -Path tests/test_formatter.py
    .\scripts\run_tests.ps1 -Fast -FailFast
    .\scripts\run_tests.ps1 -Markers "unit"
#>

param(
    [string]$Path = "tests",
    [switch]$Coverage,
    [switch]$Verbose,
    [switch]$Fast,
    [switch]$FailFast,
    [string]$Markers
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "=== Running Tests ==="
Write-Info ""

# Check if pytest is installed
try {
    $null = python -m pytest --version 2>&1
} catch {
    Write-Error "✗ pytest not found. Please run setup.ps1 first."
    exit 1
}

# Build pytest command
$pytestArgs = @($Path)

# Add verbosity
if ($Verbose) {
    $pytestArgs += "-v"
} else {
    $pytestArgs += "-q"
}

# Add coverage
if ($Coverage) {
    Write-Info "Generating coverage report..."
    $pytestArgs += "--cov=code_agent"
    $pytestArgs += "--cov=tools"
    $pytestArgs += "--cov=utils"
    $pytestArgs += "--cov-report=term"
    $pytestArgs += "--cov-report=html"
    $pytestArgs += "--cov-report=xml"
}

# Add parallel execution
if ($Fast) {
    Write-Info "Running tests in parallel..."
    $pytestArgs += "-n"
    $pytestArgs += "auto"
}

# Add fail fast
if ($FailFast) {
    $pytestArgs += "--maxfail=1"
}

# Add markers
if ($Markers) {
    Write-Info "Running tests with markers: $Markers"
    $pytestArgs += "-m"
    $pytestArgs += $Markers
}

# Add standard options
$pytestArgs += "--tb=short"
$pytestArgs += "--strict-markers"

# Run tests
Write-Info "Running: pytest $($pytestArgs -join ' ')"
Write-Info ""

try {
    $startTime = Get-Date
    python -m pytest @pytestArgs
    $exitCode = $LASTEXITCODE
    $endTime = Get-Date
    $duration = $endTime - $startTime

    Write-Info ""
    Write-Info "=== Test Results ==="

    if ($exitCode -eq 0) {
        Write-Success "✓ All tests passed!"
    } elseif ($exitCode -eq 1) {
        Write-Error "✗ Some tests failed"
    } elseif ($exitCode -eq 2) {
        Write-Error "✗ Test execution interrupted"
    } elseif ($exitCode -eq 3) {
        Write-Error "✗ Internal error"
    } elseif ($exitCode -eq 4) {
        Write-Error "✗ pytest command line usage error"
    } elseif ($exitCode -eq 5) {
        Write-Warning "! No tests collected"
    }

    Write-Info "Duration: $($duration.TotalSeconds.ToString('F2')) seconds"

    if ($Coverage) {
        Write-Info ""
        Write-Success "✓ Coverage report generated:"
        Write-Info "  - HTML: htmlcov/index.html"
        Write-Info "  - XML: coverage.xml"

        if (Test-Path "htmlcov/index.html") {
            $openReport = Read-Host "Open HTML coverage report? (y/N)"
            if ($openReport -eq 'y' -or $openReport -eq 'Y') {
                Start-Process "htmlcov/index.html"
            }
        }
    }

    exit $exitCode

} catch {
    Write-Error "✗ Error running tests: $_"
    exit 1
}
