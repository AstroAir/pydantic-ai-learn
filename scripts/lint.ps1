#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run code quality checks for pydantic-ai-learn project

.DESCRIPTION
    This script runs various code quality tools:
    - Ruff (linting)
    - MyPy (type checking)
    - Optionally: Bandit (security), pip-audit (dependencies)

.PARAMETER Tool
    Specific tool to run (ruff, mypy, bandit, pip-audit, all)

.PARAMETER Fix
    Auto-fix issues where possible (ruff only)

.PARAMETER Strict
    Run in strict mode with all checks

.EXAMPLE
    .\scripts\lint.ps1
    .\scripts\lint.ps1 -Tool ruff
    .\scripts\lint.ps1 -Fix
    .\scripts\lint.ps1 -Strict
#>

param(
    [ValidateSet("ruff", "mypy", "bandit", "pip-audit", "all")]
    [string]$Tool = "all",
    [switch]$Fix,
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

Write-Info "=== Code Quality Checks ==="
Write-Info ""

$allPassed = $true

# Ruff linting
function Run-Ruff {
    Write-Info "Running Ruff linting..."

    try {
        $null = python -m ruff --version 2>&1
    } catch {
        Write-Warning "! Ruff not installed, skipping"
        return $true
    }

    try {
        if ($Fix) {
            Write-Info "Auto-fixing issues..."
            python -m ruff check . --fix
            $exitCode = $LASTEXITCODE
        } else {
            python -m ruff check .
            $exitCode = $LASTEXITCODE
        }

        if ($exitCode -eq 0) {
            Write-Success "✓ Ruff: No issues found"
            return $true
        } else {
            Write-Error "✗ Ruff: Issues found"
            return $false
        }
    } catch {
        Write-Error "✗ Ruff: Error running check"
        return $false
    }
}

# MyPy type checking
function Run-MyPy {
    Write-Info "Running MyPy type checking..."

    try {
        $null = python -m mypy --version 2>&1
    } catch {
        Write-Warning "! MyPy not installed, skipping"
        return $true
    }

    try {
        $myPyArgs = @("code_agent", "tools", "utils")

        if ($Strict) {
            $myPyArgs += "--strict"
        }

        $myPyArgs += "--show-error-codes"
        $myPyArgs += "--pretty"

        python -m mypy @myPyArgs
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Success "✓ MyPy: No type errors"
            return $true
        } else {
            Write-Warning "! MyPy: Type errors found"
            return $false
        }
    } catch {
        Write-Error "✗ MyPy: Error running check"
        return $false
    }
}

# Bandit security scan
function Run-Bandit {
    Write-Info "Running Bandit security scan..."

    try {
        $null = python -m bandit --version 2>&1
    } catch {
        Write-Warning "! Bandit not installed, skipping"
        return $true
    }

    try {
        python -m bandit -r code_agent tools utils -f screen
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Success "✓ Bandit: No security issues"
            return $true
        } else {
            Write-Warning "! Bandit: Security issues found"
            return $false
        }
    } catch {
        Write-Error "✗ Bandit: Error running scan"
        return $false
    }
}

# pip-audit dependency check
function Run-PipAudit {
    Write-Info "Running pip-audit dependency check..."

    try {
        $null = python -m pip_audit --version 2>&1
    } catch {
        Write-Warning "! pip-audit not installed, skipping"
        return $true
    }

    try {
        python -m pip_audit --requirement requirements.txt --requirement requirements-dev.txt
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Success "✓ pip-audit: No vulnerable dependencies"
            return $true
        } else {
            Write-Warning "! pip-audit: Vulnerable dependencies found"
            return $false
        }
    } catch {
        Write-Error "✗ pip-audit: Error running check"
        return $false
    }
}

# Run selected tools
switch ($Tool) {
    "ruff" {
        $allPassed = Run-Ruff
    }
    "mypy" {
        $allPassed = Run-MyPy
    }
    "bandit" {
        $allPassed = Run-Bandit
    }
    "pip-audit" {
        $allPassed = Run-PipAudit
    }
    "all" {
        $ruffPassed = Run-Ruff
        Write-Info ""

        $mypyPassed = Run-MyPy
        Write-Info ""

        if ($Strict) {
            $banditPassed = Run-Bandit
            Write-Info ""

            $auditPassed = Run-PipAudit
            Write-Info ""

            $allPassed = $ruffPassed -and $mypyPassed -and $banditPassed -and $auditPassed
        } else {
            $allPassed = $ruffPassed -and $mypyPassed
        }
    }
}

# Summary
Write-Info "=== Summary ==="
if ($allPassed) {
    Write-Success "✓ All checks passed!"
    exit 0
} else {
    Write-Error "✗ Some checks failed"
    Write-Info ""
    Write-Info "Tips:"
    Write-Info "  - Run with -Fix to auto-fix Ruff issues"
    Write-Info "  - Check output above for specific issues"
    Write-Info "  - See CONTRIBUTING.md for coding standards"
    exit 1
}
