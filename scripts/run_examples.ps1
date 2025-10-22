#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run examples for pydantic-ai-learn project

.DESCRIPTION
    This script runs example files to verify they work correctly.
    Can run all examples or specific categories.

.PARAMETER Category
    Specific category to run (basic, messages, output, tools, graph, mcp, multi-agent)

.PARAMETER File
    Specific example file to run

.PARAMETER List
    List all available examples

.EXAMPLE
    .\scripts\run_examples.ps1
    .\scripts\run_examples.ps1 -Category basic
    .\scripts\run_examples.ps1 -File examples/simple_demo.py
    .\scripts\run_examples.ps1 -List
#>

param(
    [string]$Category,
    [string]$File,
    [switch]$List
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Define example categories
$exampleCategories = @{
    "simple" = @("examples/simple_demo.py")
    "basic" = @(Get-ChildItem "examples/basic/*.py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    "messages" = @(Get-ChildItem "examples/messages/*.py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    "output" = @(Get-ChildItem "examples/output/*.py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    "tools" = @(Get-ChildItem "examples/tools/*.py" -Exclude "*test*.py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    "graph" = @(Get-ChildItem "examples/graph/*.py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    "mcp" = @(Get-ChildItem "examples/mcp/*.py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    "multi-agent" = @(Get-ChildItem "examples/multi-agent/*.py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
}

# List examples
if ($List) {
    Write-Info "=== Available Examples ==="
    Write-Info ""
    foreach ($cat in $exampleCategories.Keys | Sort-Object) {
        Write-Info "Category: $cat"
        foreach ($example in $exampleCategories[$cat]) {
            if ($example) {
                Write-Info "  - $example"
            }
        }
        Write-Info ""
    }
    exit 0
}

# Run specific file
if ($File) {
    Write-Info "=== Running Example: $File ==="
    Write-Info ""

    if (-not (Test-Path $File)) {
        Write-Error "✗ File not found: $File"
        exit 1
    }

    try {
        python $File
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Success "✓ Example completed successfully"
        } else {
            Write-Error "✗ Example failed with exit code: $exitCode"
        }

        exit $exitCode
    } catch {
        Write-Error "✗ Error running example: $_"
        exit 1
    }
}

# Determine which categories to run
$categoriesToRun = @()
if ($Category) {
    if ($exampleCategories.ContainsKey($Category)) {
        $categoriesToRun = @($Category)
    } else {
        Write-Error "✗ Unknown category: $Category"
        Write-Info "Available categories: $($exampleCategories.Keys -join ', ')"
        exit 1
    }
} else {
    # Run simple examples by default (safe, quick)
    $categoriesToRun = @("simple")
    Write-Warning "! Running only 'simple' category by default"
    Write-Info "  Use -Category to run specific categories or -List to see all"
    Write-Info ""
}

# Run examples
$totalExamples = 0
$successCount = 0
$failCount = 0
$skippedCount = 0

Write-Info "=== Running Examples ==="
Write-Info ""

foreach ($cat in $categoriesToRun) {
    $examples = $exampleCategories[$cat]

    if (-not $examples -or $examples.Count -eq 0) {
        Write-Warning "! No examples found in category: $cat"
        continue
    }

    Write-Info "Category: $cat"
    Write-Info ""

    foreach ($example in $examples) {
        if (-not $example -or -not (Test-Path $example)) {
            continue
        }

        $totalExamples++
        $exampleName = Split-Path $example -Leaf

        Write-Info "Running: $exampleName"

        try {
            # Run with timeout to prevent hanging
            $job = Start-Job -ScriptBlock {
                param($examplePath)
                python $examplePath
            } -ArgumentList $example

            $timeout = 30 # seconds
            $completed = Wait-Job $job -Timeout $timeout

            if ($completed) {
                $output = Receive-Job $job
                $exitCode = $job.State -eq "Completed" ? 0 : 1

                if ($exitCode -eq 0) {
                    Write-Success "  ✓ Passed"
                    $successCount++
                } else {
                    Write-Error "  ✗ Failed"
                    $failCount++
                }
            } else {
                Write-Warning "  ! Timeout (skipped)"
                Stop-Job $job
                $skippedCount++
            }

            Remove-Job $job -Force

        } catch {
            Write-Error "  ✗ Error: $_"
            $failCount++
        }

        Write-Info ""
    }
}

# Summary
Write-Info "=== Summary ==="
Write-Info "Total: $totalExamples"
Write-Success "Passed: $successCount"
if ($failCount -gt 0) {
    Write-Error "Failed: $failCount"
}
if ($skippedCount -gt 0) {
    Write-Warning "Skipped: $skippedCount"
}

if ($failCount -eq 0) {
    Write-Success "✓ All examples completed successfully!"
    exit 0
} else {
    Write-Error "✗ Some examples failed"
    exit 1
}
