[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("smoke-test","crawl","process","train","pipeline","status")]
    [string]$Mode,
    [double]$MaxHours = 8,
    [int]$MaxPages = 1000,
    [double]$MaxDiskGB = 20,
    [double]$MaxRamGB = 6,
    [string]$SeedFile,
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python 3.11+ was not found on PATH."
}

if ($Mode -ne "status") {
    # Windows PowerShell may turn native stderr into NativeCommandError.
    # Capture it without aborting, then explicitly inspect Python's exit code.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $dependencyOutput = & python -c "import yaml, crawl4ai, psutil" 2>&1
    $dependencyExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference

    if ($dependencyExitCode -ne 0) {
        Write-Host "Installing Python dependencies..."
        & python -m pip install crawl4ai pyyaml psutil
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed. Review the installation output above."
        }

        & python -m playwright install chromium
        if ($LASTEXITCODE -ne 0) {
            throw "Playwright Chromium install failed."
        }
    }
}

$argsList = @("-m", "data_pipeline.runner", "--mode", $Mode)

if ($Mode -in @("crawl", "pipeline")) {
    $argsList += @(
        "--max-hours", "$MaxHours",
        "--max-pages", "$MaxPages",
        "--max-disk-gb", "$MaxDiskGB",
        "--max-ram-gb", "$MaxRamGB"
    )

    if ($SeedFile) {
        $argsList += @("--seed-file", $SeedFile)
    }
    if ($Resume) {
        $argsList += "--resume"
    }
    if ($DryRun) {
        $argsList += "--dry-run"
    }
}

& python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Runner exited with code $LASTEXITCODE"
}
