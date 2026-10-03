[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][ValidateSet("smoke-test","crawl","process","train","pipeline","status")][string]$Mode,
 [double]$MaxHours=8,[int]$MaxPages=1000,[double]$MaxDiskGB=20,[double]$MaxRamGB=6,
 [string]$SeedFile,[switch]$Resume,[switch]$DryRun
)
$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python 3.11+ was not found on PATH." }
if ($Mode -ne "status") {
 python -c "import yaml, crawl4ai, psutil" 2>$null
 if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing requirements..."
  python -m pip install crawl4ai pyyaml psutil
  if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
  python -m playwright install chromium
  if ($LASTEXITCODE -ne 0) { throw "Playwright Chromium install failed" }
 }
}
$argsList=@("data_pipeline/runner.py","--mode",$Mode)
if ($Mode -in @("crawl","pipeline")) {
 $argsList+=@("--max-hours","$MaxHours","--max-pages","$MaxPages","--max-disk-gb","$MaxDiskGB","--max-ram-gb","$MaxRamGB")
 if ($SeedFile) {$argsList+=@("--seed-file",$SeedFile)}
 if ($Resume) {$argsList+="--resume"}
 if ($DryRun) {$argsList+="--dry-run"}
}
python @argsList
if ($LASTEXITCODE -ne 0) { throw "Runner exited with code $LASTEXITCODE" }
