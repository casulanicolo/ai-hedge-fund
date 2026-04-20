# scripts/run_backtest_fullhistory.ps1
# Full-history forward-only backtest (2019 → today).
# Long-running (overnight) — invokes walk-forward harness so IS + OOS run sequentially.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/run_backtest_fullhistory.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/run_backtest_fullhistory.ps1 -Tickers "AAPL,MSFT,NVDA"
#   powershell -ExecutionPolicy Bypass -File scripts/run_backtest_fullhistory.ps1 -NoCache

param(
    [string]$Tickers   = "all",
    [double]$Capital   = 100000.0,
    [string]$IsStart   = "2019-01-02",
    [string]$IsEnd     = "2022-12-30",
    [string]$OosStart  = "2023-01-02",
    [string]$OosEnd    = (Get-Date).ToString("yyyy-MM-dd"),
    [string]$Label     = "fullhistory",
    [switch]$NoCache
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$ts = (Get-Date).ToString("yyyy-MM-dd_HHmmss")
$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir "backtest_fullhistory_$ts.log"

Write-Host "[$ts] Forward-only walk-forward backtest" -ForegroundColor Cyan
Write-Host "  Tickers : $Tickers"
Write-Host "  Capital : `$$Capital"
Write-Host "  IS      : $IsStart -> $IsEnd"
Write-Host "  OOS     : $OosStart -> $OosEnd"
Write-Host "  Log     : $logFile"
Write-Host ""

$cacheArg = if ($NoCache) { "--no-cache" } else { "" }

$args = @(
    "-m", "src.backtesting.cli",
    "--walk-forward",
    "--tickers",   $Tickers,
    "--capital",   $Capital,
    "--is-start",  $IsStart,
    "--is-end",    $IsEnd,
    "--oos-start", $OosStart,
    "--oos-end",   $OosEnd,
    "--label",     $Label
)
if ($NoCache) { $args += "--no-cache" }

python @args 2>&1 | Tee-Object -FilePath $logFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "Backtest failed (exit $LASTEXITCODE) — see $logFile" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Backtest complete" -ForegroundColor Green
Write-Host "Log: $logFile"
Write-Host "Results: results/forward_${Label}_IS_*.json + forward_${Label}_OOS_*.json"
