# scripts/start_monitor.ps1
# Local dev wrapper — starts the monitor daemon in the foreground.
# Usage:
#   .\scripts\start_monitor.ps1
#   .\scripts\start_monitor.ps1 --now                 # skip market-hours check
#   .\scripts\start_monitor.ps1 --llm-monitor         # enable LLM review layer
#   .\scripts\start_monitor.ps1 --cycle 60            # 60-second cycle

param(
    [switch]$now,
    [switch]$llmMonitor,
    [int]$cycle = 0
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$args = @()
if ($now)       { $args += "--now" }
if ($llmMonitor){ $args += "--llm-monitor" }
if ($cycle -gt 0){ $args += "--cycle"; $args += $cycle }

Write-Host "Starting Athanor monitor daemon..." -ForegroundColor Cyan
Write-Host "Args: $args" -ForegroundColor DarkGray
Write-Host "Kill switch: create .kill_monitor in project root to stop gracefully." -ForegroundColor Yellow
Write-Host ""

python -m src.monitor.daemon @args
