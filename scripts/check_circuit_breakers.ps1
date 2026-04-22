# scripts/check_circuit_breakers.ps1 — Fase 8
# Run all CB checks and log to logs/circuit_breakers.log
# Schedule every 5 minutes during market hours (09:25–16:05 ET)
#
# Task Scheduler example (every 5 min, Mon-Fri):
#   Trigger: Daily 09:25 ET, repeat every 5 min for 6h45min
#   Action : powershell.exe -File "C:\...\scripts\check_circuit_breakers.ps1"

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-Location $ProjectRoot

$LogDir  = Join-Path $ProjectRoot "logs"
$LogFile = Join-Path $LogDir "circuit_breakers.log"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Output "[$Timestamp] Running CB checks..." | Tee-Object -FilePath $LogFile -Append

$Result = python -m src.risk._cb_runner 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Output "[$Timestamp] CB check FAILED (exit $LASTEXITCODE)" | Tee-Object -FilePath $LogFile -Append
    Write-Output $Result | Tee-Object -FilePath $LogFile -Append
    exit 1
}

Write-Output $Result | Tee-Object -FilePath $LogFile -Append
Write-Output "[$Timestamp] CB checks complete." | Tee-Object -FilePath $LogFile -Append
