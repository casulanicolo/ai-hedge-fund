# train_nightly.ps1 — Fase 7
# Nightly cron wrapper for meta-learner training.
# Scheduled via Windows Task Scheduler at 02:00 ET.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\train_nightly.ps1
#
# Logs are appended to logs\train_nightly.log

param(
    [string]$ProjectRoot = $PSScriptRoot + "\.."
)

$LogDir  = Join-Path $ProjectRoot "logs"
$LogFile = Join-Path $LogDir "train_nightly.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $LogFile "[$Timestamp] === train_nightly.ps1 START ==="

Push-Location $ProjectRoot
try {
    python -m src.ml.train_meta_learner 2>&1 | Tee-Object -Append -FilePath $LogFile
    $ExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
if ($ExitCode -eq 0) {
    Add-Content $LogFile "[$Timestamp] === train_nightly.ps1 DONE (exit 0) ==="
} else {
    Add-Content $LogFile "[$Timestamp] === train_nightly.ps1 FAILED (exit $ExitCode) ==="
}

exit $ExitCode
