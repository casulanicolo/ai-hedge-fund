# scripts/send_premarket.ps1
# Cron: Mon-Fri 08:30 ET (12:30 UTC)
# 1. Runs the pipeline (signals only, no execution, no email)
# 2. Builds pre-market HTML
# 3. Sends via send_premarket()

param(
    [string[]]$Tickers = @(),
    [switch]$DryRun
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$date = (Get-Date).ToString("yyyy-MM-dd")
Write-Host "[$date] Pre-market briefing started" -ForegroundColor Cyan

# Step 1: Run pipeline (no execute, no email — just generate fresh signals + SQLite)
$ticker_args = if ($Tickers.Count -gt 0) { $Tickers } else { @() }
Write-Host "Running pipeline (signal-only)..."
python -m src.run_pipeline @ticker_args --mode full
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline failed — sending critical alert" -ForegroundColor Red
    python -c "
from src.alerts.email_sender import send_critical_alert
send_critical_alert('Pipeline failed at pre-market', 'run_pipeline exited non-zero at $date 08:30 ET')
"
    exit 1
}

# Step 2 + 3: Build HTML and send
if ($DryRun) {
    Write-Host "DRY RUN — building HTML without sending"
    python -m src.alerts.premarket_builder --preview --out "logs/premarket_preview_$date.html"
    Write-Host "Preview: logs/premarket_preview_$date.html" -ForegroundColor Green
} else {
    python -c "
import sys, pathlib
from src.alerts.premarket_builder import build_context, build_html, build_text
from src.alerts.email_sender import send_premarket

ctx  = build_context()
html = build_html(ctx)
text = build_text(ctx)
ok   = send_premarket(html, text, date=ctx['date'])
if ok:
    print('[OK] Pre-market email sent')
else:
    print('[WARN] Pre-market email failed — check SMTP config', file=sys.stderr)
    sys.exit(1)
"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pre-market email send failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[$date] Pre-market briefing complete" -ForegroundColor Green
