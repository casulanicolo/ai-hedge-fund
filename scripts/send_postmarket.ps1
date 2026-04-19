# scripts/send_postmarket.ps1
# Cron: Mon-Fri 17:00 ET (21:00 UTC) — 1 hour after NYSE close
# Builds and sends EOD digest from today's SQLite data + Alpaca EOD positions.

param(
    [switch]$DryRun
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$date = (Get-Date).ToString("yyyy-MM-dd")
Write-Host "[$date] Post-market digest started" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "DRY RUN — building HTML without sending"
    python -m src.alerts.postmarket_builder --preview --out "logs/postmarket_preview_$date.html"
    Write-Host "Preview: logs/postmarket_preview_$date.html" -ForegroundColor Green
} else {
    python -c "
import sys
from src.alerts.postmarket_builder import build_context, build_html, build_text
from src.alerts.email_sender import send_postmarket

ctx  = build_context()
html = build_html(ctx)
text = build_text(ctx)

# Summary line from context for email subject
s = ctx['summary']
summary_line = f\"{s['pl_str']} | {s['fills']} FILL / {s['exits']} EXIT / {s['rejects']} REJECT\"

ok = send_postmarket(html, text, date=ctx['date'], summary_line=summary_line)
if ok:
    print('[OK] Post-market digest sent')
else:
    print('[WARN] Post-market digest failed — check SMTP config', file=sys.stderr)
    sys.exit(1)
"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Post-market email send failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[$date] Post-market digest complete" -ForegroundColor Green
