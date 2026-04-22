# scripts/backup_db.ps1
# Backup SQLite DB + logs to local backup directory (or remote share).
# Usage (local Windows): .\scripts\backup_db.ps1
# Usage (VPS via cron): use scripts/backup_db.sh instead

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$BackupRoot  = (Join-Path (Split-Path -Parent $PSScriptRoot) "db\backups"),
    [int]   $RetainDays  = 30
)

$ErrorActionPreference = "Stop"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$DbPath    = Join-Path $ProjectRoot "db\hedge_fund.db"
$LogDir    = Join-Path $ProjectRoot "logs"

if (-not (Test-Path $BackupRoot)) {
    New-Item -ItemType Directory -Path $BackupRoot | Out-Null
}

# ── SQLite online backup (safe while DB is in use) ────────────────────────────
$BackupDb = Join-Path $BackupRoot "hedge_fund_${Timestamp}.db"
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Backing up DB → $BackupDb"

python -c "
import sqlite3, sys
src  = sqlite3.connect(r'$DbPath')
dst  = sqlite3.connect(r'$BackupDb')
src.backup(dst)
dst.close(); src.close()
size = __import__('os').path.getsize(r'$BackupDb')
print(f'Backup OK: {size/1024/1024:.1f} MB')
"
if ($LASTEXITCODE -ne 0) { throw "DB backup failed" }

# ── Compress logs ─────────────────────────────────────────────────────────────
$LogArchive = Join-Path $BackupRoot "logs_${Timestamp}.zip"
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Archiving logs → $LogArchive"
if (Test-Path $LogDir) {
    Compress-Archive -Path "$LogDir\*.log" -DestinationPath $LogArchive -ErrorAction SilentlyContinue
}

# ── Cleanup old backups ────────────────────────────────────────────────────────
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Removing backups older than $RetainDays days..."
Get-ChildItem $BackupRoot -File | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetainDays) } | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Host "  Removed: $($_.Name)"
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Backup complete."
