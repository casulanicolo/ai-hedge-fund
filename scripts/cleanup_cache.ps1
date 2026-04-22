# scripts/cleanup_cache.ps1
# Remove cache files older than RetainDays (default: 30).
# Safe to run while system is live — only deletes from cache/ directory.

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]   $RetainDays  = 30,
    [switch]$DryRun      = $false
)

$CacheDir = Join-Path $ProjectRoot "cache"
$Cutoff   = (Get-Date).AddDays(-$RetainDays)
$Removed  = 0
$BytesSaved = 0

if (-not (Test-Path $CacheDir)) {
    Write-Host "No cache directory found at $CacheDir — nothing to do."
    exit 0
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Cleanup: removing files older than $RetainDays days from $CacheDir"
if ($DryRun) { Write-Host "  DRY RUN — no files will be deleted" }

Get-ChildItem $CacheDir -Recurse -File | Where-Object { $_.LastWriteTime -lt $Cutoff } | ForEach-Object {
    $BytesSaved += $_.Length
    $Removed++
    if ($DryRun) {
        Write-Host "  [DRY] Would delete: $($_.FullName)"
    } else {
        Remove-Item $_.FullName -Force
    }
}

$MB = [math]::Round($BytesSaved / 1MB, 2)
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Removed file(s) $(if ($DryRun) {'would be'} else {'deleted'}), ${MB} MB freed."
