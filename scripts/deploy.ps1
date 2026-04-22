# scripts/deploy.ps1
# Athanor Alpha — One-click deploy: git push local + SSH pull + restart + health check
# Usage: .\scripts\deploy.ps1 [-VpsHost athanor-vps] [-Branch main]
#
# Prerequisites:
#   - SSH key configured for the VPS host
#   - VPS has git remote set to the same origin
#   - VPS user is 'athanor', project at /home/athanor/athanor-alpha

param(
    [string]$VpsHost  = "athanor-vps",
    [string]$VpsUser  = "athanor",
    [string]$VpsPath  = "/home/athanor/athanor-alpha",
    [string]$Branch   = "main",
    [switch]$SkipPush = $false
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

function Log($msg) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg"
}

function RunSsh($cmd) {
    $result = ssh "${VpsUser}@${VpsHost}" $cmd
    if ($LASTEXITCODE -ne 0) { throw "SSH command failed: $cmd" }
    return $result
}

Log "=== Athanor Alpha Deploy ==="
Log "Target: ${VpsUser}@${VpsHost}:${VpsPath} (branch: $Branch)"

# ── Step 1: Local git push ────────────────────────────────────────────────────
if (-not $SkipPush) {
    Log "Step 1/5: Pushing local commits to origin..."
    $status = git status --porcelain
    if ($status) {
        Write-Warning "Uncommitted changes detected. Commit or stash before deploying:"
        Write-Host $status
        throw "Dirty working tree — aborting deploy."
    }
    git push origin $Branch
    if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    Log "  Push OK"
} else {
    Log "Step 1/5: SKIP push (--SkipPush flag)"
}

# ── Step 2: VPS git pull ──────────────────────────────────────────────────────
Log "Step 2/5: Pulling on VPS..."
RunSsh "cd $VpsPath && git fetch origin && git checkout $Branch && git pull origin $Branch"
Log "  Pull OK"

# ── Step 3: Install dependencies if requirements changed ──────────────────────
Log "Step 3/5: pip install (incremental)..."
RunSsh "cd $VpsPath && .venv/bin/pip install -q -r requirements.txt"
Log "  Dependencies OK"

# ── Step 4: DB migrations (idempotent) ────────────────────────────────────────
Log "Step 4/5: DB schema migration..."
RunSsh "cd $VpsPath && .venv/bin/python -m src.db.init_db"
Log "  Schema OK"

# ── Step 5: Restart services ──────────────────────────────────────────────────
Log "Step 5/5: Restarting systemd services..."
RunSsh "sudo systemctl restart athanor-dashboard"
RunSsh "sudo systemctl restart athanor-monitor"
Start-Sleep -Seconds 5

# ── Health check ──────────────────────────────────────────────────────────────
Log "Running health check..."
$health = RunSsh "cd $VpsPath && .venv/bin/python -m src.run_pipeline --health-check 2>/dev/null"
Write-Host $health

$dashStatus = RunSsh "systemctl is-active athanor-dashboard"
$monStatus  = RunSsh "systemctl is-active athanor-monitor"

if ($dashStatus -ne "active") { Write-Warning "athanor-dashboard is NOT active: $dashStatus" }
if ($monStatus  -ne "active") { Write-Warning "athanor-monitor is NOT active: $monStatus" }

$elapsed = [math]::Round(((Get-Date) - $StartTime).TotalSeconds)
Log "=== Deploy complete in ${elapsed}s ==="
Log "Dashboard: http://${VpsHost}"
Log "Services : dashboard=$dashStatus | monitor=$monStatus"
