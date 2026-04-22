# scripts/health_check.ps1
# Full system health check — exits 0 if healthy, 1 if any critical check fails.
# Called by deploy.ps1 and can be run manually at any time.
#
# Usage:
#   .\scripts\health_check.ps1                          # local check
#   .\scripts\health_check.ps1 -VpsHost athanor-vps    # remote SSH check

param(
    [string]$VpsHost    = "",         # empty = run locally
    [string]$VpsUser    = "athanor",
    [string]$VpsPath    = "/home/athanor/athanor-alpha",
    [string]$DashUrl    = "http://localhost/ping",
    [switch]$Json       = $false
)

$checks = @()
$allOk  = $true

function Check($name, $ok, $detail = "") {
    $status = if ($ok) { "OK" } else { "FAIL" }
    $checks += [PSCustomObject]@{check = $name; status = $status; detail = $detail}
    if (-not $ok) { $script:allOk = $false }
    $color = if ($ok) { "Green" } else { "Red" }
    if (-not $Json) {
        Write-Host "[$status] $name$(if ($detail) {" — $detail"})" -ForegroundColor $color
    }
}

function RunCmd($cmd) {
    if ($VpsHost) {
        return ssh "${VpsUser}@${VpsHost}" $cmd 2>&1
    } else {
        return Invoke-Expression $cmd 2>&1
    }
}

Write-Host "=== Athanor Alpha Health Check — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan

# ── Pipeline health-check JSON ─────────────────────────────────────────────────
$hcRaw = RunCmd "cd $VpsPath && .venv/bin/python -m src.run_pipeline --health-check 2>/dev/null"
try {
    $hc = $hcRaw | ConvertFrom-Json -ErrorAction Stop
    Check "Pipeline DB connection" ($hc.db_ok -eq $true) "predictions=$($hc.predictions_total)"
    Check "Pipeline predictions today" ($hc.predictions_today -gt 0) "count=$($hc.predictions_today)"
    Check "Pipeline runs 24h" ($hc.pipeline_runs_24h -ge 1) "count=$($hc.pipeline_runs_24h)"
} catch {
    Check "Pipeline health-check" $false "JSON parse failed: $hcRaw"
}

# ── Circuit breakers ────────────────────────────────────────────────────────────
$cbOut = RunCmd "cd $VpsPath && .venv/bin/python -m src.risk._cb_runner 2>/dev/null"
$cbHardTriggered = ($LASTEXITCODE -eq 2)
Check "Circuit breakers" (-not $cbHardTriggered) (if ($cbHardTriggered) {"CRITICAL CB triggered"} else {"all clear"})

# ── Kill switch ─────────────────────────────────────────────────────────────────
$ksArmed = RunCmd "test -f $VpsPath/.athanor_kill && echo armed || echo disarmed" 2>&1
Check "Kill switch" ($ksArmed -eq "disarmed") $ksArmed

# ── Dashboard HTTP ──────────────────────────────────────────────────────────────
try {
    $resp = Invoke-WebRequest -Uri $DashUrl -TimeoutSec 10 -ErrorAction Stop
    Check "Dashboard HTTP" ($resp.StatusCode -eq 200) "HTTP $($resp.StatusCode)"
} catch {
    Check "Dashboard HTTP" $false $_.Exception.Message
}

# ── Services (VPS only) ─────────────────────────────────────────────────────────
if ($VpsHost) {
    $dashActive = RunCmd "systemctl is-active athanor-dashboard"
    $monActive  = RunCmd "systemctl is-active athanor-monitor"
    Check "Service: athanor-dashboard" ($dashActive -eq "active") $dashActive
    Check "Service: athanor-monitor"   ($monActive  -eq "active") $monActive
}

# ── Disk space ──────────────────────────────────────────────────────────────────
$diskFreeGB = RunCmd "df -BG $VpsPath | awk 'NR==2{gsub(/G/,\"\",$4); print $4}'" 2>&1
if ($diskFreeGB -match '^\d+$') {
    Check "Disk free space" ([int]$diskFreeGB -ge 2) "${diskFreeGB}GB free"
} else {
    $localFree = [math]::Round((Get-PSDrive (Split-Path (Get-Location) -Qualifier).TrimEnd(':') -ErrorAction SilentlyContinue).Free / 1GB, 1)
    Check "Disk free space" ($localFree -ge 2) "${localFree}GB free (local)"
}

# ── Summary ─────────────────────────────────────────────────────────────────────
Write-Host ""
if ($Json) {
    $checks | ConvertTo-Json -Compress | Write-Host
} else {
    $failed = $checks | Where-Object { $_.status -eq "FAIL" }
    if ($allOk) {
        Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
    } else {
        Write-Host "=== $($failed.Count) CHECK(S) FAILED ===" -ForegroundColor Red
        $failed | ForEach-Object { Write-Host "  FAIL: $($_.check) — $($_.detail)" -ForegroundColor Red }
    }
}

exit $(if ($allOk) { 0 } else { 1 })
