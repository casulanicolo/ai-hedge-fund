# run_daily.ps1
# Athanor Alpha — Run giornaliero automatico
# Uso manuale: .\scripts\run_daily.ps1
# Uso con Task Scheduler: vedere istruzioni in fondo al file

$ProjectRoot = "$env:USERPROFILE\Documents\athanor-alpha"
$LogDir = "$ProjectRoot\logs"
$LogFile = "$LogDir\pipeline_$(Get-Date -Format 'yyyy-MM-dd').log"
$LockFile = "$env:TEMP\athanor_pipeline.lock"

# --- Crea cartella logs se non esiste ---
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# --- Lock: evita run doppi ---
if (Test-Path $LockFile) {
    $msg = "$(Get-Date -Format 'HH:mm:ss') | WARN | Pipeline gia in esecuzione (lock file presente). Uscita."
    Write-Host $msg
    Add-Content -Path $LogFile -Value $msg
    exit 1
}
New-Item -ItemType File -Path $LockFile | Out-Null

try {
    $start = Get-Date
    $msg = "$(Get-Date -Format 'HH:mm:ss') | INFO | === ATHANOR ALPHA RUN GIORNALIERO AVVIATO ==="
    Write-Host $msg
    Add-Content -Path $LogFile -Value $msg

    # --- Attiva virtualenv ed esegui pipeline ---
    $activate = "$ProjectRoot\.venv\Scripts\Activate.ps1"
    & $activate

    $msg = "$(Get-Date -Format 'HH:mm:ss') | INFO | Virtualenv attivato. Avvio pipeline..."
    Write-Host $msg
    Add-Content -Path $LogFile -Value $msg

    # Esegui la pipeline completa (tutti i ticker da config/tickers.yaml)
    $output = python -m src.run_pipeline 2>&1
    $exitCode = $LASTEXITCODE

    # Scrivi output nel log
    foreach ($line in $output) {
        Add-Content -Path $LogFile -Value $line
        Write-Host $line
    }

    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds)

    if ($exitCode -eq 0) {
        $msg = "$(Get-Date -Format 'HH:mm:ss') | INFO | Pipeline completata con successo in $($elapsed)s."
    } else {
        $msg = "$(Get-Date -Format 'HH:mm:ss') | ERROR | Pipeline fallita (exit code $exitCode) dopo $($elapsed)s."
    }
    Write-Host $msg
    Add-Content -Path $LogFile -Value $msg

} finally {
    # Rimuove sempre il lock file, anche in caso di errore
    if (Test-Path $LockFile) {
        Remove-Item $LockFile -Force
    }
}

# =============================================================================
# ISTRUZIONI TASK SCHEDULER (run automatico giornaliero)
# =============================================================================
# Per schedulare il run automatico alle 15:00 (orario italiano, mercati aperti):
#
# 1. Apri Task Scheduler (cerca "Utilità di pianificazione" nel menu Start)
# 2. Clic su "Crea attivita base..."
# 3. Nome: "Athanor Alpha Daily Run"
# 4. Trigger: Ogni giorno alle 15:00 (lun-ven)
# 5. Azione: Avvia programma
#    Programma: powershell.exe
#    Argomenti: -ExecutionPolicy Bypass -File "C:\Users\tomma\Documents\athanor-alpha\scripts\run_daily.ps1"
#    Directory: C:\Users\tomma\Documents\athanor-alpha
# 6. Spunta "Esegui solo se l'utente e connesso"
# 7. Salva
#
# Oppure via PowerShell (esegui come amministratore):
#
# $action = New-ScheduledTaskAction -Execute "powershell.exe" `
#     -Argument "-ExecutionPolicy Bypass -File `"$env:USERPROFILE\Documents\athanor-alpha\scripts\run_daily.ps1`"" `
#     -WorkingDirectory "$env:USERPROFILE\Documents\athanor-alpha"
# $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 15:00
# Register-ScheduledTask -TaskName "Athanor Alpha Daily Run" -Action $action -Trigger $trigger -RunLevel Highest
