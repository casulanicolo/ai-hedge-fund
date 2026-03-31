#!/bin/bash
# scripts/run_pipeline.sh
# Wrapper per la pipeline giornaliera di Athanor Alpha
# Attiva il virtualenv, lancia la pipeline, scrive il log, gestisce il lock file (flock atomico)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCK="/tmp/athanor_pipeline.lock"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y-%m-%d).log"

# Crea la cartella logs se non esiste
mkdir -p "$LOG_DIR"

# Apri il file di lock (fd 200) e prova ad acquisire il lock in modo atomico
exec 200>"$LOCK"
if ! flock -n 200; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Pipeline gia' in esecuzione (lock occupato). Uscita." >> "$LOG_FILE"
    exit 0
fi

# Intestazione nel log
echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] Pipeline Athanor Alpha" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] PID: $$" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Project dir: $PROJECT_DIR" >> "$LOG_FILE"

# Attiva il virtualenv
cd "$PROJECT_DIR" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Impossibile accedere a $PROJECT_DIR" >> "$LOG_FILE"
    exit 1
}

source .venv/bin/activate || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Impossibile attivare .venv" >> "$LOG_FILE"
    exit 1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Virtualenv attivato: $(which python)" >> "$LOG_FILE"

# Lancia la pipeline e redirige stdout+stderr nel log
python -m src.run_pipeline >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

# Log del risultato
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [OK] Pipeline completata con successo (exit: $EXIT_CODE)" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Pipeline terminata con errore (exit: $EXIT_CODE)" >> "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [END] ========================================" >> "$LOG_FILE"

# Il lock viene rilasciato automaticamente alla chiusura del processo (fd 200)
exit $EXIT_CODE
