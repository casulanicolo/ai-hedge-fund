#!/bin/bash
# scripts/run_outcomes.sh
# Wrapper per l'outcome tracker notturno di Athanor Alpha
# Attiva il virtualenv, lancia il tracker, scrive il log, gestisce il lock file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCK="/tmp/athanor_outcomes.lock"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/outcomes_$(date +%Y-%m-%d).log"

# Crea la cartella logs se non esiste
mkdir -p "$LOG_DIR"

# Controlla lock file
if [ -f "$LOCK" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Outcome tracker gia' in esecuzione (lock: $LOCK). Uscita." >> "$LOG_FILE"
    exit 0
fi

# Crea il lock file con il PID corrente
echo $$ > "$LOCK"

# Funzione di cleanup
cleanup() {
    rm -f "$LOCK"
}
trap cleanup EXIT

# Intestazione nel log
echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] Outcome Tracker Athanor Alpha" >> "$LOG_FILE"
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

# Lancia l'outcome tracker e redirige stdout+stderr nel log
python -m src.feedback.outcome_tracker >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [OK] Outcome tracker completato con successo (exit: $EXIT_CODE)" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Outcome tracker terminato con errore (exit: $EXIT_CODE)" >> "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [END] ========================================" >> "$LOG_FILE"

exit $EXIT_CODE
