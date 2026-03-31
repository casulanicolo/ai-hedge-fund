#!/bin/bash
# scripts/run_monitor.sh
# Wrapper per il monitor daemon di Athanor Alpha
# Attiva il virtualenv, lancia il daemon, scrive il log, gestisce il lock file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCK="/tmp/athanor_monitor.lock"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/monitor_$(date +%Y-%m-%d).log"

# Crea la cartella logs se non esiste
mkdir -p "$LOG_DIR"

# Controlla lock file: se esiste, il monitor e' gia' in esecuzione
if [ -f "$LOCK" ]; then
    # Controlla se il PID nel lock e' ancora attivo
    LOCK_PID=$(cat "$LOCK" 2>/dev/null)
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Monitor gia' in esecuzione (PID: $LOCK_PID). Uscita." >> "$LOG_FILE"
        exit 0
    else
        # Lock stale (processo morto): rimuovilo e continua
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] Lock stale trovato (PID: $LOCK_PID non esiste). Rimozione e riavvio." >> "$LOG_FILE"
        rm -f "$LOCK"
    fi
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] Monitor Daemon Athanor Alpha" >> "$LOG_FILE"
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

# Lancia il monitor daemon e redirige stdout+stderr nel log
python -m src.monitor.daemon >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [OK] Monitor terminato normalmente (exit: $EXIT_CODE)" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Monitor terminato con errore (exit: $EXIT_CODE)" >> "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [END] ========================================" >> "$LOG_FILE"

exit $EXIT_CODE
