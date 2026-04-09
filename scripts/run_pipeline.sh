#!/bin/bash
# scripts/run_pipeline.sh
# Wrapper per la pipeline giornaliera di Athanor Alpha
# Accetta --mode full|light|review (default: full)
# Usa lock file separato per ciascun mode, così i tre slot non si bloccano.

# ── Argomenti ────────────────────────────────────────────────────────────────
MODE="full"
for arg in "$@"; do
    case $arg in
        --mode=*) MODE="${arg#--mode=}" ;;
        --mode)   shift; MODE="$1" ;;
    esac
done

# Valida il mode
if [[ "$MODE" != "full" && "$MODE" != "light" && "$MODE" != "review" ]]; then
    echo "Uso: $0 [--mode full|light|review]"
    exit 1
fi

# ── Percorsi ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCK="/tmp/athanor_pipeline_${MODE}.lock"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

# ── Lock atomico (separato per mode) ─────────────────────────────────────────
exec 200>"$LOCK"
if ! flock -n 200; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Pipeline ${MODE} gia' in esecuzione (lock occupato). Uscita." >> "$LOG_FILE"
    exit 0
fi

# ── Header log ───────────────────────────────────────────────────────────────
echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] Pipeline Athanor Alpha — mode=${MODE}" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] PID: $$" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Project dir: $PROJECT_DIR" >> "$LOG_FILE"

# ── Virtualenv ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Impossibile accedere a $PROJECT_DIR" >> "$LOG_FILE"
    exit 1
}
source .venv/bin/activate || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Impossibile attivare .venv" >> "$LOG_FILE"
    exit 1
}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Virtualenv attivato: $(which python)" >> "$LOG_FILE"

# ── Esecuzione pipeline ───────────────────────────────────────────────────────
python -m src.run_pipeline --mode "$MODE" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

# ── Footer log ───────────────────────────────────────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [OK] Pipeline ${MODE} completata con successo (exit: $EXIT_CODE)" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Pipeline ${MODE} terminata con errore (exit: $EXIT_CODE)" >> "$LOG_FILE"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [END] ========================================" >> "$LOG_FILE"

exit $EXIT_CODE
