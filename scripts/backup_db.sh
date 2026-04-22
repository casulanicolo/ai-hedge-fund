#!/usr/bin/env bash
# scripts/backup_db.sh — Linux VPS version (used by crontab)
# Backs up SQLite DB to db/backups/ and prunes copies > 30 days.

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="$PROJ/db/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_SRC="$PROJ/db/hedge_fund.db"
DB_DST="$BACKUP_DIR/hedge_fund_${TIMESTAMP}.db"

mkdir -p "$BACKUP_DIR"

echo "[$(date '+%H:%M:%S')] Backing up DB → $DB_DST"
"$PROJ/.venv/bin/python" - <<EOF
import sqlite3, os
src = sqlite3.connect("$DB_SRC")
dst = sqlite3.connect("$DB_DST")
src.backup(dst)
dst.close(); src.close()
size = os.path.getsize("$DB_DST")
print(f"Backup OK: {size/1024/1024:.1f} MB")
EOF

# Gzip to save space
gzip "$DB_DST"
echo "[$(date '+%H:%M:%S')] Compressed → ${DB_DST}.gz"

# Prune copies older than 30 days
find "$BACKUP_DIR" -name "*.db.gz" -mtime +30 -delete
echo "[$(date '+%H:%M:%S')] Backup complete."
