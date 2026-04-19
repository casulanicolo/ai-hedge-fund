#!/usr/bin/env bash
# scripts/install_monitor.sh
# Install Athanor monitor as a systemd timer+service on a Linux VPS.
# Run as root (or with sudo) after deploying the project to /opt/athanor-alpha.
#
# Usage: sudo bash scripts/install_monitor.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SYSTEMD_DIR="/etc/systemd/system"

echo "[install_monitor] Project dir: $PROJECT_DIR"

# ── Copy unit files ──────────────────────────────────────────────────────────
cp "$SCRIPT_DIR/athanor-monitor.service" "$SYSTEMD_DIR/"
cp "$SCRIPT_DIR/athanor-monitor.timer"   "$SYSTEMD_DIR/"

# ── Patch WorkingDirectory in service file (in case path differs) ────────────
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" \
    "$SYSTEMD_DIR/athanor-monitor.service"
sed -i "s|ExecStart=.*|ExecStart=$PROJECT_DIR/.venv/bin/python -m src.monitor.daemon|" \
    "$SYSTEMD_DIR/athanor-monitor.service"
sed -i "s|EnvironmentFile=.*|EnvironmentFile=$PROJECT_DIR/.env|" \
    "$SYSTEMD_DIR/athanor-monitor.service"

# ── Reload systemd and enable timer ─────────────────────────────────────────
systemctl daemon-reload
systemctl enable athanor-monitor.timer
systemctl start  athanor-monitor.timer

echo "[install_monitor] Timer enabled. Next fires:"
systemctl list-timers athanor-monitor.timer --no-pager

echo ""
echo "Useful commands:"
echo "  systemctl status  athanor-monitor.service"
echo "  journalctl -u athanor-monitor.service -f"
echo "  systemctl stop athanor-monitor.timer   # disable auto-start"
echo "  touch $PROJECT_DIR/.kill_monitor        # graceful stop mid-session"
