#!/bin/bash
# Pi Cloud Client watchdog — restart main service if it stops.
set -euo pipefail

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

SERVICE="pi-cloud-client"
INTERVAL_SEC=10

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/watchdog.log"
mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

log "watchdog started, interval ${INTERVAL_SEC}s"

while true; do
    if ! systemctl --user is-active --quiet "$SERVICE"; then
        log "$SERVICE not running, restarting..."
        systemctl --user reset-failed "$SERVICE" 2>/dev/null || true
        systemctl --user start "$SERVICE" 2>/dev/null || true
    fi
    sleep "$INTERVAL_SEC"
done
