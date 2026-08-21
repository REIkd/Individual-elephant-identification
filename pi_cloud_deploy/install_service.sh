#!/bin/bash
# Install systemd user services (main client + watchdog)
# Usage: chmod +x install_service.sh && ./install_service.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="pi-cloud-client"
WATCHDOG_NAME="${SERVICE_NAME}-watchdog"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.service"
WATCHDOG_FILE="${SCRIPT_DIR}/${WATCHDOG_NAME}.service"
WATCHDOG_SCRIPT="${SCRIPT_DIR}/pi_cloud_watchdog.sh"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
OLD_CRON_MARKER="# pi-cloud-client-watchdog"

echo "============================================"
echo "  Pi Cloud Client — install services"
echo "============================================"
echo "Deploy dir: ${SCRIPT_DIR}"
echo ""

for f in "$SERVICE_FILE" "$WATCHDOG_FILE" "$WATCHDOG_SCRIPT" "${SCRIPT_DIR}/run_pi_cloud_client.sh" "${SCRIPT_DIR}/pi_cloud_config.sh"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Missing ${f}"
        exit 1
    fi
done

pkill -f "pi_cloud_watchdog.sh" 2>/dev/null || true
if crontab -l 2>/dev/null | grep -qF "$OLD_CRON_MARKER"; then
    (crontab -l 2>/dev/null || true) | grep -vF "$OLD_CRON_MARKER" | crontab -
    echo "[OK] removed old cron watchdog"
fi

chmod +x "${SCRIPT_DIR}/run_pi_cloud_client.sh"
chmod +x "$WATCHDOG_SCRIPT"
chmod +x "${SCRIPT_DIR}/install_service.sh" 2>/dev/null || true
echo "[OK] executable bits set"

mkdir -p "$SYSTEMD_USER_DIR"
sed "s|__DEPLOY_DIR__|${SCRIPT_DIR}|g" "$SERVICE_FILE" > "${SYSTEMD_USER_DIR}/${SERVICE_NAME}.service"
sed "s|__DEPLOY_DIR__|${SCRIPT_DIR}|g" "$WATCHDOG_FILE" > "${SYSTEMD_USER_DIR}/${WATCHDOG_NAME}.service"
echo "[OK] installed to ${SYSTEMD_USER_DIR}/"

systemctl --user daemon-reload
echo "[OK] daemon-reload"

loginctl enable-linger "$(whoami)" 2>/dev/null || loginctl enable-linger
echo "[OK] linger enabled"

systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
systemctl --user stop "$WATCHDOG_NAME" 2>/dev/null || true

systemctl --user enable "$SERVICE_NAME"
systemctl --user enable "$WATCHDOG_NAME"
systemctl --user start "$SERVICE_NAME"
systemctl --user start "$WATCHDOG_NAME"
echo "[OK] services started"

echo ""
echo "Status:  systemctl --user status ${SERVICE_NAME} ${WATCHDOG_NAME}"
echo "Logs:    journalctl --user -u ${SERVICE_NAME} -f"
echo "Probe:   cd ${SCRIPT_DIR} && source .venv/bin/activate && python pi_cloud_client.py --probe-camera"
