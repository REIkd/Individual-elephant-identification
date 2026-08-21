#!/bin/bash
# Pi: connect to cloud API, capture camera, show results locally
#
# First time:
#   chmod +x run_pi_cloud_client.sh
#   cp pi_cloud_client_config.example.sh pi_cloud_config.sh
#   nano pi_cloud_config.sh
#   ./run_pi_cloud_client.sh

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="./pi_cloud_config.sh"
if [[ -f "$CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$CONFIG"
else
  echo "Missing pi_cloud_config.sh"
  echo "Run: cp pi_cloud_client_config.example.sh pi_cloud_config.sh"
  exit 1
fi

: "${ELEPHANT_SERVER:?Set ELEPHANT_SERVER in pi_cloud_config.sh}"
: "${CLOUD_API_KEY:?Set CLOUD_API_KEY in pi_cloud_config.sh}"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run:"
  echo "  python3 -m venv .venv && source .venv/bin/activate"
  echo "  pip install -r requirements-pi-client.txt"
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "Server: $ELEPHANT_SERVER"
echo "Testing health (max 5s)..."
if ! curl -sf --connect-timeout 3 --max-time 5 "${ELEPHANT_SERVER%/}/health" >/dev/null; then
  echo "[FAIL] Cannot reach ${ELEPHANT_SERVER}/health"
  exit 1
fi

echo "Cloud OK. Starting camera client..."
echo "Press q to quit, r to reset session"

STREAM_ARGS=()
if [[ -n "${STREAM_ID:-}" ]]; then
  STREAM_ARGS=(--stream-id "$STREAM_ID")
fi

CAMERA="${CAMERA_DEVICE:-auto}"
UPLOAD_W="${UPLOAD_WIDTH:-1280}"
JPEG_Q="${JPEG_QUALITY:-88}"
SEND_IV="${SEND_INTERVAL:-0.08}"
CAM_W="${CAMERA_WIDTH:-1280}"
CAM_H="${CAMERA_HEIGHT:-720}"

exec python pi_cloud_client.py \
  --server "$ELEPHANT_SERVER" \
  --api-key "$CLOUD_API_KEY" \
  "${STREAM_ARGS[@]}" \
  --camera "$CAMERA" \
  --camera-width "$CAM_W" \
  --camera-height "$CAM_H" \
  --upload-width "$UPLOAD_W" \
  --send-interval "$SEND_IV" \
  --jpeg-quality "$JPEG_Q"
