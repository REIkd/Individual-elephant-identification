#!/usr/bin/env bash
# 服务器代码 scp 上传（Git Bash / Linux / macOS）
# 用法: bash deploy_pack/upload_to_server.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SERVER="${ELEPHANT_SERVER_SSH:-root@120.196.88.140}"
PORT="${ELEPHANT_SSH_PORT:-12222}"
REMOTE="${ELEPHANT_REMOTE_DIR:-/root/elephant_cloud}"

files=(
  cloud_server.py
  cloud_inference.py
  cloud_render.py
  elephant_clip_recorder.py
  video_tracker_yolo.py
  predict.py
  classifier.py
  elephant_net.py
  paths.py
  allowed_elephants.json
  class_names.json
  cloud_clip_env.sh
  start_cloud_server_linux.sh
  start_cloud_server_cpu.sh
  start_cloud_server_low_vram.sh
  uovision_camera.py
  uovision_registry.py
  uovision_open_api.py
  uovision_video_pipeline.py
  uovision_cameras.json
  uovision_config.example.sh
  setup_uovision_ir.sh
)

echo "=== 上传到 ${SERVER}:${REMOTE} (port ${PORT}) ==="
for f in "${files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  跳过: $f"
    continue
  fi
  echo "  -> $f"
  scp -P "$PORT" "$f" "${SERVER}:${REMOTE}/"
done

echo ""
echo "完成。SSH 重启:"
echo "  ssh -p ${PORT} ${SERVER}"
echo "  cd ${REMOTE} && bash start_cloud_server_linux.sh"
