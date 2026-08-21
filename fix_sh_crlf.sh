#!/usr/bin/env bash
# 在 Linux 服务器上修复 Windows CRLF 导致的 "command not found" ($'\r')
set -euo pipefail
cd "$(dirname "$0")"
for f in uovision_config.example.sh cloud_clip_env.sh start_cloud_server_linux.sh setup_uovision_ir.sh; do
  if [[ -f "$f" ]]; then
    sed -i 's/\r$//' "$f"
    echo "fixed: $f"
  fi
done
echo "Done. Restart: nohup bash start_cloud_server_linux.sh > cloud.log 2>&1 &"
