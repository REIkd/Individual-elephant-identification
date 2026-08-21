#!/usr/bin/env bash
# 从 U 盘/拷贝目录安装到服务器 ~/elephant_cloud
# 用法: bash install_on_server.sh /mnt/usb/elephant_deploy/server

set -euo pipefail

SRC="${1:-}"
DEST="${ELEPHANT_CLOUD_DIR:-$HOME/elephant_cloud}"

if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "用法: bash install_on_server.sh <拷贝过来的 server 目录>"
  echo "示例: bash install_on_server.sh /mnt/usb/elephant_deploy/server"
  exit 1
fi

mkdir -p "$DEST"
echo "=== 安装到 $DEST ==="

shopt -s nullglob
for f in "$SRC"/*; do
  base="$(basename "$f")"
  if [[ "$base" == "install_on_server.sh" ]]; then
    continue
  fi
  cp -f "$f" "$DEST/$base"
  echo "  + $base"
done

chmod +x "$DEST"/start_cloud_server*.sh 2>/dev/null || true
chmod +x "$DEST"/setup_uovision_ir.sh 2>/dev/null || true

if [[ -f "$DEST/deploy_pack/server/cloud_clip_env.sh" ]]; then
  cp -f "$DEST/deploy_pack/server/cloud_clip_env.sh" "$DEST/cloud_clip_env.sh" 2>/dev/null || true
fi
if [[ -f "$SRC/cloud_clip_env.sh" ]]; then
  cp -f "$SRC/cloud_clip_env.sh" "$DEST/cloud_clip_env.sh"
fi

echo ""
echo "=== 完成 ==="
echo "模型文件（若尚未在服务器）请单独拷贝:"
echo "  best_elephant_model.pth"
echo "  class_names.json"
echo "  yolov8m.pt"
echo ""
echo "启动:"
echo "  cd $DEST"
echo "  bash start_cloud_server_linux.sh"
echo ""
echo "录像库: http://<服务器IP>:9998/watch/clips"
