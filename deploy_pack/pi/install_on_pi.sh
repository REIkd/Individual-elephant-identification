#!/usr/bin/env bash
# 从 U 盘/拷贝目录安装到树莓派 ~/pi_cloud_deploy
# 用法: bash install_on_pi.sh /media/pi/USB/elephant_deploy/pi

set -euo pipefail

SRC="${1:-}"
DEST="${PI_DEPLOY_DIR:-$HOME/pi_cloud_deploy}"

if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "用法: bash install_on_pi.sh <U盘上的 pi 文件夹>"
  echo "示例: bash /media/pi/USB/elephant_pi/install_on_pi.sh /media/pi/USB/elephant_pi"
  exit 1
fi

mkdir -p "$DEST"
echo "=== 安装到 $DEST ==="

shopt -s nullglob
for f in "$SRC"/*; do
  base="$(basename "$f")"
  if [[ "$base" == "install_on_pi.sh" ]]; then
    continue
  fi
  cp -f "$f" "$DEST/$base"
  echo "  + $base"
done

chmod +x "$DEST"/*.sh 2>/dev/null || true

if command -v dos2unix >/dev/null 2>&1; then
  dos2unix "$DEST"/*.sh 2>/dev/null || true
fi

if [[ ! -f "$DEST/pi_cloud_config.sh" ]]; then
  if [[ -f "$DEST/pi_cloud_client_config.example.sh" ]]; then
    cp "$DEST/pi_cloud_client_config.example.sh" "$DEST/pi_cloud_config.sh"
    echo "已生成 pi_cloud_config.sh，请编辑 ELEPHANT_SERVER 和 CLOUD_API_KEY"
  fi
fi

if [[ ! -d "$DEST/.venv" ]]; then
  echo "=== 创建 Python 虚拟环境 ==="
  python3 -m venv "$DEST/.venv"
  # shellcheck source=/dev/null
  source "$DEST/.venv/bin/activate"
  pip install -U pip
  pip install -r "$DEST/requirements-pi-client.txt"
else
  echo "=== 更新依赖（可选）==="
  # shellcheck source=/dev/null
  source "$DEST/.venv/bin/activate"
  pip install -r "$DEST/requirements-pi-client.txt" -q || true
fi

mkdir -p "${PI_CLIP_DIR:-$HOME/elephant_clips}"

echo ""
echo "=== 完成 ==="
echo "1. 编辑配置: nano $DEST/pi_cloud_config.sh"
echo "2. 测试摄像头: cd $DEST && source .venv/bin/activate && python pi_cloud_client.py --probe-camera"
echo "3. 运行: cd $DEST && ./run_pi_cloud_client.sh"
echo "4. 网页录像库: http://<服务器>:9998/watch/clips"
