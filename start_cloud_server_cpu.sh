#!/usr/bin/env bash
# CPU 推理（共享 GPU 显存被占满时使用，速度较慢但可运行）

set -euo pipefail
cd "$(dirname "$0")"

: "${CLOUD_API_KEY:=elephant-demo-2026}"
: "${CLOUD_PORT:=9998}"

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=""

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run: bash install_gpu_pytorch.sh"
  exit 1
fi

if [[ ! -f "best_elephant_model.pth" ]]; then
  echo "Missing best_elephant_model.pth"
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "=== CPU mode (no GPU) ==="
python -c "
import torch
print('PyTorch', torch.__version__, '| CUDA visible:', torch.cuda.is_available())
"

echo "========================================"
echo " Elephant Cloud Server (CPU)"
echo " Port: ${CLOUD_PORT}"
echo " API Key: ${CLOUD_API_KEY}"
echo " YOLO: yolov8n | infer max width: 960"
echo " 网页流: 1280px JPEG 80 | 预期延迟: 约 120~350 ms/帧（CPU）"
echo "========================================"

exec python cloud_server.py \
  --host 0.0.0.0 \
  --port "${CLOUD_PORT}" \
  --api-key "${CLOUD_API_KEY}" \
  --gpu 0 \
  --yolo-weights yolov8n.pt \
  --yolo-imgsz 480 \
  --infer-max-width 960 \
  --recog-interval 2 \
  --min-conf 42 \
  --min-margin 12 \
  --stream-max-width 1280 \
  --stream-jpeg-quality 80 \
  --stream-fps 15
