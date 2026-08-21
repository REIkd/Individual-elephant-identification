#!/usr/bin/env bash
# 共享 GPU / 空闲显存 < 3GB：YOLO 在 GPU，分类器在 CPU，yolov8n 最省显存

set -euo pipefail
cd "$(dirname "$0")"

: "${CLOUD_API_KEY:=elephant-demo-2026}"
: "${CLOUD_PORT:=9998}"
: "${ELEPHANT_GPU:=0}"

export CUDA_VISIBLE_DEVICES="${ELEPHANT_GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

echo "=== Low VRAM mode (YOLO GPU + classifier CPU) ==="
python -c "
import torch
print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device:', torch.cuda.get_device_name(0))
"

echo "=== GPU memory (physical GPU ${ELEPHANT_GPU}) ==="
nvidia-smi -i "${ELEPHANT_GPU}" --query-gpu=memory.used,memory.free,memory.total --format=csv 2>/dev/null || true

echo "========================================"
echo " Elephant Cloud Server (LOW VRAM, GPU ${ELEPHANT_GPU})"
echo " YOLO: yolov8n | classify: CPU | infer width: 480"
echo "========================================"

exec python cloud_server.py \
  --host 0.0.0.0 \
  --port "${CLOUD_PORT}" \
  --api-key "${CLOUD_API_KEY}" \
  --gpu 0 \
  --low-vram \
  --yolo-weights yolov8n.pt \
  --yolo-imgsz 416 \
  --infer-max-width 480 \
  --recog-interval 3 \
  --min-conf 42 \
  --min-margin 12 \
  --stream-max-width 854 \
  --stream-jpeg-quality 68 \
  --stream-fps 15
