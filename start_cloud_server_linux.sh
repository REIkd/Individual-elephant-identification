#!/usr/bin/env bash
# Run on Linux AI server with NVIDIA GPU (default GPU 0)

set -euo pipefail
cd "$(dirname "$0")"

# UOVision 红外相机（可选）
if [[ -f "uovision_config.example.sh" ]]; then
  # shellcheck source=/dev/null
  source uovision_config.example.sh
fi

# Pi 录像上传模式：关闭 MJPEG 直播，录像 3 天自动清理
if [[ -f "cloud_clip_env.sh" ]]; then
  # shellcheck source=/dev/null
  source cloud_clip_env.sh
fi

: "${CLOUD_API_KEY:=elephant-demo-2026}"
: "${CLOUD_PORT:=9998}"
: "${ELEPHANT_GPU:=0}"
# 仅暴露一张卡，避免 PyTorch/Ultralytics 误用另一张 GPU
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

echo "=== GPU check (physical GPU ${ELEPHANT_GPU} -> logical cuda:0) ==="
python -c "
import torch
print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA not available. Run: bash install_gpu_pytorch.sh')
for i in range(torch.cuda.device_count()):
    print(f'  cuda:{i}:', torch.cuda.get_device_name(i))
"

echo "=== GPU memory (physical GPU ${ELEPHANT_GPU}, before start) ==="
MEM_CSV="$(nvidia-smi -i "${ELEPHANT_GPU}" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "${MEM_CSV}" ]]; then
  echo "index,memory.used [MiB],memory.free [MiB],memory.total [MiB]"
  echo "${ELEPHANT_GPU},${MEM_CSV}"
  FREE_MIB="$(echo "${MEM_CSV}" | awk -F', ' '{print $2}' | tr -d ' ')"
  MIN_FREE_MIB="${ELEPHANT_MIN_FREE_MIB:-2500}"
  if [[ "${FREE_MIB}" =~ ^[0-9]+$ ]] && (( FREE_MIB < MIN_FREE_MIB )) && [[ "${ELEPHANT_FORCE_GPU:-0}" != "1" ]]; then
    echo ""
    echo "[ERROR] GPU ${ELEPHANT_GPU} 空闲显存仅 ${FREE_MIB} MiB，至少需要约 ${MIN_FREE_MIB} MiB 才能跑 YOLO+分类。"
    echo "        进程列表为空但显存已满，通常是同一物理 GPU 上还有其他容器/用户在占用。"
    echo "        可选方案："
    echo "          1) 联系平台换独占 GPU 或更空闲的实例"
        echo "          2) 低显存混合: bash start_cloud_server_low_vram.sh"
        echo "          3) CPU 模式: bash start_cloud_server_cpu.sh"
        echo "          4) 仍强行启动: ELEPHANT_FORCE_GPU=1 bash start_cloud_server_linux.sh"
    exit 1
  fi
else
  nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv 2>/dev/null || true
fi

echo "========================================"
echo " Elephant Cloud Server (physical GPU ${ELEPHANT_GPU})"
echo " Port: ${CLOUD_PORT}"
echo " API Key: ${CLOUD_API_KEY}"

# 远距离场景：export ELEPHANT_FAR=1 再启动（更高分辨率检测 + 更严出名字）
if [[ "${ELEPHANT_FAR:-0}" == "1" ]]; then
  YOLO_W="yolov8m.pt"
  YOLO_SZ=960
  INFER_W=1920
  MIN_CONF=55
  MIN_MARGIN=22
  echo " Mode: FAR (远距离优化)"
else
  YOLO_W="yolov8m.pt"
  YOLO_SZ=640
  INFER_W=1280
  MIN_CONF=50
  MIN_MARGIN=18
  echo " Mode: standard (可设 ELEPHANT_FAR=1 加强远距离)"
fi
echo " YOLO: ${YOLO_W} | infer max width: ${INFER_W}"
echo "========================================"

ALLOWED_ARGS=()
if [[ -n "${ELEPHANT_ALLOWED:-}" ]]; then
  ALLOWED_ARGS=(--allowed-elephants "$ELEPHANT_ALLOWED")
elif [[ -f allowed_elephants.json ]]; then
  ALLOWED_ARGS=(--allowed-elephants allowed_elephants.json)
fi

exec python cloud_server.py \
  --host 0.0.0.0 \
  --port "${CLOUD_PORT}" \
  --api-key "${CLOUD_API_KEY}" \
  --gpu 0 \
  --yolo-weights "${YOLO_W}" \
  --yolo-imgsz "${YOLO_SZ}" \
  --infer-max-width "${INFER_W}" \
  --recog-interval 2 \
  --min-conf "${MIN_CONF}" \
  --min-margin "${MIN_MARGIN}" \
  --stream-max-width 1280 \
  --stream-jpeg-quality 80 \
  --stream-fps 20 \
  --freeze-locked \
  "${ALLOWED_ARGS[@]}"
