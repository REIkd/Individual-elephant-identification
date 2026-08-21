#!/usr/bin/env bash
# 一次性配置 5 台红外相机：注册 /add → setServerBase → 识别参数
# 在云端项目目录执行，需 cloud_server 已启动且 UOVISION_GATEWAY_URL 可达

set -euo pipefail
cd "$(dirname "$0")"

if [[ -f "uovision_config.example.sh" ]]; then
  # shellcheck source=/dev/null
  source uovision_config.example.sh
fi

: "${CLOUD_BASE:=http://127.0.0.1:9998}"
: "${CLOUD_API_KEY:=elephant-demo-2026}"
: "${UOVISION_CALLBACK_BASE:=http://120.196.88.140:9998}"
: "${UOVISION_GATEWAY_URL:=http://open.uovcloud.com:98}"
: "${UOVISION_CAMERA_MODEL:=UML7}"

if ! curl -sf "${CLOUD_BASE}/health" >/dev/null 2>&1; then
  echo "ERROR: cloud_server 未在 ${CLOUD_BASE} 运行。"
  echo "请先: source uovision_config.example.sh && bash start_cloud_server_linux.sh"
  exit 1
fi

GW_CHECK="$(curl -s "${CLOUD_BASE}/health" | python -c "import sys,json; print(json.load(sys.stdin).get('uovision_gateway') or '')" 2>/dev/null || true)"
if [[ -z "${GW_CHECK}" ]]; then
  echo "ERROR: cloud_server 内 UOVISION_GATEWAY_URL 未生效（health 里 uovision_gateway 为空）。"
  echo "请重启服务:"
  echo "  pkill -f cloud_server.py; sleep 2"
  echo "  source uovision_config.example.sh"
  echo "  nohup bash start_cloud_server_linux.sh > cloud.log 2>&1 &"
  exit 1
fi
echo "UOVision 网关: ${GW_CHECK}"

echo ""
echo "=== 0. 注册相机到 UOVision 平台 (/add, model=${UOVISION_CAMERA_MODEL}) ==="
echo "    若机型不对，请 export UOVISION_CAMERA_MODEL=UMW7 等后再运行"
curl -sS -X POST "${CLOUD_BASE}/api/v1/uovision/register-all" \
  -H "X-Api-Key: ${CLOUD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${UOVISION_CAMERA_MODEL}\"}" | python -m json.tool

echo ""
echo "=== 1. 验证 5 台相机 IMEI / ICCID ==="
curl -sS "${CLOUD_BASE}/api/v1/uovision/verify-all" \
  -H "X-Api-Key: ${CLOUD_API_KEY}" | python -m json.tool

echo ""
echo "=== 2. setServerBase + 下发识别参数（原视频上传）==="
curl -sS -X POST "${CLOUD_BASE}/api/v1/uovision/setup-all" \
  -H "X-Api-Key: ${CLOUD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"dataSvr\":\"${UOVISION_CALLBACK_BASE}\",\"register\":false,\"configure\":true,\"trigger_capture\":false}" \
  | python -m json.tool

echo ""
echo "=== 3. 相机总览页 ==="
echo "  ${UOVISION_CALLBACK_BASE}/watch/ir"
echo ""
echo "=== 4. 手动触发某台测试（示例：01 号）==="
echo "  curl -X POST \"${CLOUD_BASE}/api/v1/uovision/ir/863386077691243/start-recording\" -H \"X-Api-Key: ${CLOUD_API_KEY}\""
