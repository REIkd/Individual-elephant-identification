# Active Pi config (edit this file; sourced by run_pi_cloud_client.sh)

export ELEPHANT_SERVER="http://120.196.88.140:9998"
export CLOUD_API_KEY="elephant-demo-2026"
export STREAM_ID="elephant-live"

export CAMERA_DEVICE="auto"
export CAMERA_WIDTH="1920"
export CAMERA_HEIGHT="1080"

# 识别仍上传 JPEG（可小于采集分辨率以减轻云端负载）
export UPLOAD_WIDTH="1280"
export JPEG_QUALITY="88"
export SEND_INTERVAL="0.08"

# 本地录像：1920 全高清，检测到大象后录制并上传
export PI_CLIP_ENABLE="1"
export PI_CLIP_DIR="$HOME/elephant_clips"
export PI_CLIP_MAX_WIDTH="1920"
export PI_CLIP_FPS="15"
export PI_CLIP_PRE_SEC="2"
export PI_CLIP_POST_SEC="10"
export PI_CLIP_RETENTION_DAYS="3"
export PI_CLIP_DELETE_AFTER_UPLOAD="1"
