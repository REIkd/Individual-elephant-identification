# Copy to pi_cloud_config.sh and edit (source only — do NOT chmod +x):
#   cp pi_cloud_client_config.example.sh pi_cloud_config.sh
#   nano pi_cloud_config.sh

# --- LAN: home PC ---
# export ELEPHANT_SERVER="http://192.168.x.x:8000"

# --- Public cloud (port 9998) ---
export ELEPHANT_SERVER="http://120.196.88.140:9998"

export CLOUD_API_KEY="elephant-demo-2026"

# Web: http://120.196.88.140:9998/watch/elephant-live
export STREAM_ID="elephant-live"

# Camera: auto | /dev/video0 | 2  (run: v4l2-ctl --list-devices)
export CAMERA_DEVICE="auto"
export CAMERA_WIDTH="1280"
export CAMERA_HEIGHT="720"

# Upload quality — affects cloud recognition (higher = clearer, more bandwidth)
export UPLOAD_WIDTH="1280"
export JPEG_QUALITY="88"
export SEND_INTERVAL="0.08"

# 远距离：1080p 采集 + 1920 上传（大象在画面里较小时）
# export CAMERA_WIDTH="1920"
# export CAMERA_HEIGHT="1080"
# export UPLOAD_WIDTH="1920"
# export JPEG_QUALITY="90"

# systemd headless mode (set by run_pi_cloud_client.sh --service)
# export HEADLESS="1"
