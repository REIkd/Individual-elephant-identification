# Copy to pi_cloud_config.sh and edit if needed (source only, do NOT chmod +x):
# cp pi_cloud_client_config.example.sh pi_cloud_config.sh
# Remote AI server (public port 9998)
export ELEPHANT_SERVER="http://120.196.88.140:9998"

# Must match server CLOUD_API_KEY
export CLOUD_API_KEY="elephant-demo-2026"

# Web watch URL: http://120.196.88.140:9998/watch/elephant-live
export STREAM_ID="elephant-live"

# Camera: auto | /dev/video0 | 2 (会转成 /dev/video2). Run: v4l2-ctl --list-devices
export CAMERA_DEVICE="auto"
export CAMERA_WIDTH="1280"
export CAMERA_HEIGHT="720"

# 上传画质（影响云端识别准确率；越大越清晰但延迟略增）
export UPLOAD_WIDTH="1280"
export JPEG_QUALITY="88"
export SEND_INTERVAL="0.08"
