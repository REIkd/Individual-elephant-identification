# UOVision / 林境开放平台 + 大象识别云端
# 用法: source uovision_config.example.sh  或在 start 脚本里 export

# 厂商开放平台（下发 modifysettings / acquirefile 转发目标）
export UOVISION_GATEWAY_URL="http://open.uovcloud.com:98"

# 相机硬件型号（/add 与 modifySettings 必需，须与机身/林境平台一致）
# 可选: UML5P UMW5P UML8 UML7 UMW7 UMW2 UMQ2 UMSW3 UMQ6 UML6
export UOVISION_CAMERA_MODEL="UML7"

export UOVISION_CALLBACK_BASE="http://120.196.88.140:9998"
export UOVISION_DATA_DIR="data/uovision"
export UOVISION_MAX_MB="512"
# 官方 Swagger：videoLength 仅 5–60 秒
export UOVISION_VIDEO_LENGTH="60"
# remoteControl：1=实时唤醒，2–9=延迟 0.5H–24H（不是「开始录像」）
export UOVISION_REMOTE_CONTROL="1"

# 每台相机网页保留最近几条识别视频（默认 3）
export UOVISION_IR_HISTORY_LIMIT="${UOVISION_IR_HISTORY_LIMIT:-3}"
# 红外原视频/识别结果磁盘保留天数（默认 3 天自动删除）
export UOVISION_RETENTION_DAYS="${UOVISION_RETENTION_DAYS:-3}"
# 心跳/上传活动在线判定窗口（秒）；相机心跳间隔可能较长，默认 24 小时
export UOVISION_HEARTBEAT_ONLINE_SEC="${UOVISION_HEARTBEAT_ONLINE_SEC:-86400}"

# 相机注册表（5 台 IMEI）
export UOVISION_CAMERAS_FILE="uovision_cameras.json"
