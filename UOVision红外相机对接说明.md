# UOVision 红外相机 + 大象识别对接说明

开放平台：`http://open.uovcloud.com:98/swagger-ui.html`  
识别服务器：`http://120.196.88.140:9998`

## 5 台相机（已登记在 uovision_cameras.json）

| 编号 | SN | IMEI（cameraCode） | ICCID |
|------|-----|-------------------|-------|
| 01 | SN147000150790030 | **863386077691243** | 898604851925c0009742 |
| 02 | SN147000150790022 | **863386077682150** | 898604851925c0009737 |
| 03 | SN147000150790031 | **863386077672052** | 898604851925c0009733 |
| 04 | SN147000150790032 | **863386077686185** | 898604851925c0009736 |
| 05 | SN147000150790026 | **863386077682325** | 898604851925c0009732 |

物联卡号均为 144085711。API 里 **`cameraCode` = IMEI**，不是 SN。

## 数据流

```
[红外相机 PIR/远程] → 录制 ≤60s 原视频
        → POST http://120.196.88.140:9998/servlet/original(2)
        → GPU 识别打框 → latest.mp4
        → 网页 /watch/ir 或 /watch/ir-{IMEI}
```

## 一次性部署（服务器上）

### 1. 上传文件并启动

```bash
cd ~/elephant_cloud
source uovision_config.example.sh
bash start_cloud_server_linux.sh
```

需包含：`uovision_cameras.json`、`uovision_registry.py`、`uovision_open_api.py` 等。

### 2. 批量配置 5 台相机

```bash
bash setup_uovision_ir.sh
```

或手动：

```bash
# 验证 IMEI + ICCID
curl "http://127.0.0.1:9998/api/v1/uovision/verify-all" -H "X-Api-Key: elephant-demo-2026"

# setServerBase + 下发原视频上传参数
curl -X POST "http://127.0.0.1:9998/api/v1/uovision/setup-all" \
  -H "X-Api-Key: elephant-demo-2026" \
  -H "Content-Type: application/json" \
  -d '{"dataSvr":"http://120.196.88.140:9998","configure":true}'
```

### 3. 测试触发（01 号）

```bash
curl -X POST "http://127.0.0.1:9998/api/v1/uovision/ir/863386077691243/start-recording" \
  -H "X-Api-Key: elephant-demo-2026"
```

相机上传原视频后，打开：

- 总览：`http://120.196.88.140:9998/watch/ir`
- 01 号：`http://120.196.88.140:9998/watch/ir-863386077691243`

## 网页入口

| 地址 | 说明 |
|------|------|
| `/watch/ir` | 5 台相机状态总览 |
| `/watch/ir-863386077691243` | 01 号最新识别视频 |
| `/watch/ir-863386077682150` | 02 号 |
| `/watch/ir-863386077672052` | 03 号 |
| `/watch/ir-863386077686185` | 04 号 |
| `/watch/ir-863386077682325` | 05 号 |

## API 摘要

| 接口 | 说明 |
|------|------|
| `GET /api/v1/uovision/cameras` | 5 台列表 + 识别状态 |
| `GET /api/v1/uovision/verify-all` | 核对 IMEI/ICCID |
| `POST /api/v1/uovision/setup-all` | 批量 setServerBase + 改参 |
| `POST /api/v1/uovision/ir/{IMEI}/start-recording` | 改参 + getFile 触发 |
| `GET /api/v1/uovision/ir/{IMEI}/status` | 处理进度 |
| `POST /servlet/original` | 相机上传原视频 |
| `POST /camera/heatbeat` | 相机定时心跳（1.4，文档路径） |
| `POST /servlet/heartbeat` | 同上（curl 示例兼容路径） |
| `GET /api/v1/uovision/ir/{IMEI}/heartbeat` | 查询在线/电量/信号 |

## 环境变量

```bash
export UOVISION_GATEWAY_URL="http://open.uovcloud.com:98"
export UOVISION_CALLBACK_BASE="http://120.196.88.140:9998"
export UOVISION_CAMERAS_FILE="uovision_cameras.json"
export UOVISION_VIDEO_LENGTH="60"
```

## 识别参数（自动下发）

- `cameraMode=2` 拍照+录像
- `sendVideoSize=2` 原视频上传
- `videoLength=60` 单段最长 60 秒（官方上限）
- `remoteControl=1` 远程实时唤醒（配合 getFile）
- `triggerMode=0` PIR 触发

## scp 上传清单

```bash
scp -P 12222 cloud_server.py cloud_inference.py uovision_camera.py uovision_open_api.py \
  uovision_registry.py uovision_cameras.json uovision_video_pipeline.py \
  uovision_config.example.sh setup_uovision_ir.sh \
  video_tracker_yolo.py predict.py cloud_render.py \
  root@120.196.88.140:/root/elephant_cloud/
```

服务器建议安装 `ffmpeg`（浏览器 H.264 兼容）。
