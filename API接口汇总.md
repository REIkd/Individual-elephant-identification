# 大象识别云端 · API 接口汇总

> 识别服务器（本项目）：**`http://120.196.88.140:9998`**  
> UOVision 厂商平台（仅后台转发）：**`http://open.uovcloud.com:98`**  
> 相机上传、网页观看走 **9998**，不走厂商平台。

最后更新：2026-07-28

---

## 1. 鉴权

管理类接口需 HTTP Header：

```http
X-Api-Key: elephant-demo-2026
```

| 类型 | 说明 |
|------|------|
| **公开** | 无需 API Key |
| **管理** | 需要 `X-Api-Key` |
| **相机** | 相机 4G 直传，无需 Key |

---

## 2. 系统

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/health` | 公开 | 服务状态、网关地址、保留策略配置 |

**`/health` 返回字段示例：**

| 字段 | 说明 |
|------|------|
| `status` | `ok` 表示服务正常 |
| `uovision_gateway` | 厂商平台地址 |
| `uovision_retention_days` | 红外文件磁盘保留天数（默认 3） |
| `uovision_ir_history_limit` | 每台相机网页可回看条数（默认 3） |
| `clip_retention_days` | Pi 录像库保留天数（默认 3） |
| `uovision_data_dir` | 红外数据目录 |

---

## 3. 红外相机 · 上传（相机 → 9998）

| 方法 | 路径 | 状态 | 说明 |
|------|------|------|------|
| POST | `/servlet/original2` | ✅ | **主动上报**原图/原视频（主路径） |
| POST | `/servlet/original` | ✅ | 平台拉取后回传（处理同 original2） |
| POST | `/servlet/photos` | ✅ | 缩略图/预览图 |
| POST | `/servlet/compress` | ❌ | 压缩图（未开发） |
| POST | `/servlet/thumbnail` | ❌ | 缩略图文档路径（未开发，已用 photos 替代） |

### 3.1 请求 Header（必需）

| Header | 说明 |
|--------|------|
| `X-CameraCode` | 相机 **IMEI**（15 位数字） |
| `X-File-Id` | 文件 ID（整数） |
| `X-File-Size` | 文件字节数 |
| `X-Is-Hq` | `1` = 原图；`2` = 原视频 |
| `Content-Type` | 原图：`image/jpeg`；视频：`video/mp4` 或 `video/mpeg4` |

### 3.2 请求 Body

二进制文件流（JPEG 或 MP4/MPEG4）。

### 3.3 服务器处理

- **原图**（`X-Is-Hq=1`）→ 落盘 + 单张 YOLO/分类识别  
- **原视频**（`X-Is-Hq=2`）→ 落盘 + GPU 排队逐帧识别打框 → 网页播放  

### 3.4 响应格式

成功时返回 JSON，含 `code: 0`、`message: "OK"`、`cameraCode` 等字段。

---

## 4. 红外相机 · 心跳（相机 → 9998）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/camera/heatbeat` | 对接文档标准路径（厂商原文拼写） |
| POST | `/camera/heartbeat` | 别名 |
| POST | `/servlet/heartbeat` | curl 示例兼容路径 |
| GET | `/servlet/heartbeat` | 联调备用（Query 传参） |

**Content-Type：** `application/json`  
**必填字段：** `cameraCode`（IMEI）

**可选字段：** `latitude`, `longitude`, `altitude`, `temperature`, `signal`, `battery`, `gps`, `model`, `capacity`, `freeSpace`, `iccid`, `firmware`

**响应示例：**

```json
{
  "code": 0,
  "message": "OK",
  "cameraCode": "863386077682150",
  "received_at": "2026-07-28T06:00:00+00:00"
}
```

**在线判定：** 默认 2 小时内收到心跳视为在线（`UOVISION_HEARTBEAT_ONLINE_SEC` 可调）。

---

## 5. 红外相机 · 网页观看（浏览器，公开）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/watch/ir` | 5 台相机总览（在线/识别状态） |
| GET | `/watch/ir-{IMEI}` | 单台观看页（最新自动播放 + 最近 N 条切换） |
| GET | `/watch/{stream_id}` | `stream_id = ir-{IMEI}` 时同上 |

**示例：**

| 说明 | URL |
|------|-----|
| 总览 | http://120.196.88.140:9998/watch/ir |
| 01 号 | http://120.196.88.140:9998/watch/ir-863386077691243 |
| 02 号 | http://120.196.88.140:9998/watch/ir-863386077682150 |
| 03 号 | http://120.196.88.140:9998/watch/ir-863386077672052 |
| 04 号 | http://120.196.88.140:9998/watch/ir-863386077686185 |
| 05 号 | http://120.196.88.140:9998/watch/ir-863386077682325 |

---

## 6. 红外相机 · 状态与视频（公开）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/uovision/cameras` | 注册表全部相机 + 流水线/心跳状态 |
| GET | `/api/v1/uovision/ir/{IMEI}/status` | 最新视频识别进度（网页轮询） |
| GET | `/api/v1/uovision/ir/{IMEI}/heartbeat` | 单台在线/电量/信号 |
| GET | `/api/v1/uovision/ir/{IMEI}/videos?limit=3` | 最近 N 条已识别视频列表 |
| GET | `/api/v1/uovision/ir/{IMEI}/latest.mp4` | 播放最新识别 MP4 |
| GET | `/api/v1/uovision/ir/{IMEI}/videos/{file_id}.mp4` | 播放指定 file_id 的 MP4 |

**`/videos` 列表项字段：**

| 字段 | 说明 |
|------|------|
| `file_id` | 相机文件 ID |
| `updated_at` | 处理完成时间 |
| `elephant_names` | 识别到的象名 |
| `video_url` | MP4 播放地址 |
| `duration_sec` | 视频时长（如有） |

---

## 7. 红外相机 · 管理（需 API Key）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/uovision/register-all` | 批量 POST `/add` 注册 IMEI 到厂商平台 |
| POST | `/api/v1/uovision/setup-all` | 批量 `setServerBase` + 下发识别参数 |
| POST | `/api/v1/uovision/set-server-base` | 单独设置 `dataSvr` |
| GET | `/api/v1/uovision/verify-all` | 验证全部相机 IMEI/ICCID/机型 |
| GET | `/api/v1/uovision/query-model?camera_code={IMEI}` | 查平台登记机型 |
| GET | `/api/v1/uovision/camera-info?camera_code={IMEI}` | 读相机参数（ICCID、电量等） |
| GET | `/api/v1/uovision/probe-iccid?iccid=&candidate_imeis=` | ICCID 反查 IMEI |
| GET | `/api/v1/uovision/settings-template?camera_code={IMEI}` | 改参 JSON 模板 |
| POST | `/api/v1/uovision/ir/{IMEI}/configure` | 单台 merge 参数 + modifySettings |
| POST | `/api/v1/uovision/ir/{IMEI}/start-recording` | 改参 + 远程 getFile 触发拍摄 |
| POST | `/api/v1/uovision/ir/{IMEI}/trigger-capture` | 仅远程 getFile |
| GET | `/api/v1/uovision/events?limit=50&camera_code=` | 最近上传/识别事件（内存，重启清空） |
| GET | `/api/v1/uovision/heartbeats` | 全部相机心跳摘要 |

### 7.1 setup-all 请求体

```json
{
  "dataSvr": "http://120.196.88.140:9998",
  "register": false,
  "model": "UML7",
  "configure": true,
  "trigger_capture": false,
  "video_length_sec": 60
}
```

### 7.2 set-server-base 请求体

```json
{
  "imeiList": ["863386077691243", "863386077682150"],
  "dataSvr": "http://120.196.88.140:9998"
}
```

### 7.3 常用 curl

```bash
# 健康检查
curl http://120.196.88.140:9998/health

# 验证 5 台相机
curl "http://120.196.88.140:9998/api/v1/uovision/verify-all" \
  -H "X-Api-Key: elephant-demo-2026"

# 批量配置
curl -X POST "http://120.196.88.140:9998/api/v1/uovision/setup-all" \
  -H "X-Api-Key: elephant-demo-2026" \
  -H "Content-Type: application/json" \
  -d '{"dataSvr":"http://120.196.88.140:9998","configure":true}'

# 远程触发 02 号
curl -X POST "http://120.196.88.140:9998/api/v1/uovision/ir/863386077682150/start-recording" \
  -H "X-Api-Key: elephant-demo-2026"

# 最近 3 条视频
curl "http://120.196.88.140:9998/api/v1/uovision/ir/863386077682150/videos?limit=3"
```

> **注意：** 远程改参/getFile 使用的机型以厂商平台 `queryModel` 为准（可能与机身铭牌不同）。若 modify 报 code 35，请核对平台登记机型。

---

## 8. 厂商平台转发（9998 → open.uovcloud.com，需 API Key）

| 方法 | 路径 | 转发目标 |
|------|------|----------|
| POST | `/acquirefile` | `POST /acquireFile` |
| POST | `/acquireFile` | 同上（大小写别名） |
| POST | `/modifysettings` | `POST /modifySettings{Model}` |

---

## 9. Pi 客户端 · 推理（需 API Key）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/infer` | 上传 JPEG 帧识别（multipart: `image`, 可选 `session_id`） |
| POST | `/api/v1/reset` | 重置指定 session 的跟踪状态（form: `session_id`） |

**`/api/v1/infer` 返回：** 识别框、象名、置信度、`latency_ms` 等。

---

## 10. Pi · MJPEG 直播（默认关闭）

环境变量 `LIVE_STREAM_ENABLE=0` 时不可用；当前部署以**录像上传**为主。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/stream/{stream_id}` | MJPEG 流 |
| GET | `/snapshot/{stream_id}` | 最新 JPEG 快照 |
| GET | `/api/v1/stream/{stream_id}/status` | 流状态 |
| GET | `/api/v1/stream/{stream_id}/tracks` | 当前识别框/象名（网页轮询） |

---

## 11. Pi · 录像库（上传模式，公开浏览）

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/watch/clips` | 公开 | 录像库网页 |
| GET | `/watch/clips/{clip_id}` | 公开 | 单条播放页 |
| GET | `/api/v1/clips?limit=50&session_id=` | 公开 | 录像列表 JSON |
| GET | `/api/v1/clips/{clip_id}` | 公开 | 单条元数据 |
| GET | `/api/v1/clips/{clip_id}/video.mp4` | 公开 | 播放 MP4 |
| POST | `/api/v1/clips/upload` | 管理 | Pi 上传标注 MP4（multipart: `video` + `meta` JSON） |

---

## 12. 识别配置 · 候选象

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/api/v1/class-names` | 公开 | 全部象名（17 类） |
| GET | `/api/v1/allowed-elephants?session_id=` | 公开 | 查询全局/会话候选象 |
| PUT | `/api/v1/allowed-elephants` | 管理 | 设置全局候选象 |
| PUT | `/api/v1/session/{session_id}/allowed-elephants` | 管理 | 设置单会话候选象 |

**设置全局候选象：**

```bash
curl -X PUT "http://120.196.88.140:9998/api/v1/allowed-elephants" \
  -H "X-Api-Key: elephant-demo-2026" \
  -H "Content-Type: application/json" \
  -d '{"names":["凯恩","隆隆"]}'

# 恢复全部 17 类
curl -X PUT "http://120.196.88.140:9998/api/v1/allowed-elephants" \
  -H "X-Api-Key: elephant-demo-2026" \
  -H "Content-Type: application/json" \
  -d '{"names":null}'
```

---

## 13. 五台红外相机 IMEI

| 编号 | SN | IMEI（cameraCode） |
|------|-----|-------------------|
| 01 | SN147000150790030 | 863386077691243 |
| 02 | SN147000150790022 | 863386077682150 |
| 03 | SN147000150790031 | 863386077672052 |
| 04 | SN147000150790032 | 863386077686185 |
| 05 | SN147000150790026 | 863386077682325 |

API 与上传 Header 中的 **`cameraCode` = IMEI**，不是 SN、不是 ICCID。

---

## 14. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `CLOUD_API_KEY` | `elephant-demo-2026` | 管理 API 密钥 |
| `CLOUD_PORT` | `9998` | 服务端口 |
| `UOVISION_GATEWAY_URL` | `http://open.uovcloud.com:98` | 厂商平台 |
| `UOVISION_CALLBACK_BASE` | `http://120.196.88.140:9998` | setServerBase 的 dataSvr |
| `UOVISION_CAMERAS_FILE` | `uovision_cameras.json` | 相机注册表 |
| `UOVISION_VIDEO_LENGTH` | `60` | 单段录像上限（秒，官方 5–60） |
| `UOVISION_RETENTION_DAYS` | `3` | 红外文件磁盘保留天数 |
| `UOVISION_IR_HISTORY_LIMIT` | `3` | 每台网页可回看条数 |
| `ELEPHANT_CLIP_RETENTION_DAYS` | `3` | Pi 录像库保留天数 |
| `LIVE_STREAM_ENABLE` | `0` | MJPEG 直播开关 |

配置见 `uovision_config.example.sh`、`cloud_clip_env.sh`。

---

## 15. 数据流

```
[红外相机 PIR / 远程触发]
    → POST /servlet/original2（+ 可选 /servlet/photos）
    → GPU 识别打框
    → /watch/ir-{IMEI}（最近 3 条）
    → 超过 3 天自动删除

[Pi 客户端]
    → POST /api/v1/clips/upload
    → /watch/clips
    → 超过 3 天自动删除

[运维 / 远程]
    → POST /api/v1/uovision/ir/{IMEI}/start-recording
    → 厂商平台 getFile → 相机上传 → 同上识别链路
```

---

## 16. 相关文档

| 文件 | 内容 |
|------|------|
| `红外相机对接接口清单.md` | UOVision 对接细节、两台服务器说明 |
| `UOVision红外相机对接说明.md` | 部署步骤与测试流程 |
| `deploy_pack/服务器部署清单.md` | 服务器部署 |
| `deploy_pack/树莓派部署清单.md` | Pi 部署 |

---

## 17. 网络要求

- 公网 **TCP 9998** 必须对相机 4G 出口可访问  
- 防火墙/安全组放行 `120.196.88.140:9998`  
- 相机 `setServerBase` 的 `dataSvr` 填 `http://120.196.88.140:9998`（不带 `/servlet/...` 后缀）
