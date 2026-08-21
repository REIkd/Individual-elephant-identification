# Pi + 云端大象识别部署说明

Pi 只负责 **摄像头 + 显示**；**YOLO 检测 + 个体分类** 在云端（GPU 服务器或家里高配 PC）运行。

---

## 架构

```
[树莓派]  USB 摄像头 1280×720
    │  每 0.15s 上传 JPEG（宽 640）
    ▼
[云端]  cloud_server.py  (FastAPI)
    │  YOLO + ResNet 跟踪识别
    ▼ JSON: 框 + 名字
[树莓派]  在原画面画框 → OpenCV 窗口
```

---

## 一、需要拷贝的文件

### 云端（GPU PC / 云主机）

整个项目目录中至少包含：

| 文件 | 说明 |
|------|------|
| `cloud_server.py` | API 服务 |
| `cloud_inference.py` | 推理封装 |
| `video_tracker_yolo.py` | 检测跟踪逻辑 |
| `predict.py` / `elephant_net.py` | 分类器 |
| `best_elephant_model.pth` | 个体模型 |
| `class_names.json` | 类别名 |
| `yolov8n.pt` | YOLO 权重（首次可自动下载） |
| `requirements-cloud.txt` | 依赖 |

### 树莓派（瘦客户端）

| 文件 | 说明 |
|------|------|
| `pi_cloud_client.py` | 客户端 |
| `requirements-pi-client.txt` | 依赖 |

Pi **不需要** `.pth`、`video_tracker_yolo.py`、PyTorch。

---

## 二、云端部署（Windows / Linux + 可选 NVIDIA GPU）

### 1. 创建环境

```bash
cd "你的项目目录"
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements-cloud.txt
# 有 GPU 请按 pytorch.org 安装 CUDA 版 torch
```

### 2. 设置 API 密钥（推荐）

```bash
# Windows PowerShell
$env:CLOUD_API_KEY="请改成随机长字符串"

# Linux
export CLOUD_API_KEY="请改成随机长字符串"
```

### 3. 启动服务

```bash
python cloud_server.py --host 0.0.0.0 --port 8000 --api-key %CLOUD_API_KEY%
```

Linux / macOS 将 `%CLOUD_API_KEY%` 改为 `$CLOUD_API_KEY`。

启动后浏览器或 curl 测试：

```bash
curl http://127.0.0.1:8000/health
```

应返回 `{"status":"ok",...}`。

### 4. 局域网 IP

记下云端机器 IP，例如 `192.168.1.100`，Pi 将连接：

`http://192.168.1.100:8000`

**防火墙**：放行 **8000** 端口（Windows 防火墙 / 云安全组）。

---

## 三、树莓派部署

### 1. 安装依赖

```bash
mkdir -p ~/elephant_cloud && cd ~/elephant_cloud
# 拷贝 pi_cloud_client.py 和 requirements-pi-client.txt

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-pi-client.txt
pip install opencv-python
sudo apt install -y fonts-noto-cjk
```

### 2. 运行客户端

```bash
export CLOUD_API_KEY="与云端相同的密钥"

python pi_cloud_client.py \
  --server http://192.168.1.100:8000 \
  --api-key "$CLOUD_API_KEY" \
  --camera 0
```

窗口按 **q** 退出，**r** 重置云端跟踪 session。

### 3. 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--upload-width` | 640 | 上传图宽度，越小越省带宽 |
| `--send-interval` | 0.15 | 发送间隔（秒），越小越跟手 |
| `--jpeg-quality` | 75 | JPEG 质量 |
| `--camera-width/height` | 1280/720 | 本地采集分辨率 |
| `--headless` | 关 | 无窗口，仅终端打印 |

更跟手（略增带宽）：

```bash
python pi_cloud_client.py --server http://192.168.1.100:8000 \
  --api-key "$CLOUD_API_KEY" --send-interval 0.1
```

---

## 四、开发调试（Pi 和云在同一 WiFi）

1. **先在电脑上** 启动 `cloud_server.py`
2. 电脑防火墙允许 8000
3. Pi 上 `--server http://电脑局域网IP:8000`
4. 确认 Pi 能 ping 通电脑

本机也可用 Pi 客户端连 `http://127.0.0.1:8000` 做联调。

---

## 五、公网部署（可选）

1. 云服务器租 **带 GPU** 实例，上传项目 + 模型
2. 使用 **HTTPS**（Nginx + Let's Encrypt）
3. **必须** 设置 `CLOUD_API_KEY`
4. Pi 上 `--server https://你的域名`

不建议无鉴权暴露到公网。

---

## 六、API 说明

### `GET /health`

健康检查。

### `POST /api/v1/infer`

- Header: `X-Api-Key: 你的密钥`（若云端启用了鉴权）
- Form:
  - `image`: JPEG 文件
  - `session_id`: 可选，同一摄像头会话请保持不变（客户端自动维护）

响应示例：

```json
{
  "session_id": "abc123",
  "frame_width": 640,
  "frame_height": 360,
  "tracks": [
    {"track_id": 1, "bbox": [120, 80, 200, 180], "name": "威望", "color_bgr": [80, 255, 100]}
  ],
  "latency_ms": 45.2
}
```

### `POST /api/v1/reset`

Form: `session_id` — 清空该 session 的跟踪状态。

---

## 七、故障排除

| 现象 | 处理 |
|------|------|
| Pi 无法连接 | 检查 IP、端口、防火墙；`curl http://IP:8000/health` |
| 401 错误 | `--api-key` 与云端 `CLOUD_API_KEY` 不一致 |
| 延迟高 | 减小 `--upload-width`；Pi 与云同区域；用有线网 |
| 中文方块 | Pi 上 `sudo apt install fonts-noto-cjk` |
| 云端 OOM | 换 GPU 实例或 `--yolo-imgsz 416` |

---

## 八、与本地 Pi 全量推理对比

| | Pi 本地跑模型 | Pi + 云 |
|--|----------------|---------|
| Pi 算力 | 吃满，2～6 FPS | 很轻，预览可 20+ FPS |
| 个体名字 | 慢 | 云端 GPU 快 |
| 网络 | 不需要 | 必须稳定 |
| 离线 | 可以 | 不可以 |

---

## 九、一键启动脚本（本地 PC + Pi）

### 电脑（Windows）

1. 首次安装依赖（只需一次）：
   ```powershell
   cd "项目目录"
   python -m venv .venv
   ```
   **双击** `install_cloud_deps.bat`（推荐，已处理 Windows 编码与代理问题）

   或手动安装（PowerShell）：
   ```powershell
   $env:NO_PROXY="*"
   .\.venv\Scripts\activate
   pip install -r requirements-cloud.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
   ```
2. 编辑 `start_cloud_server.bat` 里的 `CLOUD_API_KEY`（可选改端口）
3. **双击** `start_cloud_server.bat` 启动后端
4. `ipconfig` 查看 IPv4，例如 `192.168.1.100`

### 树莓派

1. 将整个 **`pi_cloud_deploy`** 文件夹拷到 Pi（U 盘即可），内含：
   - `pi_cloud_client.py`
   - `run_pi_cloud_client.sh`
   - `pi_cloud_client_config.example.sh`
   - `requirements-pi-client.txt`
2. 在 Pi 上：
   ```bash
   cd ~/pi_cloud_deploy
   cp pi_cloud_client_config.example.sh pi_cloud_config.sh
   nano pi_cloud_config.sh   # 改 ELEPHANT_SERVER 为 http://电脑IP:8000
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-pi-client.txt
   sudo apt install -y fonts-noto-cjk
   chmod +x run_pi_cloud_client.sh
   ./run_pi_cloud_client.sh
   ```

默认 API Key 在脚本里均为 `elephant-demo-2026`，**两边必须一致**。

---

完成以上步骤后，演示流程为：**电脑先启动 `start_cloud_server.bat` → Pi 再运行 `./run_pi_cloud_client.sh`**。
