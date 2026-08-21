# Pi 云端客户端（整夹拷到树莓派 ~/pi_cloud_deploy）

## 模式说明（当前）

- **网页直播：已关闭**
- **本地 1920 录像**：检测到大象后保存带标注 MP4
- **自动上传**到服务器 `/watch/clips`
- **3 天自动删除**（Pi 本地 + 服务器）

## U 盘安装（推荐）

Windows 打包到 U 盘：

```powershell
cd "D:\Project\Individual elephant identification"
.\deploy_pack\copy_to_usb_for_pi.ps1 -UsbRoot "E:\elephant_pi"
```

树莓派（U 盘插入后）：

```bash
bash /media/pi/USB/elephant_pi/install_on_pi.sh /media/pi/USB/elephant_pi
nano ~/pi_cloud_deploy/pi_cloud_config.sh
cd ~/pi_cloud_deploy && ./run_pi_cloud_client.sh
```

详见项目 `deploy_pack/树莓派部署清单.md`

## 首次配置（手动）

```bash
cd ~/pi_cloud_deploy
sudo apt install -y dos2unix v4l-utils fonts-noto-cjk ffmpeg
dos2unix *.sh
cp pi_cloud_client_config.example.sh pi_cloud_config.sh
nano pi_cloud_config.sh

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi-client.txt
chmod +x run_pi_cloud_client.sh install_service.sh
```

## 观看录像（不是直播）

http://120.196.88.140:9998/watch/clips
