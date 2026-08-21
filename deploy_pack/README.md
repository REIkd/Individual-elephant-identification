# 部署方式说明

| 设备 | 方式 | 脚本/文档 |
|------|------|-----------|
| **AI 服务器** | 网络上传代码（scp） | `upload_to_server.ps1` · `服务器部署清单.md` |
| **树莓派 Pi** | **U 盘/移动硬盘拷贝** | `copy_to_usb_for_pi.ps1` · `树莓派部署清单.md` |

---

## 一、服务器：scp 上传（Windows）

```powershell
cd "D:\Project\Individual elephant identification"
.\deploy_pack\upload_to_server.ps1
```

默认上传到 `root@120.196.88.140:12222` → `/root/elephant_cloud/`  
上传完成后 SSH 登录重启：

```bash
ssh -p 12222 root@120.196.88.140
cd ~/elephant_cloud && bash start_cloud_server_linux.sh
```

详见 **`服务器部署清单.md`**

---

## 二、树莓派：U 盘拷贝

**1. Windows 打包到 U 盘**

```powershell
cd "D:\Project\Individual elephant identification"
.\deploy_pack\copy_to_usb_for_pi.ps1 -UsbRoot "E:\elephant_pi"
```

会生成 `E:\elephant_pi\`（仅 Pi 文件，不含服务器代码）。

**2. U 盘插到 Pi，安装**

```bash
bash /media/pi/USB/elephant_pi/install_on_pi.sh /media/pi/USB/elephant_pi
nano ~/pi_cloud_deploy/pi_cloud_config.sh
cd ~/pi_cloud_deploy && ./run_pi_cloud_client.sh
```

详见 **`树莓派部署清单.md`**

---

## 功能（当前版本）

- 网页直播：**关闭**
- Pi：**1920 本地录像** → 上传服务器
- 观看：**http://服务器:9998/watch/clips**
- 录像 **3 天** 自动删除（Pi 本地 + 服务器）
