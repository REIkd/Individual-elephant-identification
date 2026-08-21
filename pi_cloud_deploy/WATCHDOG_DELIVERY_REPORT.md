# Pi Cloud Client — 看门狗重构交付报告

> 项目：树莓派云端大象识别客户端  
> 日期：2026-06-24  
> 平台：Raspberry Pi aarch64, Debian 13 (trixie), systemd 257

---

## 一、需求

1. **插电开机自启动 / 断电重启**：树莓派通电后服务自动启动，无需人工登录。
2. **周期性看门狗**：每隔固定秒数检测主服务是否在运行，未运行则自动拉起。
3. **简单可落地**：机制不复杂，现场经得起 `ps aux` 与 `systemctl status` 检查。
4. **容忍服务器连接抖动**：云端掉线时避免服务进入 failed 状态或日志刷屏。

---

## 二、实现方案

### 总体架构

    ┌─────────────────────────────────────────────────────┐
    │              systemd user instance                   │
    │  ┌───────────────────────────────────────────────┐  │
    │  │      pi-cloud-client.service  (主业务)        │  │
    │  │  Type=simple, Restart=always, RestartSec=10   │  │
    │  │  StartLimitIntervalSec=0 (持续重试)           │  │
    │  └───────────────────────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────┐  │
    │  │   pi-cloud-client-watchdog.service (看门狗)   │  │
    │  │  每 10s 执行 pi_cloud_watchdog.sh              │  │
    │  │  inactive → reset-failed → start              │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
                     两者均 enable → default.target
                     配合 loginctl enable-linger 实现开机自启

### 技术选型

| 组件 | 方案 | 理由 |
|------|------|------|
| 进程管理 | systemd user service | 无需 sudo；配合 linger 可开机自启 |
| 开机自启 | `loginctl enable-linger` + `WantedBy=default.target` | 用户级服务无需登录即可启动 |
| 崩溃/被杀恢复 | `Restart=always` + `RestartSec=10` | systemd 事件驱动，进程消亡立即拉起 |
| 防启动风暴 | `StartLimitIntervalSec=0` | 云端/摄像头瞬断场景需要持续重试，不设上限 |
| 兜底看门狗 | `pi-cloud-client-watchdog.service` | 独立进程，每 10s 轮询，`ps aux` 可见 |
| 日志 | `logs/watchdog.log` + journald | 双通道，便于排查 |

### 文件改动

| 文件 | 改动 |
|------|------|
| `pi-cloud-client.service` | `Type=simple`；取消启动限制；保留 `After=network-online.target` |
| `pi-cloud-client-watchdog.service` | 新建看门狗服务单元 |
| `pi_cloud_watchdog.sh` | 重写为长期运行循环脚本，写死 10s 检测间隔，双通道日志 |
| `install_service.sh` | 安装并启用主服务 + 看门狗服务；清理旧 cron 条目及残留进程 |
| `run_pi_cloud_client.sh` | 服务模式自动传递 `--wait-for-server` |
| `pi_cloud_client.py` | 新增 `--wait-for-server`（启动 health 失败时循环等待）；infer 错误 5s 速率限制 |

---

## 三、测试结果

### 环境

- 硬件：Raspberry Pi 5 (aarch64)
- 操作系统：Debian GNU/Linux 13 (trixie)
- 内核：Linux 6.18.29+rpt-rpi-v8
- systemd：257
- 用户：raspi123（sudo 需密码，因此全程使用 user 级服务）

### 已执行测试

| # | 测试项 | 方法 | 结果 |
|---|--------|------|------|
| 1 | 一键安装 | `./install_service.sh` | ✅ PASS，双服务 enable，无旧 cron 残留 |
| 2 | 服务运行状态 | `systemctl --user status` | ✅ PASS，主服务与看门狗均 active |
| 3 | `ps aux` 可见性 | `psaux \| grep -E 'pi_cloud_client\|pi_cloud_watchdog'` | ✅ PASS，看到 Python 主进程 + bash 看门狗进程 |
| 4 | kill -9 模拟崩溃 | `kill -9 $(MainPID)`，等待 12s | ✅ PASS，systemd 自动重启主服务 |
| 5 | 看门狗兜底拉起 | `systemctl --user stop pi-cloud-client`，等待 12s | ✅ PASS，看门狗检测到 inactive 并重启服务 |
| 6 | 云端掉线容忍 | 当前服务器不可达，主进程持续运行且日志不刷屏 | ✅ PASS，infer 错误每 5s 打印一次 |

### 现场最终验证（需手动执行）

    sudo reboot
    # 重新 SSH 登录后
    ps aux | grep -E 'pi_cloud_client|pi_cloud_watchdog' | grep -v grep
    systemctl --user status pi-cloud-client pi-cloud-client-watchdog

预期看到：
- `python3 pi_cloud_client.py ... --headless --wait-for-server`
- `bash /home/raspi123/Desktop/pi_cloud_deploy/pi_cloud_watchdog.sh`
- 两个服务状态均为 `active (running)`

---

## 四、使用说明

### 安装 / 重装

    cd ~/Desktop/pi_cloud_deploy
    ./install_service.sh

### 日常操作

    # 查看状态
    systemctl --user status pi-cloud-client pi-cloud-client-watchdog

    # 查看日志
    journalctl --user -u pi-cloud-client -f
    tail -f ~/Desktop/pi_cloud_deploy/logs/watchdog.log

    # 停止服务
    systemctl --user stop pi-cloud-client pi-cloud-client-watchdog

    # 禁用开机自启
    systemctl --user disable pi-cloud-client pi-cloud-client-watchdog

### 服务行为

| 场景 | 行为 |
|------|------|
| 树莓派通电开机 | 主服务 + 看门狗均自动启动 |
| Python 进程崩溃 / kill -9 | systemd 约 10s 内自动重启 |
| 手动 `systemctl stop` | 看门狗 10s 内检测并重新拉起 |
| 云端服务器掉线 | 服务不退出，持续重试；infer 错误每 5s 打印一次 |
| 开机时云端尚未就绪 | `run_pi_cloud_client.sh` 内重试最多 120s；成功后 Python 仍带 `--wait-for-server` |

---

## 五、设计决策记录

1. **为什么用独立看门狗 service 替代 cron？**
   - cron 最小粒度 1 分钟，不满足“每隔多少秒检测”需求。
   - cron 环境缺少 `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS`，在启动顺序、systemd user bus 未就绪时容易失效。
   - 独立 service 作为长期进程，`ps aux` 可见，便于现场检查。

2. **为什么 `StartLimitIntervalSec=0`？**
   - 摄像头、网络、云端均可能出现短暂中断，若 60s/7min 内失败几次就停止重试，会导致服务在真实演示场景中长时间挂掉。
   - 不设启动上限，配合指数退避的是脚本内部 health 重试，避免 CPU/日志风暴。

3. **为什么保留 user 级服务而非系统级？**
   - 当前用户 sudo 需要密码，脚本无法无人值守使用 sudo。
   - `loginctl enable-linger` 已启用，user 级服务足以实现开机自启。

4. **为什么主客户端增加 `--wait-for-server`？**
   - 避免“脚本启动时云端刚好不可达 → 进程退出 → systemd 重启”的循环。
   - 服务模式下持续等待云端恢复，符合“云端连接有时候会出岔子”的实际场景。

---

## 六、看门狗检查手册

以下命令可用于日常或现场验证看门狗是否正常工作。

### 1. 服务状态检查

    systemctl --user is-active pi-cloud-client pi-cloud-client-watchdog

预期输出：

    active
    active

### 2. 进程可见性检查

    ps aux | grep -E 'pi_cloud_client|pi_cloud_watchdog' | grep -v grep

预期看到 **两个进程**：

    raspi123  ...  python3 pi_cloud_client.py ... --headless --wait-for-server
    raspi123  ...  bash /home/raspi123/Desktop/pi_cloud_deploy/pi_cloud_watchdog.sh

### 3. 模拟崩溃：kill -9 主进程

    PID=$(systemctl --user show -p MainPID --value pi-cloud-client)
    echo "主进程 PID: $PID"
    kill -9 "$PID"
    sleep 12
    systemctl --user is-active pi-cloud-client
    ps aux | grep -E 'pi_cloud_client|pi_cloud_watchdog' | grep -v grep

预期：主服务恢复 active，Python 进程 PID 改变，watchdog 进程仍在。

### 4. 模拟手动停止

    systemctl --user stop pi-cloud-client
    sleep 12
    systemctl --user is-active pi-cloud-client
    ps aux | grep -E 'pi_cloud_client|pi_cloud_watchdog' | grep -v grep
    tail -n 10 /home/raspi123/Desktop/pi_cloud_deploy/logs/watchdog.log

预期：主服务被看门狗重新拉起，watchdog.log 中出现：

    [...] pi-cloud-client 未运行，尝试拉起...

### 5. 开机自启验证

    sudo reboot

重新 SSH 登录后：

    systemctl --user is-active pi-cloud-client pi-cloud-client-watchdog
    ps aux | grep -E 'pi_cloud_client|pi_cloud_watchdog' | grep -v grep

预期：两个服务均 active，两个进程均存在。

### 6. 日志排查

主服务日志：

    journalctl --user -u pi-cloud-client -n 50 --no-pager

看门狗日志：

    tail -n 30 /home/raspi123/Desktop/pi_cloud_deploy/logs/watchdog.log

### 7. 常见问题

| 现象 | 可能原因 | 处理 |
|------|---------|------|
| 两个服务都 inactive | 没有 enable 或 linger 没开 | `cd ~/Desktop/pi_cloud_deploy && ./install_service.sh` |
| 主服务 failed | 启动限制或配置错误 | `systemctl --user reset-failed pi-cloud-client && systemctl --user start pi-cloud-client` |
| 只有 Python 没有 watchdog | 看门狗服务没启用 | `systemctl --user enable --now pi-cloud-client-watchdog` |
| reboot 后不启动 | linger 未启用 | `loginctl enable-linger` |

### 8. 一键验证命令块

    # 状态
    systemctl --user status pi-cloud-client pi-cloud-client-watchdog --no-pager

    # 进程
    ps aux | grep -E 'pi_cloud_client|pi_cloud_watchdog' | grep -v grep

    # kill -9 测试
    PID=$(systemctl --user show -p MainPID --value pi-cloud-client)
    kill -9 "$PID"
    sleep 12
    echo "kill 后状态："
    systemctl --user is-active pi-cloud-client

    # stop 测试
    systemctl --user stop pi-cloud-client
    sleep 12
    echo "stop 后状态："
    systemctl --user is-active pi-cloud-client
    tail -n 5 ~/Desktop/pi_cloud_deploy/logs/watchdog.log
