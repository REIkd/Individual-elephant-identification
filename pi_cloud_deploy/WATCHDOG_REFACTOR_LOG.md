# Pi Cloud Client 看门狗重构日志

**作者**：孙子恒  
**邮箱**：13533742541@163.com  
**微信**：Sunmichael090325  
**日期**：2026-06-24  
**Commit**：`feature/auto-start-watchdog` 分支最新提交，以 `git log` 为准

---

## 1. 重构背景

上一个看门狗实现基于 `cron` 每分钟调用一次 `pi_cloud_watchdog.sh`。在最终测试中发现以下问题：

1. **检测粒度不足**：cron 最小调度粒度为 1 分钟，无法满足“每隔多少秒检测一次”的需求。
2. **环境变量脆弱**：脚本里手动设置的 `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` 在 systemd user bus 未就绪或启动顺序异常时会失效，导致看门狗无法操作 user service。
3. **无独立可见进程**：现场检查时 `ps aux` 只能看到业务 Python 进程，没有一个清晰的看门狗守护进程。
4. **启动风暴风险**：systemd 服务配置了 `StartLimitIntervalSec=420 StartLimitBurst=3`，云端/摄像头短暂不可达时容易进入 failed 状态。
5. **主客户端对断连敏感**：`pi_cloud_client.py` 启动时 health 检查失败直接退出，云端抖动会触发反复重启。

---

## 2. 重构目标

1. 插电开机自启动（断电重启）。
2. 每固定秒数检测主服务，未运行则自动拉起。
3. 看门狗机制简单、可落地、`ps aux` 可见。
4. 对云端服务器连接抖动具备容忍能力。
5. 本 commit 不引入日志 rotation，留待后续 commit 实现。

---

## 3. 最终架构

    systemd user instance
    ├── pi-cloud-client.service        (主业务)
    │   ├── Type=simple
    │   ├── Restart=always
    │   ├── RestartSec=10
    │   └── StartLimitIntervalSec=0    # 持续重试，不设上限
    │
    └── pi-cloud-client-watchdog.service  (看门狗)
        ├── ExecStart=pi_cloud_watchdog.sh
        ├── Restart=always
        └── 每 10s 轮询主服务状态

    两个服务均 enable → default.target
    配合 loginctl enable-linger 实现开机无需登录自启

---

## 4. 文件级变更详情

### 4.1 `pi-cloud-client.service`

- `Type` 从 `exec` 改为 `simple`，提升兼容性。
- 移除 `StartLimitIntervalSec` / `StartLimitBurst` 限制，改为 `StartLimitIntervalSec=0`。
- 保留 `After=network-online.target` 与 `Wants=network-online.target`。

### 4.2 `pi-cloud-client-watchdog.service`（新增）

- 独立 user service。
- `WorkingDirectory` 指向项目目录。
- `Restart=always`，自身失败也能被 systemd 拉起。

### 4.3 `pi_cloud_watchdog.sh`

- 由单次 cron 脚本改为长期运行循环。
- 写死检测间隔 **10 秒**。
- 检测逻辑：`systemctl --user is-active --quiet pi-cloud-client` 失败时，先 `reset-failed` 再 `start`。
- 输出同时写入 `logs/watchdog.log` 与 stdout（journald），便于双通道排查。
- 兼容手动执行时补全 `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS`。

### 4.4 `install_service.sh`

- 安装并启用主服务 + 看门狗服务。
- 启动新服务前，先 `pkill -f pi_cloud_watchdog.sh` 并移除旧 cron 条目，避免旧 cron 遗留进程与新 service 看门狗并存。
- 安装完成后输出状态/日志/停止/禁用命令。

### 4.5 `run_pi_cloud_client.sh`

- 检测到 `--service` 参数时，追加 `--wait-for-server` 传给 Python。

### 4.6 `pi_cloud_client.py`

- 新增 `--wait-for-server` 参数：启动 health 检查失败时不退出，循环等待。
- 新增 `--health-retry-interval` 参数（默认 5s）控制等待间隔。
- `infer()` 中对 `requests.RequestException` 做 5 秒速率限制，避免云端掉线时 journald 被刷屏。

### 4.7 `WATCHDOG_DELIVERY_REPORT.md`（由 `DELIVERY_REPORT.md` 重命名）

- 重写为本次重构的交付报告，包含架构、测试结果、使用说明、设计决策。

### 4.8 `.gitignore`（新增）

- 排除运行时目录：`.venv/`、`.vaenv/`、`logs/`、`__pycache__/`。

---

## 5. 测试结果

### 测试环境

- 硬件：Raspberry Pi 5 (aarch64)
- OS：Debian GNU/Linux 13 (trixie)
- systemd：257
- 用户：raspi123（sudo 需密码，故使用 user 级服务）

### 已执行用例

| 编号 | 用例 | 方法 | 结果 |
|------|------|------|------|
| T01 | 一键安装 | `./install_service.sh` | 通过 |
| T02 | 双服务 active | `systemctl --user is-active` | 通过 |
| T03 | `ps aux` 可见性 | `ps aux \| grep -E 'pi_cloud_client\|pi_cloud_watchdog'` | 通过 |
| T04 | kill -9 恢复 | `kill -9 $(MainPID)`，等待 12s | 通过，systemd 约 10s 内重启 |
| T05 | 看门狗兜底 | `systemctl --user stop pi-cloud-client`，等待 12s | 通过，10s 内被看门狗拉起 |
| T06 | 云端掉线容忍 | 服务器不可达场景下观察主进程 | 通过，进程未退出，日志 5s 一次 |

### 待现场验证

    sudo reboot
    # 重新登录后
    ps aux | grep -E 'pi_cloud_client|pi_cloud_watchdog' | grep -v grep
    systemctl --user status pi-cloud-client pi-cloud-client-watchdog

---

## 6. 关键设计决策

1. **为什么不用 cron？**
   - cron 最小 1 分钟，不满足秒级检测。
   - cron 环境变量依赖 systemd user bus，启动阶段容易失效。
   - 独立 service 在 `ps aux` 中可见，便于现场检查。

2. **为什么取消 StartLimit？**
   - 摄像头、网络、云端均可能短暂中断。
   - 旧的 3 次/7 分钟限制会导致真实演示场景中服务长时间挂掉。
   - 不设上限，配合脚本内部 health 重试与错误速率限制，避免资源风暴。

3. **为什么保留 user 级服务？**
   - 当前用户 sudo 需要密码，脚本无法无人值守调用 sudo。
   - `loginctl enable-linger` 已启用，user 级服务已能满足开机自启。

4. **为什么主客户端增加 `--wait-for-server`？**
   - 防止“启动时云端不可达 → 退出 → systemd 重启”的循环。
   - 符合用户反馈的“服务器连接有时候会出岔子”场景。

---

## 7. 已知限制与下一步

1. **日志 rotation**：当前 `logs/watchdog.log` 为简单追加，无 rotation。长期运行可能缓慢增长，下一 commit 实现按大小/时间轮转。
2. **系统级服务**：若后续需要更高可靠性，可迁移到 `/etc/systemd/system/` 级服务，但需要用户现场输入 sudo 密码完成安装。
3. **运行时日志清理**：`logs/watchdog.log` 已加入 `.gitignore`，不会被提交。

---

## 8. 附录：常用命令

    # 查看状态
    systemctl --user status pi-cloud-client pi-cloud-client-watchdog

    # 查看日志
    journalctl --user -u pi-cloud-client -f
    tail -f ~/Desktop/pi_cloud_deploy/logs/watchdog.log

    # 停止服务
    systemctl --user stop pi-cloud-client pi-cloud-client-watchdog

    # 禁用开机自启
    systemctl --user disable pi-cloud-client pi-cloud-client-watchdog
