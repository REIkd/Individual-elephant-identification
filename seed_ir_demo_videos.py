#!/usr/bin/env python3
"""
将本地演示 MP4（默认 Single1~4.mp4）按相机 01~04 的 IMEI 上传到识别服务器，
走 /servlet/original2 接口排队 GPU 识别，完成后可在 /watch/ir 总览观看。

用法:
  python seed_ir_demo_videos.py
  python seed_ir_demo_videos.py --server http://120.196.88.140:9998 --wait
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER = "http://120.196.88.140:9998"
DEFAULT_VIDEOS = [
    ("Single1.mp4", "863386077691243", "01"),
    ("Single2.mp4", "863386077682150", "02"),
    ("Single3.mp4", "863386077672052", "03"),
    ("Single4.mp4", "863386077686185", "04"),
]

SERVER_DOWN_HINT = """
服务器未响应（端口拒绝连接或超时）。请先在 GPU 服务器上启动 cloud_server：

  ssh root@120.196.88.140 -p 12222
  cd /root/elephant_cloud
  pkill -f cloud_server.py || true
  sleep 2
  nohup bash start_cloud_server_linux.sh > cloud.log 2>&1 &
  tail -f cloud.log

看到「模型预热完成」后，在本机再执行：
  .\\.venv\\Scripts\\python.exe seed_ir_demo_videos.py --server http://120.196.88.140:9998 --wait
"""


def check_server(server: str, timeout: float = 8.0) -> dict:
    url = f"{server.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"无法连接 {url}：{e}{SERVER_DOWN_HINT}") from e


def upload_video(server: str, video_path: Path, imei: str, file_id: int) -> dict:
    data = video_path.read_bytes()
    url = f"{server.rstrip('/')}/servlet/original2"
    headers = {
        "X-CameraCode": imei,
        "X-File-Id": str(file_id),
        "X-File-Size": str(len(data)),
        "X-Is-Hq": "2",
        "Content-Type": "video/mp4",
    }
    resp = requests.post(url, data=data, headers=headers, timeout=600)
    resp.raise_for_status()
    return resp.json()


def poll_status(server: str, imei: str, timeout_sec: int = 3600) -> dict:
    url = f"{server.rstrip('/')}/api/v1/uovision/ir/{imei}/status"
    t0 = time.time()
    last_pct = -1.0
    while time.time() - t0 < timeout_sec:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        st = r.json()
        status = st.get("status", "idle")
        pct = float(st.get("progress_pct") or 0)
        if pct != last_pct:
            print(f"    状态: {status}  进度: {pct:.1f}%")
            last_pct = pct
        if status == "ready":
            return st
        if status == "error":
            raise RuntimeError(st.get("error") or "处理失败")
        time.sleep(5)
    raise TimeoutError(f"等待 {imei} 识别超时（{timeout_sec}s）")


def main() -> int:
    parser = argparse.ArgumentParser(description="上传演示 MP4 到红外识别流水线")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--wait", action="store_true", help="等待每台识别完成后再传下一台")
    parser.add_argument("--file-id", type=int, default=9001, help="演示用 file_id 起始值")
    args = parser.parse_args()

    print(f"服务器: {args.server}")
    print(f"总览页: {args.server}/watch/ir\n")

    try:
        health = check_server(args.server)
        print(f"健康检查: {health.get('status', 'ok')} ({health.get('service', '')})\n")
    except ConnectionError as e:
        print(str(e), file=sys.stderr)
        return 2

    for idx, (name, imei, cam_id) in enumerate(DEFAULT_VIDEOS):
        path = ROOT / name
        if not path.is_file():
            print(f"[跳过] 未找到 {path}")
            continue
        file_id = args.file_id + idx
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[{cam_id}] 上传 {name} ({size_mb:.1f} MB) → IMEI {imei}")
        try:
            result = upload_video(args.server, path, imei, file_id)
            print(f"    已排队: {json.dumps(result, ensure_ascii=False)}")
        except Exception as e:
            print(f"    上传失败: {e}", file=sys.stderr)
            return 1

        if args.wait:
            print(f"    等待识别完成…")
            try:
                st = poll_status(args.server, imei)
                names = (st.get("elephant_names") or [])
                print(f"    完成 · 识别: {', '.join(names) or '—'}")
                print(f"    观看: {args.server}{st.get('watch_url', '')}\n")
            except Exception as e:
                print(f"    识别失败: {e}", file=sys.stderr)
                return 1

    if not args.wait:
        print("\n已全部提交排队。请打开总览页刷新查看进度:")
        print(f"  {args.server}/watch/ir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
