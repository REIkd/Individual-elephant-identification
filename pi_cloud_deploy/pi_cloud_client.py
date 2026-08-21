"""
树莓派瘦客户端：本地摄像头采集 + 上传 JPEG 到云端推理 + 本地画框显示。
Pi 上无需安装 PyTorch / ultralytics。

示例:
  python pi_cloud_client.py --server http://192.168.1.100:8000 --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from pi_clip_recorder import ClipUploadWorker, PiClipRecorder
except ImportError:
    PiClipRecorder = None  # type: ignore[misc, assignment]
    ClipUploadWorker = None  # type: ignore[misc, assignment]

_CJK_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_cjk_font(size: int):
    if size in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[size]
    custom = os.environ.get("ELEPHANT_FONT", "").strip()
    if custom and Path(custom).is_file():
        try:
            f = ImageFont.truetype(custom, size)
            _CJK_FONT_CACHE[size] = f
            return f
        except OSError:
            pass
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            try:
                f = ImageFont.truetype(str(p), size)
                _CJK_FONT_CACHE[size] = f
                return f
            except OSError:
                continue
    f = ImageFont.load_default()
    _CJK_FONT_CACHE[size] = f
    return f


def draw_label(frame_bgr: np.ndarray, x: int, y: int, text: str, color_bgr, font_size=22):
    font = _get_cjk_font(font_size)
    pad_x, pad_y = 6, 4
    fill_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    scratch = Image.new("RGB", (4, 4))
    dr0 = ImageDraw.Draw(scratch)
    l, t, r, b = dr0.textbbox((0, 0), text, font=font)
    pw, ph = (r - l) + 2 * pad_x, (b - t) + 2 * pad_y
    pil = Image.new("RGB", (pw, ph), fill_rgb)
    dr = ImageDraw.Draw(pil)
    dr.text((pad_x - l, pad_y - t), text, font=font, fill=(255, 255, 255))
    y_top = int(y) - ph - 2
    if y_top < 0:
        y_top = int(y) + 4
    x_left = int(x) + 4
    fh, fw = frame_bgr.shape[:2]
    iy1, ix1 = max(0, y_top), max(0, x_left)
    iy2, ix2 = min(fh, y_top + ph), min(fw, x_left + pw)
    if iy2 <= iy1 or ix2 <= ix1:
        return
    patch = np.asarray(pil)[iy1 - y_top : iy2 - y_top, ix1 - x_left : ix2 - x_left]
    frame_bgr[iy1:iy2, ix1:ix2] = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)


def scale_bbox(bbox, sx: float, sy: float):
    x, y, w, h = bbox
    return [
        int(round(x * sx)),
        int(round(y * sy)),
        int(round(w * sx)),
        int(round(h * sy)),
    ]


class TrackSmoother:
    """在两次云端结果之间用速度外推框位置，减轻「框不跟手」。"""

    def __init__(self, max_extrap_sec: float = 0.35):
        self.max_extrap_sec = max_extrap_sec
        self._tracks: dict[int, dict] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, tracks: list[dict], cloud_t: float) -> None:
        seen: set[int] = set()
        for tr in tracks:
            tid = int(tr["track_id"])
            seen.add(tid)
            bbox = [int(v) for v in tr["bbox"]]
            prev = self._tracks.get(tid)
            vx = vy = vw = vh = 0.0
            if prev is not None:
                dt = cloud_t - float(prev["cloud_t"])
                if dt > 0.008:
                    pb = prev["cloud_bbox"]
                    vx = (bbox[0] - pb[0]) / dt
                    vy = (bbox[1] - pb[1]) / dt
                    vw = (bbox[2] - pb[2]) / dt
                    vh = (bbox[3] - pb[3]) / dt
            self._tracks[tid] = {
                "cloud_bbox": bbox,
                "cloud_t": cloud_t,
                "vx": vx,
                "vy": vy,
                "vw": vw,
                "vh": vh,
                "name": tr.get("name", "识别中..."),
                "color_bgr": tuple(tr.get("color_bgr", [80, 255, 100])),
            }

        stale = [
            tid
            for tid, info in self._tracks.items()
            if tid not in seen and cloud_t - float(info["cloud_t"]) > 0.45
        ]
        for tid in stale:
            del self._tracks[tid]

    def display(self, now_t: float, frame_w: int, frame_h: int) -> list[dict]:
        out: list[dict] = []
        for tid, info in self._tracks.items():
            dt = min(max(0.0, now_t - float(info["cloud_t"])), self.max_extrap_sec)
            x, y, w, h = info["cloud_bbox"]
            x = int(round(x + info["vx"] * dt))
            y = int(round(y + info["vy"] * dt))
            w = max(8, int(round(w + info["vw"] * dt)))
            h = max(8, int(round(h + info["vh"] * dt)))
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            out.append(
                {
                    "track_id": tid,
                    "bbox": [x, y, w, h],
                    "name": info["name"],
                    "color_bgr": info["color_bgr"],
                }
            )
        return out


def _parse_camera_arg(raw: str) -> int | str:
    s = str(raw).strip().lower()
    if s in {"auto", "scan"}:
        return "auto"
    if s.startswith("/dev/"):
        return s
    try:
        return int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"无效的 --camera: {raw!r}，请用 auto、数字 index 或 /dev/video0"
        ) from e


def _linux_video_devices() -> list[str]:
    if not sys.platform.startswith("linux"):
        return []
    out: list[str] = []
    for p in sorted(Path("/dev").glob("video*")):
        if not p.is_char_device():
            continue
        suffix = p.name[5:]
        if suffix.isdigit():
            out.append(str(p))
    return out


def _v4l2_backend() -> int:
    return getattr(cv2, "CAP_V4L2", 200)


def _device_path(device: int | str) -> str:
    """Linux V4L2 必须用设备路径，不能用数字 index。"""
    if isinstance(device, int):
        return f"/dev/video{device}"
    return str(device)


def _configure_capture(
    cap: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    use_mjpeg: bool = True,
) -> None:
    if use_mjpeg:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _read_test_frame(cap: cv2.VideoCapture, retries: int = 8) -> tuple[bool, str]:
    """isOpened 对 metadata 节点也会 True，必须 read 一帧才算成功。"""
    if not cap.isOpened():
        return False, "isOpened=False"
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            return True, f"{w}x{h}"
        time.sleep(0.05)
    return False, "read 失败（可能是 metadata 节点或分辨率不支持）"


def _open_one_device(
    device_path: str,
    *,
    width: int,
    height: int,
    use_mjpeg: bool,
) -> tuple[cv2.VideoCapture | None, str]:
    v4l2 = _v4l2_backend()

    # 1) V4L2：仅设备路径，禁止 VideoCapture(index, CAP_V4L2)
    cap = cv2.VideoCapture(device_path, v4l2)
    if cap.isOpened():
        _configure_capture(cap, width=width, height=height, use_mjpeg=use_mjpeg)
        ok, detail = _read_test_frame(cap)
        if ok:
            return cap, f"{device_path} V4L2 -> {detail}"
        cap.release()

    # 2) GStreamer 回退（仍用 device 路径）
    gst = getattr(cv2, "CAP_GSTREAMER", None)
    if gst is not None and sys.platform.startswith("linux"):
        size_tries = []
        for w, h in ((width, height), (1280, 720), (640, 480)):
            if w > 0 and h > 0 and (w, h) not in size_tries:
                size_tries.append((w, h))
        for w, h in size_tries:
            if use_mjpeg:
                pipe = (
                    f"v4l2src device={device_path} ! image/jpeg,width={w},height={h} ! "
                    "jpegdec ! videoconvert ! appsink drop=1 max-buffers=1"
                )
                cap = cv2.VideoCapture(pipe, gst)
                if cap.isOpened():
                    ok, detail = _read_test_frame(cap)
                    if ok:
                        return cap, f"{device_path} GStreamer MJPEG {w}x{h} -> {detail}"
                    cap.release()
            pipe = (
                f"v4l2src device={device_path} ! video/x-raw,width={w},height={h} ! "
                "videoconvert ! appsink drop=1 max-buffers=1"
            )
            cap = cv2.VideoCapture(pipe, gst)
            if cap.isOpened():
                ok, detail = _read_test_frame(cap)
                if ok:
                    return cap, f"{device_path} GStreamer RAW {w}x{h} -> {detail}"
                cap.release()

    return None, f"{device_path}: 无法采集（可能不是视频节点，试其它 /dev/video*）"


def open_video_capture(
    camera: int | str,
    *,
    width: int = 1280,
    height: int = 720,
    use_mjpeg: bool = True,
    auto_scan: bool = True,
) -> tuple[cv2.VideoCapture | None, str]:
    """打开摄像头并实测 read 一帧。Linux 上只用 /dev/video* 路径，不用 index。"""
    res_candidates: list[tuple[int, int, str]] = []
    for w, h in ((width, height), (1280, 720), (1920, 1080), (640, 480)):
        if w > 0 and h > 0 and (w, h) not in [(a, b) for a, b, _ in res_candidates]:
            res_candidates.append((w, h, f"{w}x{h}"))

    if camera == "auto":
        targets = _linux_video_devices()
    elif isinstance(camera, int):
        primary = _device_path(camera)
        targets = [primary]
        if auto_scan and sys.platform.startswith("linux"):
            targets.extend(d for d in _linux_video_devices() if d != primary)
    else:
        targets = [camera]
        if auto_scan and sys.platform.startswith("linux"):
            targets.extend(d for d in _linux_video_devices() if d != camera)

    if not targets:
        targets = ["/dev/video0", "/dev/video1", "/dev/video2"]

    errors: list[str] = []
    for device_path in targets:
        for w, h, tag in res_candidates:
            cap, msg = _open_one_device(
                device_path, width=w, height=h, use_mjpeg=use_mjpeg
            )
            if cap is not None:
                if device_path != _device_path(camera) and camera != "auto":
                    return cap, f"已自动切换: {msg}"
                return cap, msg
            errors.append(f"{msg} @ {tag}")

    return None, "\n  ".join(errors[:12])


def probe_cameras(width: int = 1280, height: int = 720) -> int:
    print(f"OpenCV {cv2.__version__} | platform={sys.platform}")
    devs = _linux_video_devices()
    if devs:
        print("发现设备:", ", ".join(devs))
    else:
        print("未发现 /dev/video*（摄像头未插入或驱动未加载？）")
    if sys.platform.startswith("linux"):
        print("建议同时运行: v4l2-ctl --list-devices")

    ok_any = False
    for dev in devs or ["/dev/video0", "/dev/video1", "/dev/video2"]:
        cap, msg = open_video_capture(
            dev, width=width, height=height, auto_scan=False
        )
        if cap is not None:
            ok_any = True
            print(f"[OK] {msg}")
            cap.release()
        else:
            print(f"[FAIL] {dev}\n  {msg}")
    if not ok_any:
        print("\n排查建议:")
        print("  1) sudo apt install -y v4l-utils")
        print("  2) groups  确认用户在 video 组")
        print("  3) 换 USB 口 / 带供电 Hub")
        print("  4) pi_cloud_config.sh: export CAMERA_DEVICE=auto")
        return 1
    return 0


class AsyncInferWorker:
    """后台上传推理，主循环只负责采集和显示，避免 infer 阻塞导致框滞后。"""

    def __init__(self, client: "CloudElephantClient"):
        self._client = client
        self._queue: Queue[bytes] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: tuple[float, dict] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    def submit(self, jpeg_bytes: bytes) -> None:
        try:
            self._queue.put_nowait(jpeg_bytes)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(jpeg_bytes)
            except Full:
                pass

    def pop_result(self) -> tuple[float, dict] | None:
        with self._lock:
            result = self._latest
            self._latest = None
            return result

    def _loop(self) -> None:
        while self._running:
            try:
                payload = self._queue.get(timeout=0.08)
            except Empty:
                continue
            resp = self._client.infer(payload)
            if resp:
                with self._lock:
                    self._latest = (time.perf_counter(), resp)


class CloudElephantClient:
    def __init__(
        self,
        server_url: str,
        api_key: str = "",
        timeout: float = 8.0,
        stream_id: str = "",
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._fixed_stream_id = (stream_id or "").strip()
        self.session_id = self._fixed_stream_id or uuid.uuid4().hex
        self.headers = {}
        if api_key:
            self.headers["X-Api-Key"] = api_key

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def infer(self, jpeg_bytes: bytes) -> dict | None:
        files = {"image": ("frame.jpg", jpeg_bytes, "image/jpeg")}
        data = {"session_id": self.session_id}
        try:
            r = requests.post(
                f"{self.server_url}/api/v1/infer",
                files=files,
                data=data,
                headers=self.headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
            body = r.json()
            if body.get("session_id") and not self._fixed_stream_id:
                self.session_id = body["session_id"]
            return body
        except requests.RequestException as e:
            detail = ""
            if getattr(e, "response", None) is not None:
                try:
                    detail = f" | HTTP {e.response.status_code}: {e.response.text[:200]}"
                except Exception:
                    pass
            print(f"[云端错误] {e}{detail}")
            return None

    def reset(self) -> None:
        try:
            requests.post(
                f"{self.server_url}/api/v1/reset",
                data={"session_id": self.session_id},
                headers=self.headers,
                timeout=3,
            )
        except requests.RequestException:
            pass
        if not self._fixed_stream_id:
            self.session_id = uuid.uuid4().hex


def main():
    parser = argparse.ArgumentParser(description="Pi 云端大象识别客户端")
    parser.add_argument(
        "--server",
        required=False,
        help="云端地址，如 http://192.168.1.100:8000",
    )
    parser.add_argument("--api-key", default=os.environ.get("CLOUD_API_KEY", ""))
    parser.add_argument(
        "--stream-id",
        default=os.environ.get("STREAM_ID", "elephant-live"),
        help="云端 session_id（仅用于识别会话，不再提供网页直播）",
    )
    parser.add_argument(
        "--camera",
        type=_parse_camera_arg,
        default=_parse_camera_arg(os.environ.get("CAMERA_DEVICE", "auto")),
        help="auto=扫描全部 /dev/video* | index | /dev/video0；环境变量 CAMERA_DEVICE",
    )
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument(
        "--upload-width",
        type=int,
        default=1280,
        help="上传前缩放到该宽度（PC 算力够可设 1280~1920，框会映射回本地显示分辨率）",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--send-interval",
        type=float,
        default=0.05,
        help="向云端发送间隔（秒），0.05≈20次/秒",
    )
    parser.add_argument(
        "--smooth-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="本地速度外推框位置，减轻网络/推理延迟造成的滞后",
    )
    parser.add_argument(
        "--sync-infer",
        action="store_true",
        help="同步推理（会阻塞画面；默认后台异步推理更跟手）",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--probe-camera",
        action="store_true",
        help="仅检测摄像头（不需连云端），测完退出",
    )
    parser.add_argument(
        "--no-camera-mjpeg",
        action="store_true",
        help="不请求 MJPEG（部分摄像头需关闭此项）",
    )
    parser.add_argument(
        "--clip-enable",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PI_CLIP_ENABLE", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        help="检测到大象时本地录制 1920 标注 MP4 并上传（默认开启）",
    )
    parser.add_argument(
        "--clip-dir",
        default=os.environ.get("PI_CLIP_DIR", "data/pi_clips"),
        help="Pi 本地录像目录",
    )
    parser.add_argument(
        "--clip-max-width",
        type=int,
        default=int(os.environ.get("PI_CLIP_MAX_WIDTH", "1920")),
        help="录像最大宽度，1920=全高清，0=不缩放",
    )
    parser.add_argument(
        "--clip-fps",
        type=float,
        default=float(os.environ.get("PI_CLIP_FPS", "15")),
        help="录像目标帧率",
    )
    parser.add_argument(
        "--clip-retention-days",
        type=float,
        default=float(os.environ.get("PI_CLIP_RETENTION_DAYS", "3")),
        help="本地录像保留天数，到期自动删除",
    )
    parser.add_argument(
        "--clip-delete-after-upload",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PI_CLIP_DELETE_AFTER_UPLOAD", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        help="上传成功后删除 Pi 本地 MP4",
    )
    args = parser.parse_args()

    if args.probe_camera:
        raise SystemExit(probe_cameras(args.camera_width, args.camera_height))

    if not args.server:
        parser.error("请指定 --server，或先用 --probe-camera 检测摄像头")

    client = CloudElephantClient(
        args.server,
        args.api_key,
        args.timeout,
        stream_id=args.stream_id,
    )
    if not client.health():
        print(f"无法连接云端: {args.server}")
        print("请确认 cloud_server.py 已启动，且 Pi 与服务器网络互通")
        return

    cap, cam_detail = open_video_capture(
        args.camera,
        width=args.camera_width,
        height=args.camera_height,
        use_mjpeg=not args.no_camera_mjpeg,
        auto_scan=True,
    )
    if cap is None:
        print("无法打开摄像头")
        print(cam_detail)
        print("\n请先运行诊断（无需云端）:")
        print("  python pi_cloud_client.py --probe-camera")
        print("  v4l2-ctl --list-devices")
        return

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Pi 客户端已连接: {args.server}")
    print(f"摄像头: {cam_detail}")
    print(f"本地采集: {aw}x{ah} | 上传宽: {args.upload_width} | 间隔: {args.send_interval}s")
    print("模式: 云端识别 + 本地录像上传（网页直播已关闭）")
    print(f"录像库: {args.server.rstrip('/')}/watch/clips")

    clip_recorder = None
    upload_worker = None
    if args.clip_enable and PiClipRecorder is not None and ClipUploadWorker is not None:
        upload_worker = ClipUploadWorker(
            args.server,
            args.api_key,
            delete_after_upload=args.clip_delete_after_upload,
        )
        upload_worker.start()

        def _on_clip(meta: dict, path: Path) -> None:
            if upload_worker is not None:
                upload_worker.enqueue(meta, path)

        clip_dir = Path(args.clip_dir)
        clip_recorder = PiClipRecorder(
            clip_dir,
            max_width=args.clip_max_width,
            target_fps=args.clip_fps,
            on_finalize=_on_clip,
        )
        purge = PiClipRecorder.purge_local_older_than(clip_dir, args.clip_retention_days)
        if purge.get("removed"):
            print(f"已清理本地过期录像 {purge['removed']} 条")
        print(
            f"本地录像: {clip_dir.resolve()} | {args.clip_max_width or aw}px | "
            f"保留 {args.clip_retention_days} 天"
        )
    elif args.clip_enable:
        print("[警告] 未找到 pi_clip_recorder.py，本地录像未启用")

    if not args.headless:
        print("按 q 退出, r 重置云端 session")

    smoother = TrackSmoother()
    worker = None if args.sync_infer else AsyncInferWorker(client)
    if worker is not None:
        worker.start()

    last_tracks: list[dict] = []
    last_cloud_ms = 0.0
    last_send_t = 0.0
    smooth_fps = 0.0
    prev_t = 0.0

    def apply_cloud_response(resp: dict, uh: int, uw: int, recv_t: float) -> None:
        nonlocal last_tracks, last_cloud_ms
        sx = uw / float(resp.get("frame_width", uw))
        sy = uh / float(resp.get("frame_height", uh))
        mapped = []
        for tr in resp.get("tracks", []):
            bb = scale_bbox(tr["bbox"], sx, sy)
            mapped.append(
                {
                    "track_id": tr["track_id"],
                    "bbox": bb,
                    "name": tr.get("name", "识别中..."),
                    "color_bgr": tuple(tr.get("color_bgr", [80, 255, 100])),
                }
            )
        last_tracks = mapped
        last_cloud_ms = float(resp.get("latency_ms", 0))
        if args.smooth_boxes:
            smoother.update(mapped, recv_t)

    last_purge_t = time.time()
    purge_interval_sec = 3600.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            if prev_t > 0:
                inst = 1.0 / max(now - prev_t, 1e-6)
                smooth_fps = 0.9 * smooth_fps + 0.1 * inst if smooth_fps > 0 else inst
            prev_t = now

            if worker is not None:
                pending = worker.pop_result()
                if pending is not None:
                    recv_t, resp = pending
                    uh, uw = frame.shape[:2]
                    apply_cloud_response(resp, uh, uw, recv_t)

            if now - last_send_t >= args.send_interval:
                last_send_t = now
                uh, uw = frame.shape[:2]
                uw_target = max(320, args.upload_width)
                scale = uw_target / float(uw)
                upload = cv2.resize(
                    frame,
                    (uw_target, max(1, int(round(uh * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
                ok, buf = cv2.imencode(
                    ".jpg",
                    upload,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                )
                if ok:
                    payload = buf.tobytes()
                    if worker is not None:
                        worker.submit(payload)
                    else:
                        resp = client.infer(payload)
                        if resp:
                            apply_cloud_response(resp, uh, uw, time.perf_counter())

            if args.smooth_boxes and smoother._tracks:
                fh, fw = frame.shape[:2]
                draw_tracks = smoother.display(now, fw, fh)
            else:
                draw_tracks = last_tracks

            for tr in draw_tracks:
                x, y, w, h = tr["bbox"]
                color = tr["color_bgr"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, x, y, tr["name"], color)

            if clip_recorder is not None:
                clip_recorder.on_frame(
                    client.session_id,
                    frame,
                    draw_tracks,
                    time.time(),
                )

            if (
                clip_recorder is not None
                and args.clip_retention_days > 0
                and now - last_purge_t >= purge_interval_sec
            ):
                last_purge_t = now
                purged = PiClipRecorder.purge_local_older_than(
                    args.clip_dir, args.clip_retention_days
                )
                if purged.get("removed"):
                    print(f"[清理] 删除本地过期录像 {purged['removed']} 条")

            info = (
                f"FPS:{smooth_fps:.0f} | Cloud:{last_cloud_ms:.0f}ms | "
                f"Trk:{len(draw_tracks)} | REC:{'ON' if clip_recorder else 'OFF'}"
            )
            cv2.putText(
                frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )

            if args.headless:
                if int(now) % 5 == 0 and int(now * 10) % 10 == 0:
                    print(info, draw_tracks)
            else:
                cv2.imshow("Pi Cloud Elephant", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    client.reset()
                    last_tracks = []
                    smoother.reset()
                    print("已重置云端 session")
    finally:
        if worker is not None:
            worker.stop()
        if upload_worker is not None:
            upload_worker.stop()

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
