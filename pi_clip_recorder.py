"""
树莓派本地录像：检测到大象时用全分辨率（1920）帧写入 MP4，完成后上传云端。

与 elephant_clip_recorder 逻辑一致，但：
- 按本地采集帧率缓冲（约 15 FPS），不依赖 MJPEG 直播
- 默认不缩放（max_width=0 表示保持原分辨率）
- 支持 finalize 回调与本地过期清理
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


def _utc_iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).isoformat()


def _meaningful_names(tracks: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for tr in tracks:
        name = str(tr.get("name") or "").strip()
        if not name or name in ("识别中...", "未知"):
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


@dataclass
class _ActiveClip:
    session_id: str
    started_at: float
    last_hit: float
    frames: list[tuple[float, np.ndarray]] = field(default_factory=list)
    names: set[str] = field(default_factory=set)

    def add_frame(self, ts: float, frame: np.ndarray, tracks: list[dict]) -> None:
        self.frames.append((ts, frame.copy()))
        for n in _meaningful_names(tracks):
            self.names.add(n)
        if tracks:
            self.last_hit = ts


class PiClipRecorder:
    def __init__(
        self,
        data_dir: Path | str = "data/pi_clips",
        *,
        pre_sec: float = 2.0,
        post_sec: float = 10.0,
        min_frames: int = 8,
        max_width: int = 0,
        target_fps: float = 15.0,
        on_finalize: Callable[[dict[str, Any], Path], None] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pre_sec = max(0.0, float(pre_sec))
        self.post_sec = max(1.0, float(post_sec))
        self.min_frames = max(3, int(min_frames))
        self.max_width = int(max_width)
        self.target_fps = max(4.0, min(30.0, float(target_fps)))
        self.on_finalize = on_finalize
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveClip] = {}
        self._pre: dict[str, deque[tuple[float, np.ndarray]]] = {}

    @classmethod
    def from_env(
        cls,
        on_finalize: Callable[[dict[str, Any], Path], None] | None = None,
    ) -> PiClipRecorder | None:
        if os.environ.get("PI_CLIP_ENABLE", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return None
        root = Path(os.environ.get("PI_CLIP_DIR", "data/pi_clips"))
        pre = float(os.environ.get("PI_CLIP_PRE_SEC", "2"))
        post = float(os.environ.get("PI_CLIP_POST_SEC", "10"))
        mw = int(os.environ.get("PI_CLIP_MAX_WIDTH", "1920"))
        fps = float(os.environ.get("PI_CLIP_FPS", "15"))
        return cls(
            data_dir=root,
            pre_sec=pre,
            post_sec=post,
            max_width=mw,
            target_fps=fps,
            on_finalize=on_finalize,
        )

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.max_width <= 0:
            return frame
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / float(w)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def on_frame(
        self,
        session_id: str,
        annotated_bgr: np.ndarray,
        tracks: list[dict],
        timestamp: float | None = None,
    ) -> None:
        ts = timestamp or time.time()
        frame = self._resize(annotated_bgr)
        has_elephant = len(tracks) > 0
        finalize_meta: dict[str, Any] | None = None
        finalize_path: Path | None = None
        with self._lock:
            pre = self._pre.setdefault(session_id, deque())
            pre.append((ts, frame.copy()))
            while pre and ts - pre[0][0] > self.pre_sec:
                pre.popleft()

            active = self._active.get(session_id)
            if has_elephant:
                if active is None:
                    active = _ActiveClip(session_id=session_id, started_at=ts, last_hit=ts)
                    for pt, pf in pre:
                        active.add_frame(pt, pf, tracks if pt == ts else [])
                    self._active[session_id] = active
                else:
                    active.add_frame(ts, frame, tracks)
            elif active is not None:
                active.add_frame(ts, frame, tracks)
                if ts - active.last_hit >= self.post_sec:
                    finalize_meta, finalize_path = self._finalize_unlocked(active)
                    del self._active[session_id]

        if finalize_meta and finalize_path and self.on_finalize:
            try:
                self.on_finalize(finalize_meta, finalize_path)
            except Exception as e:
                print(f"[录像] finalize 回调失败: {e}")

    def _finalize_unlocked(self, clip: _ActiveClip) -> tuple[dict[str, Any] | None, Path | None]:
        if len(clip.frames) < self.min_frames:
            return None, None
        clip_id = datetime.fromtimestamp(clip.started_at, tz=timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        ) + "_" + uuid.uuid4().hex[:8]
        day = datetime.fromtimestamp(clip.started_at, tz=timezone.utc).strftime("%Y%m%d")
        out_dir = self.data_dir / day
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / f"{clip_id}.mp4"
        meta_path = out_dir / f"{clip_id}.json"
        ok = self._write_mp4(clip.frames, raw_path)
        if not ok:
            return None, None
        final_path = self._try_h264(raw_path)
        duration = clip.frames[-1][0] - clip.frames[0][0]
        h, w = clip.frames[0][1].shape[:2]
        meta = {
            "clip_id": clip_id,
            "session_id": clip.session_id,
            "started_at": _utc_iso(clip.started_at),
            "ended_at": _utc_iso(clip.frames[-1][0]),
            "duration_sec": round(max(0.1, duration), 2),
            "frame_count": len(clip.frames),
            "elephant_names": sorted(clip.names),
            "video_file": final_path.name,
            "video_path": str(final_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "width": w,
            "height": h,
            "source": "pi",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta, final_path

    def _write_mp4(self, frames: list[tuple[float, np.ndarray]], out_path: Path) -> bool:
        if not frames:
            return False
        t0 = frames[0][0]
        t1 = frames[-1][0]
        duration = max(0.1, t1 - t0)
        fps = max(4.0, min(30.0, len(frames) / duration))
        if self.target_fps > 0:
            fps = min(fps, self.target_fps)
        h, w = frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            return False
        try:
            for _, fr in frames:
                if fr.shape[0] != h or fr.shape[1] != w:
                    fr = cv2.resize(fr, (w, h))
                writer.write(fr)
        finally:
            writer.release()
        return out_path.is_file() and out_path.stat().st_size > 0

    @staticmethod
    def _try_h264(src: Path) -> Path:
        dst = src.with_name(src.stem + "_h264.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(dst),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
            if dst.is_file() and dst.stat().st_size > 0:
                src.unlink(missing_ok=True)
                return dst
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass
        return src

    @staticmethod
    def purge_local_older_than(data_dir: Path | str, days: float = 3.0) -> dict[str, Any]:
        root = Path(data_dir)
        if days <= 0 or not root.is_dir():
            return {"removed": 0, "days": days, "freed_bytes": 0}
        cutoff = time.time() - float(days) * 86400.0
        removed = 0
        freed = 0
        for json_path in list(root.glob("*/*.json")):
            try:
                mtime = json_path.stat().st_mtime
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                mtime = json_path.stat().st_mtime if json_path.is_file() else 0
                meta = {}
            started = meta.get("started_at") or ""
            ts = mtime
            if started:
                try:
                    ts = datetime.fromisoformat(str(started).replace("Z", "+00:00")).timestamp()
                except ValueError:
                    ts = mtime
            if ts >= cutoff:
                continue
            vid = meta.get("video_file")
            for p in (json_path.parent / str(vid) if vid else None, json_path):
                if p is not None and p.is_file():
                    try:
                        freed += p.stat().st_size
                        p.unlink()
                    except OSError:
                        pass
            removed += 1
        return {"removed": removed, "days": days, "freed_bytes": freed}


class ClipUploadWorker:
    """后台上传 Pi 本地 MP4 到云端 /api/v1/clips/upload。"""

    def __init__(
        self,
        server_url: str,
        api_key: str = "",
        *,
        delete_after_upload: bool = True,
        timeout: float = 120.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.headers = {}
        if api_key:
            self.headers["X-Api-Key"] = api_key
        self.delete_after_upload = delete_after_upload
        self.timeout = timeout
        self._queue: deque[tuple[dict[str, Any], Path]] = deque()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def enqueue(self, meta: dict[str, Any], video_path: Path) -> None:
        with self._lock:
            self._queue.append((meta, video_path))

    def _loop(self) -> None:
        import requests

        while self._running:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.popleft()
            if item is None:
                time.sleep(0.2)
                continue
            meta, path = item
            self._upload_one(requests, meta, path)

    def _upload_one(self, requests_mod, meta: dict[str, Any], path: Path) -> None:
        if not path.is_file():
            return
        names = meta.get("elephant_names") or []
        payload = {
            "session_id": meta.get("session_id") or "",
            "elephant_names": names,
            "started_at": meta.get("started_at") or "",
            "duration_sec": meta.get("duration_sec") or 0,
            "frame_count": meta.get("frame_count") or 0,
            "width": meta.get("width") or 0,
            "height": meta.get("height") or 0,
            "source": "pi",
            "clip_id": meta.get("clip_id") or "",
        }
        try:
            with path.open("rb") as fh:
                files = {"video": (path.name, fh, "video/mp4")}
                data = {"metadata": json.dumps(payload, ensure_ascii=False)}
                r = requests_mod.post(
                    f"{self.server_url}/api/v1/clips/upload",
                    files=files,
                    data=data,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            r.raise_for_status()
            body = r.json()
            print(
                f"[上传] 成功 clip={body.get('clip_id')} "
                f"象名={','.join(names) or '—'} → {body.get('watch_url', '')}"
            )
            if self.delete_after_upload:
                meta_path = Path(str(meta.get("meta_path") or ""))
                for p in (path, meta_path):
                    try:
                        if p.is_file():
                            p.unlink()
                    except OSError:
                        pass
        except Exception as e:
            print(f"[上传] 失败 {path.name}: {e}")
