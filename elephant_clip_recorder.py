"""
Pi 直播流：检测到大象时自动在服务器端录制带标注 MP4，供事后查看。

Pi 仍只上传 JPEG；本模块在 cloud_inference.process_jpeg 里按帧缓冲并落盘。
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
from typing import Any

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


class ElephantClipRecorder:
    def __init__(
        self,
        data_dir: Path | str = "data/clips",
        *,
        pre_sec: float = 2.0,
        post_sec: float = 10.0,
        min_frames: int = 8,
        max_width: int = 854,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pre_sec = max(0.0, float(pre_sec))
        self.post_sec = max(1.0, float(post_sec))
        self.min_frames = max(3, int(min_frames))
        self.max_width = max(320, int(max_width))
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveClip] = {}
        self._pre: dict[str, deque[tuple[float, np.ndarray]]] = {}

    @classmethod
    def from_env(cls) -> ElephantClipRecorder | None:
        if os.environ.get("ELEPHANT_CLIP_ENABLE", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return None
        root = Path(os.environ.get("ELEPHANT_CLIP_DIR", "data/clips"))
        pre = float(os.environ.get("ELEPHANT_CLIP_PRE_SEC", "2"))
        post = float(os.environ.get("ELEPHANT_CLIP_POST_SEC", "10"))
        mw = int(os.environ.get("ELEPHANT_CLIP_MAX_WIDTH", "854"))
        return cls(data_dir=root, pre_sec=pre, post_sec=post, max_width=mw)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.max_width <= 0 or w <= self.max_width:
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
                    self._finalize_unlocked(active)
                    del self._active[session_id]

    def _finalize_unlocked(self, clip: _ActiveClip) -> None:
        if len(clip.frames) < self.min_frames:
            return
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
            return
        final_path = self._try_h264(raw_path)
        duration = clip.frames[-1][0] - clip.frames[0][0]
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
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_mp4(frames: list[tuple[float, np.ndarray]], out_path: Path) -> bool:
        if not frames:
            return False
        t0 = frames[0][0]
        t1 = frames[-1][0]
        duration = max(0.1, t1 - t0)
        fps = max(4.0, min(20.0, len(frames) / duration))
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
        if dst.is_file() and dst.stat().st_size > 0:
            try:
                src.unlink(missing_ok=True)
            except OSError:
                pass
            return dst
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
                timeout=120,
            )
            if dst.is_file() and dst.stat().st_size > 0:
                src.unlink(missing_ok=True)
                return dst
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass
        return src

    def list_clips(
        self,
        *,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not self.data_dir.is_dir():
            return items
        json_files = sorted(
            self.data_dir.glob("*/*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for meta_path in json_files:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if session_id and meta.get("session_id") != session_id:
                continue
            vid = meta_path.parent / str(meta.get("video_file") or "")
            if not vid.is_file():
                continue
            meta["watch_url"] = f"/watch/clips/{meta.get('clip_id')}"
            meta["video_url"] = f"/api/v1/clips/{meta.get('clip_id')}/video.mp4"
            items.append(meta)
            if len(items) >= max(1, min(limit, 200)):
                break
        return items

    def get_clip(self, clip_id: str) -> dict[str, Any] | None:
        for meta in self.list_clips(limit=500):
            if meta.get("clip_id") == clip_id:
                return meta
        return None

    def clip_video_path(self, clip_id: str) -> Path | None:
        meta = self.get_clip(clip_id)
        if not meta:
            return None
        p = Path(str(meta.get("video_path") or ""))
        return p if p.is_file() else None

    def import_uploaded_clip(
        self,
        video_bytes: bytes,
        *,
        session_id: str = "",
        elephant_names: list[str] | None = None,
        started_at: float | None = None,
        duration_sec: float = 0.0,
        source: str = "pi",
        device_id: str = "",
        frame_count: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> dict[str, Any]:
        """保存 Pi 上传的 MP4 与元数据，供 /watch/clips 浏览。"""
        if not video_bytes:
            raise ValueError("视频为空")
        ts = started_at or time.time()
        clip_id = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        ) + "_" + uuid.uuid4().hex[:8]
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")
        out_dir = self.data_dir / day
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / f"{clip_id}.mp4"
        raw_path.write_bytes(video_bytes)
        final_path = self._try_h264(raw_path)
        names = sorted({str(n).strip() for n in (elephant_names or []) if str(n).strip()})
        meta = {
            "clip_id": clip_id,
            "session_id": session_id or device_id or "pi",
            "started_at": _utc_iso(ts),
            "ended_at": _utc_iso(ts + max(0.1, float(duration_sec or 0))),
            "duration_sec": round(max(0.1, float(duration_sec or 0)), 2),
            "frame_count": int(frame_count or 0),
            "elephant_names": names,
            "video_file": final_path.name,
            "video_path": str(final_path.resolve()),
            "source": source,
            "device_id": device_id or None,
            "width": int(width or 0),
            "height": int(height or 0),
        }
        meta_path = out_dir / f"{clip_id}.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        meta["watch_url"] = f"/watch/clips/{clip_id}"
        meta["video_url"] = f"/api/v1/clips/{clip_id}/video.mp4"
        return meta

    def purge_older_than(self, days: float = 3.0) -> dict[str, Any]:
        """删除超过 retention 天的片段（mp4 + json）。"""
        if days <= 0:
            return {"removed": 0, "days": days}
        cutoff = time.time() - float(days) * 86400.0
        removed = 0
        freed = 0
        if not self.data_dir.is_dir():
            return {"removed": 0, "days": days, "freed_bytes": 0}
        for json_path in list(self.data_dir.glob("*/*.json")):
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
            vid_name = str(meta.get("video_file") or "")
            vid_path = json_path.parent / vid_name if vid_name else None
            for p in (vid_path, json_path):
                if p is not None and p.is_file():
                    try:
                        freed += p.stat().st_size
                        p.unlink()
                    except OSError:
                        pass
            removed += 1
        return {"removed": removed, "days": days, "freed_bytes": freed}
