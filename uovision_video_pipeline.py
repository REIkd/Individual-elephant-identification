"""
UOVision 原视频：上传 → GPU 识别打框 → /watch/ir-{IMEI} 播放最新一段。

与 Pi 实时 MJPEG 分离：红外相机走 MP4 点播（清晰流畅，延迟为处理时间）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from uovision_camera import UOVisionConfig, load_uovision_config


def ir_stream_id(camera_code: str) -> str:
    code = "".join(c for c in str(camera_code).strip() if c.isalnum())
    return f"ir-{code or 'unknown'}"


def parse_ir_stream_id(stream_id: str) -> str | None:
    s = (stream_id or "").strip()
    if not s.lower().startswith("ir-"):
        return None
    return s[3:].strip() or None


def default_ir_history_limit() -> int:
    """每台红外相机网页保留最近几条识别视频（默认 3）。"""
    raw = os.environ.get("UOVISION_IR_HISTORY_LIMIT", "3")
    try:
        n = int(raw)
    except ValueError:
        n = 3
    return max(1, min(n, 20))


@dataclass
class IrCameraState:
    camera_code: str
    status: str = "idle"  # idle | queued | processing | ready | error
    file_id: int | None = None
    source_path: str = ""
    output_path: str = ""
    watch_url: str = ""
    error: str = ""
    progress_pct: float = 0.0
    frames_done: int = 0
    frames_total: int = 0
    updated_at: str = ""
    upload_route: str = ""  # original | original2
    meta: dict[str, Any] = field(default_factory=dict)


class UOVisionVideoPipeline:
    """按相机维护「最新一段」识别视频，后台单线程处理避免 GPU 争抢。"""

    def __init__(
        self,
        process_fn: Callable[[str, str, Callable[[float, int, int], None]], dict],
        cfg: UOVisionConfig | None = None,
    ):
        self._cfg = cfg or load_uovision_config()
        self._process_fn = process_fn
        self._lock = threading.Lock()
        self._states: dict[str, IrCameraState] = {}
        self._queue: list[dict[str, Any]] = []
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _processed_dir(self, camera_code: str) -> Path:
        return self._cfg.data_dir / "processed" / camera_code.strip()

    def _load_persisted(self, camera_code: str) -> IrCameraState | None:
        meta_path = self._processed_dir(camera_code) / "latest.json"
        if not meta_path.is_file():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        st = IrCameraState(camera_code=camera_code.strip())
        st.status = str(data.get("status", "ready"))
        st.file_id = data.get("file_id")
        st.source_path = str(data.get("source_path", ""))
        st.output_path = str(data.get("output_path", ""))
        st.watch_url = str(data.get("watch_url", ""))
        st.error = str(data.get("error", ""))
        st.progress_pct = float(data.get("progress_pct", 100.0))
        st.updated_at = str(data.get("updated_at", ""))
        st.upload_route = str(data.get("upload_route", ""))
        st.meta = data.get("meta") or {}
        out = Path(st.output_path)
        if st.status == "ready" and (not out.is_file()):
            st.status = "idle"
        return st

    def _persist(self, st: IrCameraState) -> None:
        out_dir = self._processed_dir(st.camera_code)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "camera_code": st.camera_code,
            "status": st.status,
            "file_id": st.file_id,
            "source_path": st.source_path,
            "output_path": st.output_path,
            "watch_url": st.watch_url,
            "error": st.error,
            "progress_pct": st.progress_pct,
            "frames_done": st.frames_done,
            "frames_total": st.frames_total,
            "updated_at": st.updated_at,
            "upload_route": st.upload_route,
            "meta": st.meta,
        }
        (out_dir / "latest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_state(self, camera_code: str) -> IrCameraState:
        code = camera_code.strip()
        with self._lock:
            st = self._states.get(code)
            if st is None:
                st = self._load_persisted(code)
                if st is None:
                    st = IrCameraState(camera_code=code)
                    st.watch_url = f"/watch/{ir_stream_id(code)}"
                self._states[code] = st
            return st

    def reload_state(self, camera_code: str) -> None:
        """清理磁盘后刷新内存状态（避免网页仍显示已删除的视频）。"""
        code = camera_code.strip()
        with self._lock:
            st = self._load_persisted(code)
            if st is None:
                st = IrCameraState(camera_code=code)
                st.watch_url = f"/watch/{ir_stream_id(code)}"
            self._states[code] = st

    def reload_all_states(self) -> None:
        processed = self._cfg.data_dir / "processed"
        if not processed.is_dir():
            return
        for cam_dir in processed.iterdir():
            if cam_dir.is_dir():
                self.reload_state(cam_dir.name)

    def _video_meta_path(self, out_dir: Path, file_id: int) -> Path:
        return out_dir / f"{file_id}.json"

    def _load_video_meta(self, out_dir: Path, file_id: int) -> dict[str, Any]:
        path = self._video_meta_path(out_dir, file_id)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_video_meta(
        self,
        out_dir: Path,
        *,
        file_id: int,
        camera_code: str,
        updated_at: str,
        video_file: str,
        source_path: str,
        process: dict[str, Any],
    ) -> None:
        payload = {
            "file_id": file_id,
            "camera_code": camera_code,
            "updated_at": updated_at,
            "video_file": video_file,
            "source_path": source_path,
            "elephant_names": (process or {}).get("elephant_names") or [],
            "process": process or {},
        }
        self._video_meta_path(out_dir, file_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _video_entry(
        self,
        file_id: int,
        path: Path,
        meta: dict[str, Any],
        camera_code: str,
    ) -> dict[str, Any]:
        names = meta.get("elephant_names") or []
        proc = meta.get("process") or {}
        if not names and isinstance(proc, dict):
            names = proc.get("elephant_names") or []
        return {
            "file_id": file_id,
            "updated_at": meta.get("updated_at") or "",
            "elephant_names": list(names) if isinstance(names, list) else [],
            "video_url": f"/api/v1/uovision/ir/{camera_code}/videos/{file_id}.mp4",
            "duration_sec": proc.get("duration_sec"),
        }

    def list_videos(self, camera_code: str, limit: int | None = None) -> list[dict[str, Any]]:
        """按 file_id 降序返回最近若干条已识别视频。"""
        code = camera_code.strip()
        lim = max(1, min(int(limit if limit is not None else default_ir_history_limit()), 500))
        out_dir = self._processed_dir(code)
        if not out_dir.is_dir():
            return []

        by_id: dict[int, dict[str, Any]] = {}
        for web in out_dir.glob("*_web.mp4"):
            stem = web.stem
            if not stem.endswith("_web"):
                continue
            try:
                file_id = int(stem[: -len("_web")])
            except ValueError:
                continue
            meta = self._load_video_meta(out_dir, file_id)
            by_id[file_id] = self._video_entry(file_id, web, meta, code)

        for ann in out_dir.glob("*_annotated.mp4"):
            stem = ann.stem
            if not stem.endswith("_annotated"):
                continue
            try:
                file_id = int(stem[: -len("_annotated")])
            except ValueError:
                continue
            if file_id in by_id:
                continue
            meta = self._load_video_meta(out_dir, file_id)
            by_id[file_id] = self._video_entry(file_id, ann, meta, code)

        latest = out_dir / "latest.mp4"
        latest_json = out_dir / "latest.json"
        if latest.is_file() and latest_json.is_file():
            try:
                state = json.loads(latest_json.read_text(encoding="utf-8"))
                file_id = int(state.get("file_id") or 0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                file_id = 0
            if file_id and file_id not in by_id:
                proc = (state.get("meta") or {}).get("process") or {}
                meta = {
                    "updated_at": state.get("updated_at") or "",
                    "elephant_names": proc.get("elephant_names") or [],
                    "process": proc,
                }
                by_id[file_id] = self._video_entry(file_id, latest, meta, code)

        items = sorted(by_id.values(), key=lambda x: int(x.get("file_id") or 0), reverse=True)
        return items[:lim]

    def get_video_path(self, camera_code: str, file_id: int) -> Path | None:
        code = camera_code.strip()
        fid = int(file_id)
        out_dir = self._processed_dir(code)
        for name in (f"{fid}_web.mp4", f"{fid}_annotated.mp4"):
            p = out_dir / name
            if p.is_file():
                return p
        st = self.get_state(code)
        if st.file_id == fid:
            p = self.get_latest_video_path(code)
            if p is not None:
                return p
        return None

    def _trim_history(self, camera_code: str, keep: int | None = None) -> None:
        keep_n = int(keep if keep is not None else default_ir_history_limit())
        keep_n = max(1, min(keep_n, 20))
        code = camera_code.strip()
        out_dir = self._processed_dir(code)
        if not out_dir.is_dir():
            return
        videos = self.list_videos(code, limit=999)
        for item in videos[keep_n:]:
            fid = int(item.get("file_id") or 0)
            if fid <= 0:
                continue
            for name in (
                f"{fid}_web.mp4",
                f"{fid}_annotated.mp4",
                f"{fid}.json",
                f"{fid}_web.h264.mp4",
            ):
                p = out_dir / name
                if p.is_file():
                    p.unlink(missing_ok=True)

    def get_latest_video_path(self, camera_code: str) -> Path | None:
        st = self.get_state(camera_code)
        if st.status != "ready" or not st.output_path:
            p = self._processed_dir(camera_code) / "latest.mp4"
            return p if p.is_file() else None
        p = Path(st.output_path)
        return p if p.is_file() else None

    def enqueue(
        self,
        *,
        camera_code: str,
        source_path: str | Path,
        file_id: int,
        upload_route: str = "original",
    ) -> dict[str, Any]:
        code = camera_code.strip()
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(str(src))

        st = self.get_state(code)
        st.status = "queued"
        st.file_id = int(file_id)
        st.source_path = str(src.resolve())
        st.output_path = ""
        st.error = ""
        st.progress_pct = 0.0
        st.frames_done = 0
        st.frames_total = 0
        st.upload_route = upload_route
        st.watch_url = f"/watch/{ir_stream_id(code)}"
        st.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist(st)

        job = {
            "camera_code": code,
            "source_path": str(src.resolve()),
            "file_id": int(file_id),
            "upload_route": upload_route,
            "enqueued_at": time.time(),
        }
        with self._lock:
            self._queue = [j for j in self._queue if j["camera_code"] != code]
            self._queue.append(job)
        return {
            "ok": True,
            "camera_code": code,
            "file_id": file_id,
            "status": "queued",
            "watch_url": st.watch_url,
            "stream_id": ir_stream_id(code),
        }

    def _worker_loop(self) -> None:
        while True:
            job: dict[str, Any] | None = None
            with self._lock:
                if self._queue:
                    job = self._queue.pop(0)
            if job is None:
                time.sleep(0.4)
                continue
            try:
                self._run_job(job)
            except Exception as e:
                st = self.get_state(job["camera_code"])
                st.status = "error"
                st.error = str(e)
                st.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist(st)

    def _run_job(self, job: dict[str, Any]) -> None:
        code = job["camera_code"]
        file_id = int(job["file_id"])
        src = Path(job["source_path"])
        st = self.get_state(code)
        st.status = "processing"
        st.progress_pct = 0.0
        st.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist(st)

        out_dir = self._processed_dir(code)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_out = out_dir / f"{file_id}_annotated.mp4"
        web_out = out_dir / "latest.mp4"

        def progress(pct: float, done: int, total: int) -> None:
            st.progress_pct = round(pct, 1)
            st.frames_done = done
            st.frames_total = total
            st.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist(st)

        result = self._process_fn(str(src), str(raw_out), progress)
        web_copy = out_dir / f"{file_id}_web.mp4"
        final_path = self._finalize_web_video(raw_out, web_out)
        shutil.copy2(final_path, web_copy)

        st.status = "ready"
        st.output_path = str(final_path.resolve())
        st.progress_pct = 100.0
        st.error = ""
        st.meta = {
            "process": result,
            "file_id": file_id,
            "source_path": str(src),
        }
        st.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist(st)
        self._save_video_meta(
            out_dir,
            file_id=file_id,
            camera_code=code,
            updated_at=st.updated_at,
            video_file=web_copy.name,
            source_path=str(src),
            process=result if isinstance(result, dict) else {},
        )
        self._trim_history(code)

    @staticmethod
    def _finalize_web_video(raw_out: Path, web_out: Path) -> Path:
        """尽量转成浏览器可播的 H.264（ffmpeg）；失败则复制 mp4v。"""
        if not raw_out.is_file():
            raise FileNotFoundError(str(raw_out))
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            tmp = web_out.with_suffix(".h264.mp4")
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(raw_out),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3600,
                )
                shutil.copy2(tmp, web_out)
                tmp.unlink(missing_ok=True)
                return web_out
            except (subprocess.SubprocessError, OSError):
                pass
        shutil.copy2(raw_out, web_out)
        return web_out


_pipeline: UOVisionVideoPipeline | None = None


def init_video_pipeline(
    process_fn: Callable[[str, str, Callable[[float, int, int], None]], dict],
    cfg: UOVisionConfig | None = None,
) -> UOVisionVideoPipeline:
    global _pipeline
    _pipeline = UOVisionVideoPipeline(process_fn, cfg)
    return _pipeline


def get_video_pipeline() -> UOVisionVideoPipeline:
    if _pipeline is None:
        raise RuntimeError("UOVision 视频流水线尚未初始化")
    return _pipeline
