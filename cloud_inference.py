"""
云端推理：按 session 维护跟踪状态，供 FastAPI 调用。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cloud_render import StreamTrackSmoother, encode_jpeg, render_annotated_frame
from video_tracker_yolo import YOLOElephantTracker

_clip_recorder = None


def set_clip_recorder(recorder) -> None:
    global _clip_recorder
    _clip_recorder = recorder


@dataclass
class _Session:
    tracker: YOLOElephantTracker
    last_seen: float = field(default_factory=time.time)
    latest_jpeg: bytes | None = None
    latest_jpeg_ts: float = 0.0
    stream_frame: np.ndarray | None = None
    stream_smoother: StreamTrackSmoother = field(default_factory=StreamTrackSmoother)
    stream_infer_t: float = 0.0
    allowed_elephants: list[str] | None = None


class CloudInferenceService:
    """多 session 大象检测 + 个体识别（有 GPU 时在云端跑 YOLO + ResNet）。"""

    def __init__(
        self,
        model_path: str = "best_elephant_model.pth",
        class_names_path: str = "class_names.json",
        yolo_weights: str = "yolov8m.pt",
        yolo_imgsz: int = 640,
        infer_max_width: int = 1280,
        detect_interval: int = 1,
        recog_interval: int = 2,
        classify: bool = True,
        min_confidence: float = 42.0,
        min_margin: float = 12.0,
        freeze_when_locked: bool = True,
        session_ttl_sec: int = 600,
        stream_max_width: int = 854,
        stream_jpeg_quality: int = 68,
        gpu_id: int = 0,
        stream_fps: float = 20.0,
        low_vram: bool = False,
        force_cpu: bool = False,
        allowed_elephants: list[str] | None = None,
    ):
        self._model_path = model_path
        self._class_names_path = class_names_path
        self._yolo_weights = yolo_weights
        self._yolo_imgsz = yolo_imgsz
        self._infer_max_width = infer_max_width
        self._detect_interval = detect_interval
        self._recog_interval = recog_interval
        self._classify = classify
        self._min_confidence = min_confidence
        self._min_margin = min_margin
        self._freeze_when_locked = freeze_when_locked
        self._session_ttl_sec = session_ttl_sec
        self._stream_max_width = max(320, int(stream_max_width))
        self._stream_jpeg_quality = max(40, min(95, int(stream_jpeg_quality)))
        self._gpu_id = max(0, int(gpu_id))
        self._stream_fps = max(8.0, min(30.0, float(stream_fps)))
        self._low_vram = bool(low_vram)
        self._force_cpu = bool(force_cpu)
        self._default_allowed = allowed_elephants
        self._allowed_file = Path(
            os.environ.get("ELEPHANT_ALLOWED_FILE", "allowed_elephants.json")
        )
        if self._default_allowed is None:
            self._default_allowed = self._load_allowed_file()
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self._tracker: YOLOElephantTracker | None = None
        self._tracker_init_lock = threading.Lock()

    def _load_allowed_file(self) -> list[str] | None:
        if not self._allowed_file.is_file():
            return None
        try:
            data = json.loads(self._allowed_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _save_allowed_file(self, names: list[str] | None) -> None:
        try:
            if names:
                self._allowed_file.write_text(
                    json.dumps(names, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            elif self._allowed_file.is_file():
                self._allowed_file.unlink()
        except OSError:
            pass

    def get_default_allowed(self) -> list[str] | None:
        return list(self._default_allowed) if self._default_allowed else None

    def set_default_allowed(self, names: list[str] | None) -> None:
        self._default_allowed = names
        self._save_allowed_file(names)
        with self._lock:
            self._sync_classifier_allowed(None)

    def set_session_allowed(
        self, session_id: str, names: list[str] | None
    ) -> bool:
        with self._lock:
            tracker = self._ensure_tracker()
            if session_id not in self._sessions:
                self._sessions[session_id] = _Session(
                    tracker=tracker, allowed_elephants=names
                )
            else:
                self._sessions[session_id].allowed_elephants = names
        self._sync_classifier_allowed(self._sessions.get(session_id))
        return True

    def get_session_allowed(self, session_id: str) -> list[str] | None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return None
            if sess.allowed_elephants is not None:
                return list(sess.allowed_elephants)
        return self.get_default_allowed()

    def _sync_classifier_allowed(self, sess: _Session | None) -> None:
        if self._tracker is None or self._tracker.classifier is None:
            return
        allowed = self._default_allowed
        if sess is not None and sess.allowed_elephants is not None:
            allowed = sess.allowed_elephants
        self._tracker.classifier.set_allowed_elephants(allowed)

    def _ensure_tracker(self) -> YOLOElephantTracker:
        """全服务共用一套 YOLO + 分类器，避免每 session 重复占 GPU 显存。"""
        if self._tracker is not None:
            return self._tracker
        with self._tracker_init_lock:
            if self._tracker is None:
                self._tracker = self._new_tracker()
                self._sync_classifier_allowed(None)
                self._log_gpu_memory("模型加载后")
        return self._tracker

    @staticmethod
    def _log_gpu_memory(label: str) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            idx = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(idx) / (1024**2)
            reserved = torch.cuda.memory_reserved(idx) / (1024**2)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024**2)
            print(
                f"GPU 显存 [{label}] cuda:{idx} "
                f"已分配 {alloc:.0f} MiB | 缓存 {reserved:.0f} MiB | 总量 {total:.0f} MiB"
            )
        except Exception:
            pass

    def _new_tracker(self) -> YOLOElephantTracker:
        yolo_gpu = -1 if self._force_cpu else self._gpu_id
        cls_gpu = -1 if (self._force_cpu or self._low_vram) else None
        return YOLOElephantTracker(
            model_path=self._model_path,
            class_names_path=self._class_names_path,
            use_yolo=True,
            yolo_weights=self._yolo_weights,
            yolo_imgsz=self._yolo_imgsz,
            infer_max_width=self._infer_max_width,
            detect_interval=self._detect_interval,
            recognition_interval=self._recog_interval,
            classify_elephants=self._classify,
            show_overlay_confidence=False,
            freeze_recognition_when_locked=self._freeze_when_locked,
            clear_after_empty_detects=2,
            max_lost_frames=12,
            min_confidence=self._min_confidence,
            min_margin=self._min_margin,
            ema_alpha=0.18,
            cuda_device=yolo_gpu,
            classifier_cuda_device=cls_gpu,
            yolo_use_predict=self._low_vram or self._force_cpu,
        )

    def _release_gpu_cache(self) -> None:
        if not self._low_vram and not self._force_cpu:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _cleanup_stale_sessions(self) -> None:
        now = time.time()
        stale = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_seen > self._session_ttl_sec
        ]
        for sid in stale:
            del self._sessions[sid]

    def _get_session_unlocked(self, session_id: str | None) -> tuple[str, _Session]:
        self._cleanup_stale_sessions()
        tracker = self._ensure_tracker()
        if session_id and session_id in self._sessions:
            sess = self._sessions[session_id]
            sess.last_seen = time.time()
            return session_id, sess
        sid = session_id or uuid.uuid4().hex
        sess = _Session(tracker=tracker)
        self._sessions[sid] = sess
        return sid, sess

    def _get_session(self, session_id: str | None) -> tuple[str, _Session]:
        with self._lock:
            return self._get_session_unlocked(session_id)

    def warmup(self) -> None:
        """启动时加载唯一一套模型，避免首次请求超时。"""
        print("正在预热模型（YOLO + 分类器，约 30～90 秒）…")
        t0 = time.perf_counter()
        self._ensure_tracker()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._log_gpu_memory("预热完成")
        print(f"模型预热完成 ({time.perf_counter() - t0:.1f}s)")
        # 启动时跑一帧假图，OOM 在 warmup 阶段暴露，而不是等 Pi 连上才 500
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", dummy, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                self.process_jpeg(buf.tobytes(), session_id="__warmup__")
                self.reset_session("__warmup__")
                self._release_gpu_cache()
                print("✓ 预热推理自检通过")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    "GPU 显存不足，预热推理失败。请改用: bash start_cloud_server_low_vram.sh "
                    "或 bash start_cloud_server_cpu.sh"
                ) from e
            raise

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                sess = self._sessions[session_id]
                sess.tracker.trackers.clear()
                sess.tracker.frame_count = 0
                sess.tracker._empty_yolo_streak = 0
                sess.stream_smoother.reset()
                sess.stream_frame = None
                del self._sessions[session_id]
                return True
            return False

    def get_latest_jpeg(self, session_id: str) -> bytes | None:
        jpeg, _ = self.get_latest_jpeg_with_ts(session_id)
        return jpeg

    def get_latest_jpeg_with_ts(self, session_id: str) -> tuple[bytes | None, float]:
        jpeg = self.render_stream_jpeg(session_id)
        if jpeg is None:
            return None, 0.0
        with self._lock:
            sess = self._sessions.get(session_id)
            ts = sess.stream_infer_t if sess else 0.0
        return jpeg, ts

    def render_stream_jpeg(self, session_id: str) -> bytes | None:
        """按当前时间外推框并渲染 MJPEG 帧（比仅推理完成时更新更流畅）。"""
        now = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None or sess.stream_frame is None:
                return None
            frame = sess.stream_frame
            fh, fw = frame.shape[:2]
            tracks = sess.stream_smoother.display(now, fw, fh)
            n_trk = len(tracks)
        info = f"Trk:{n_trk} | Live"
        annotated = render_annotated_frame(frame, tracks, info_line=info)
        return encode_jpeg(
            annotated,
            quality=self._stream_jpeg_quality,
            max_width=0,
        )

    @staticmethod
    def _prepare_stream_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if max_width > 0 and w > max_width:
            scale = max_width / float(w)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        return frame

    def get_stream_status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None or sess.stream_frame is None:
                return {"active": False, "session_id": session_id}
            return {
                "active": True,
                "session_id": session_id,
                "updated_sec_ago": round(time.time() - sess.stream_infer_t, 1),
                "tracks": len(sess.tracker.trackers),
                "stream_fps": self._stream_fps,
            }

    def get_stream_tracks(self, session_id: str) -> dict[str, Any]:
        """供网页直播页轮询：当前画面中的大象名字与框颜色。"""
        now = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None or sess.stream_frame is None:
                return {"active": False, "session_id": session_id, "tracks": []}
            fh, fw = sess.stream_frame.shape[:2]
            tracks = sess.stream_smoother.display(now, fw, fh)
        items: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for tr in tracks:
            name = str(tr.get("name", "识别中..."))
            if name in seen_names:
                continue
            seen_names.add(name)
            b, g, r = tr.get("color_bgr", [80, 255, 100])
            items.append(
                {
                    "track_id": int(tr["track_id"]),
                    "name": name,
                    "color_rgb": [int(r), int(g), int(b)],
                }
            )
        return {
            "active": True,
            "session_id": session_id,
            "tracks": items,
        }

    @staticmethod
    def decode_jpeg(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码 JPEG 图像")
        return frame

    def process_jpeg(
        self,
        image_bytes: bytes,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        frame = self.decode_jpeg(image_bytes)
        live = os.environ.get("LIVE_STREAM_ENABLE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        record_infer = os.environ.get("ELEPHANT_CLIP_RECORD_INFER", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        with self._lock:
            sid, sess = self._get_session_unlocked(session_id)
            self._sync_classifier_allowed(sess)
            tracks = self._process_frame(sess.tracker, frame)
            h, w = frame.shape[:2]
            infer_t = time.time()
            stream_tracks = tracks
            if live:
                stream_frame = self._prepare_stream_frame(frame, self._stream_max_width)
                sh, sw = stream_frame.shape[:2]
                sx, sy = sw / float(w), sh / float(h)
                stream_tracks = []
                for tr in tracks:
                    x, y, bw, bh = tr["bbox"]
                    stream_tracks.append(
                        {
                            **tr,
                            "bbox": [
                                int(round(x * sx)),
                                int(round(y * sy)),
                                max(1, int(round(bw * sx))),
                                max(1, int(round(bh * sy))),
                            ],
                        }
                    )
                sess.stream_frame = stream_frame
                sess.stream_smoother.update(stream_tracks, infer_t)
                sess.stream_infer_t = infer_t
                sess.latest_jpeg_ts = infer_t
            recorder = _clip_recorder
        self._release_gpu_cache()
        if recorder is not None and record_infer and live:
            try:
                info = f"Trk:{len(tracks)} | Live"
                with self._lock:
                    sf = self._sessions.get(sid)
                    sf_frame = sf.stream_frame if sf else frame
                    st_tracks = stream_tracks
                annotated = render_annotated_frame(sf_frame, st_tracks, info_line=info)
                recorder.on_frame(sid, annotated, tracks, infer_t)
            except Exception:
                pass
        jpeg = None
        if live:
            with self._lock:
                sf = self._sessions.get(sid)
                if sf and sf.stream_frame is not None:
                    info = f"Trk:{len(stream_tracks)} | Live"
                    annotated = render_annotated_frame(
                        sf.stream_frame, stream_tracks, info_line=info
                    )
                    jpeg = encode_jpeg(
                        annotated,
                        quality=self._stream_jpeg_quality,
                        max_width=0,
                    )
                    if jpeg is not None and sid in self._sessions:
                        self._sessions[sid].latest_jpeg = jpeg
        result: dict[str, Any] = {
            "session_id": sid,
            "frame_width": w,
            "frame_height": h,
            "tracks": tracks,
            "allowed_elephants": self.get_session_allowed(sid),
        }
        if live:
            result["watch_path"] = f"/watch/{sid}"
            result["stream_path"] = f"/stream/{sid}"
        else:
            result["clips_path"] = "/watch/clips"
        return result

    def infer_still_jpeg(self, jpeg_bytes: bytes) -> dict[str, Any]:
        """单张原图识别（红外相机等），不污染 Pi 直播 session 跟踪状态。"""
        frame = self.decode_jpeg(jpeg_bytes)
        with self._lock:
            self._sync_classifier_allowed(None)
            tracker = self._ensure_tracker()
            saved = {
                "frame_count": tracker.frame_count,
                "trackers": dict(tracker.trackers),
                "next_id": tracker.next_id,
            }
            try:
                tracker.frame_count = 0
                tracker.trackers.clear()
                tracker.next_id = 0
                tracks = self._process_frame(tracker, frame)
            finally:
                tracker.frame_count = saved["frame_count"]
                tracker.trackers.clear()
                tracker.trackers.update(saved["trackers"])
                tracker.next_id = saved["next_id"]
        h, w = frame.shape[:2]
        names = [t["name"] for t in tracks if t.get("name") and t["name"] != "识别中..."]
        return {
            "frame_width": w,
            "frame_height": h,
            "tracks": tracks,
            "elephant_names": names,
            "elephant_count": len(tracks),
            "allowed_elephants": self.get_default_allowed(),
        }

    def process_video_file(
        self,
        input_path: str,
        output_path: str,
        progress_cb=None,
    ) -> dict[str, Any]:
        """原视频离线识别，不污染 Pi 直播 session。"""
        with self._lock:
            self._sync_classifier_allowed(None)
            tracker = self._ensure_tracker()
            saved = {
                "frame_count": tracker.frame_count,
                "trackers": dict(tracker.trackers),
                "next_id": tracker.next_id,
            }
            try:
                tracker.frame_count = 0
                tracker.trackers.clear()
                tracker.next_id = 0
                result = tracker.process_video_to_file(
                    input_path, output_path, progress_cb=progress_cb
                )
            finally:
                tracker.frame_count = saved["frame_count"]
                tracker.trackers.clear()
                tracker.trackers.update(saved["trackers"])
                tracker.next_id = saved["next_id"]
                self._release_gpu_cache()
        return result

    @staticmethod
    def _process_frame(tracker: YOLOElephantTracker, frame: np.ndarray) -> list[dict]:
        tracker.frame_count += 1
        if (tracker.frame_count - 1) % tracker.detect_interval == 0:
            detections = tracker.detect_objects(frame)
            tracker._update_tracks_for_frame(frame, detections)
        else:
            tracker._update_tracks_for_frame(frame, None)

        out: list[dict] = []
        for track_id, info in tracker.trackers.items():
            x, y, bw, bh = info["bbox"]
            color = info.get("color") or (80, 255, 100)
            if isinstance(color, tuple):
                color = [int(v) for v in color]
            else:
                color = [int(v) for v in color]
            name = info.get("display_name") or "识别中..."
            out.append(
                {
                    "track_id": int(track_id),
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "name": str(name),
                    "color_bgr": color,
                }
            )
        return out
