"""
UOVision 红外相机平台对接（广东动物所部署方案）。

相机流程（厂商文档）:
  1. 平台调用部署包 POST /acquirefile 向指定相机索要原图/原视频
  2. 相机 POST http://平台/servlet/original 上传二进制文件
  3. 相机定时 POST http://平台/servlet/heartbeat 上报在线与状态（文档 1.4）

本模块:
  - 接收 /servlet/original、/servlet/heartbeat
  - 可选转发 /acquirefile 到厂商部署包网关
  - 原图 JPEG 走 CloudInferenceService 单张识别
  - 结果与文件落盘到 data/uovision/
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 最近事件内存索引（重启后清空；完整记录在磁盘 JSON）
_recent_lock = threading.Lock()
_recent_events: list[dict[str, Any]] = []
_MAX_RECENT = 200

# 心跳索引（内存 + 磁盘 latest.json；重启后从磁盘恢复）
_heartbeat_lock = threading.Lock()
_heartbeat_state: dict[str, dict[str, Any]] = {}
_HEARTBEAT_HISTORY = 50

# 广东动物所对接文档 1.4 心跳 JSON 字段（cameraCode 单独解析）
_HEARTBEAT_DOC_KEYS = frozenset(
    {
        "latitude",
        "longitude",
        "altitude",
        "temperature",
        "signal",
        "battery",
        "gps",
        "model",
        "capacity",
        "freeSpace",
        "iccid",
        "firmware",
    }
)
# 兼容旧版 / getSettings 风格字段名
_HEARTBEAT_FIELD_ALIASES = {
    "batteryLevel": "battery",
    "signalStrength": "signal",
    "sdSpace": "capacity",
    "sdFreeSpace": "freeSpace",
    "version": "firmware",
}


@dataclass
class UOVisionConfig:
    data_dir: Path = field(default_factory=lambda: Path("data/uovision"))
    gateway_url: str = ""  # 厂商部署包根地址，如 http://192.168.1.10:8080
    max_body_bytes: int = 32 * 1024 * 1024


def load_uovision_config() -> UOVisionConfig:
    root = Path(os.environ.get("UOVISION_DATA_DIR", "data/uovision"))
    gw = (
        os.environ.get("UOVISION_GATEWAY_URL") or "http://open.uovcloud.com:98"
    ).strip().rstrip("/")
    max_mb = int(os.environ.get("UOVISION_MAX_MB", "512"))
    return UOVisionConfig(data_dir=root, gateway_url=gw, max_body_bytes=max_mb * 1024 * 1024)


def _push_recent(event: dict[str, Any]) -> None:
    with _recent_lock:
        _recent_events.insert(0, event)
        del _recent_events[_MAX_RECENT:]


def list_recent_events(limit: int = 50, camera_code: str | None = None) -> list[dict]:
    with _recent_lock:
        items = list(_recent_events)
    if camera_code:
        items = [e for e in items if e.get("camera_code") == camera_code]
    return items[: max(1, min(limit, 200))]


def default_heartbeat_online_sec() -> int:
    """超过该秒数未收到心跳/上传活动则视为离线（默认 24 小时，可用环境变量覆盖）。"""
    raw = os.environ.get("UOVISION_HEARTBEAT_ONLINE_SEC", "86400")
    try:
        sec = int(raw)
    except ValueError:
        sec = 86400
    return max(60, sec)


def _heartbeat_dir(cfg: UOVisionConfig, camera_code: str) -> Path:
    return cfg.data_dir / "heartbeats" / camera_code


def _latest_upload_activity_epoch(cfg: UOVisionConfig, camera_code: str) -> float | None:
    """扫描 data/uovision/YYYYMMDD/{IMEI}/ 下最近原图/原视频上传时间。"""
    code = str(camera_code or "").strip()
    root = cfg.data_dir
    if not code or not root.is_dir():
        return None
    latest = 0.0
    for day_dir in root.iterdir():
        if not day_dir.is_dir() or day_dir.name in ("heartbeats", "processed"):
            continue
        cam_dir = day_dir / code
        if not cam_dir.is_dir():
            continue
        for item in cam_dir.iterdir():
            if not item.is_file() or item.suffix.lower() == ".json":
                continue
            latest = max(latest, item.stat().st_mtime)
    return latest if latest > 0 else None


def _load_heartbeat_from_disk(cfg: UOVisionConfig, camera_code: str) -> dict[str, Any] | None:
    path = _heartbeat_dir(cfg, camera_code) / "latest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _ensure_heartbeat_loaded(cfg: UOVisionConfig, camera_code: str) -> dict[str, Any] | None:
    with _heartbeat_lock:
        cached = _heartbeat_state.get(camera_code)
    if cached is not None:
        return cached
    disk = _load_heartbeat_from_disk(cfg, camera_code)
    if disk is not None:
        with _heartbeat_lock:
            _heartbeat_state[camera_code] = disk
    return disk


def _save_heartbeat(cfg: UOVisionConfig, record: dict[str, Any]) -> None:
    camera_code = str(record.get("camera_code") or "").strip()
    if not camera_code:
        return
    out_dir = _heartbeat_dir(cfg, camera_code)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = out_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    # 裁剪历史，避免无限增长
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
        if len(lines) > _HEARTBEAT_HISTORY:
            history_path.write_text("\n".join(lines[-_HEARTBEAT_HISTORY:]) + "\n", encoding="utf-8")
    except OSError:
        pass


def _normalize_heartbeat_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """
    解析对接文档 JSON Body。
    返回 (cameraCode, telemetry, extra_raw)。
    """
    body = dict(payload or {})
    code = str(body.pop("cameraCode", None) or body.pop("camera_code", None) or "").strip()

    telemetry: dict[str, Any] = {}
    for key in _HEARTBEAT_DOC_KEYS:
        if key in body and body[key] is not None and body[key] != "":
            telemetry[key] = body[key]
    for old, new in _HEARTBEAT_FIELD_ALIASES.items():
        if new not in telemetry and old in body and body[old] is not None and body[old] != "":
            telemetry[new] = body[old]
            body.pop(old, None)

    extra = {k: v for k, v in body.items() if k not in _HEARTBEAT_DOC_KEYS}
    return code, telemetry, extra


def _extract_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    _, telemetry, _ = _normalize_heartbeat_payload(payload)
    return telemetry


def _heartbeat_public_view(record: dict[str, Any] | None, *, online_timeout_sec: int | None = None) -> dict[str, Any]:
    timeout = online_timeout_sec if online_timeout_sec is not None else default_heartbeat_online_sec()
    if not record:
        return {
            "online": False,
            "last_heartbeat": None,
            "last_heartbeat_age_sec": None,
            "online_timeout_sec": timeout,
            "telemetry": {},
        }
    ts = float(record.get("received_at_epoch") or 0.0)
    age = max(0.0, time.time() - ts) if ts > 0 else None
    online = age is not None and age <= timeout
    return {
        "online": online,
        "last_heartbeat": record.get("received_at"),
        "last_heartbeat_age_sec": round(age, 1) if age is not None else None,
        "online_timeout_sec": timeout,
        "telemetry": dict(record.get("telemetry") or {}),
        "heartbeat_count": int(record.get("heartbeat_count") or 0),
    }


def handle_heartbeat(
    *,
    camera_code: str = "",
    payload: dict[str, Any] | None = None,
    cfg: UOVisionConfig | None = None,
    client_ip: str = "",
) -> dict[str, Any]:
    """
    处理相机 POST /camera/heatbeat（对接文档 1.4；亦兼容 /servlet/heartbeat）。
    Body 须为 application/json，cameraCode（IMEI）为必填项。
    """
    cfg = cfg or load_uovision_config()
    merged = dict(payload or {})
    if camera_code and not merged.get("cameraCode"):
        merged["cameraCode"] = camera_code.strip()

    code, telemetry, extra = _normalize_heartbeat_payload(merged)
    if not code:
        raise ValueError("缺少必填字段 cameraCode（IMEI）")

    now = time.time()
    ts_iso = datetime.now(timezone.utc).isoformat()

    prev = _ensure_heartbeat_loaded(cfg, code) or {}
    merged_telemetry = dict(prev.get("telemetry") or {})
    if telemetry:
        merged_telemetry.update(telemetry)
    record = {
        "camera_code": code,
        "received_at": ts_iso,
        "received_at_epoch": now,
        "client_ip": client_ip or None,
        "telemetry": merged_telemetry,
        "extra": extra or None,
        "heartbeat_count": int(prev.get("heartbeat_count") or 0) + 1,
        "last_activity_source": "heartbeat",
    }
    with _heartbeat_lock:
        _heartbeat_state[code] = record
    _save_heartbeat(cfg, record)
    return {
        "code": 0,
        "message": "OK",
        "cameraCode": code,
        "received_at": ts_iso,
    }


def touch_heartbeat_activity(
    camera_code: str,
    *,
    cfg: UOVisionConfig | None = None,
    source: str = "upload",
) -> None:
    """相机上传原图/原视频时刷新在线时间（保留最近一次心跳遥测）。"""
    cfg = cfg or load_uovision_config()
    code = str(camera_code or "").strip()
    if not code:
        return
    prev = _ensure_heartbeat_loaded(cfg, code) or {}
    now = time.time()
    ts_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "camera_code": code,
        "received_at": ts_iso,
        "received_at_epoch": now,
        "client_ip": prev.get("client_ip"),
        "telemetry": dict(prev.get("telemetry") or {}),
        "extra": prev.get("extra"),
        "heartbeat_count": int(prev.get("heartbeat_count") or 0),
        "last_activity_source": source,
    }
    with _heartbeat_lock:
        _heartbeat_state[code] = record
    _save_heartbeat(cfg, record)


def get_heartbeat_status(
    camera_code: str,
    cfg: UOVisionConfig | None = None,
    *,
    online_timeout_sec: int | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_uovision_config()
    code = str(camera_code or "").strip()
    record = _ensure_heartbeat_loaded(cfg, code)
    timeout = online_timeout_sec if online_timeout_sec is not None else default_heartbeat_online_sec()
    view = _heartbeat_public_view(record, online_timeout_sec=timeout)
    upload_ts = _latest_upload_activity_epoch(cfg, code)
    hb_ts = float(record.get("received_at_epoch") or 0.0) if record else 0.0
    effective_ts = max(hb_ts, upload_ts or 0.0)
    if effective_ts > 0:
        age = max(0.0, time.time() - effective_ts)
        view["online"] = age <= timeout
        view["last_heartbeat_age_sec"] = round(age, 1)
        if upload_ts and upload_ts >= hb_ts:
            view["last_heartbeat"] = datetime.fromtimestamp(upload_ts, timezone.utc).isoformat()
        elif record:
            view["last_heartbeat"] = record.get("received_at")
    if record:
        view["camera_code"] = code
        view["client_ip"] = record.get("client_ip")
    else:
        view["camera_code"] = code
    return view


def list_heartbeat_status(
    camera_codes: list[str] | None = None,
    cfg: UOVisionConfig | None = None,
    *,
    online_timeout_sec: int | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or load_uovision_config()
    codes = [str(c).strip() for c in (camera_codes or []) if str(c).strip()]
    if not codes:
        hb_root = cfg.data_dir / "heartbeats"
        if hb_root.is_dir():
            codes = sorted(p.name for p in hb_root.iterdir() if p.is_dir())
        with _heartbeat_lock:
            codes = sorted(set(codes) | set(_heartbeat_state.keys()))
    return [get_heartbeat_status(code, cfg, online_timeout_sec=online_timeout_sec) for code in codes]


def _save_payload(
    cfg: UOVisionConfig,
    camera_code: str,
    file_id: int,
    is_hq: int,
    ext: str,
    body: bytes,
) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = cfg.data_dir / day / camera_code
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = "photo" if is_hq == 1 else "video"
    path = out_dir / f"{file_id}_{kind}{ext}"
    path.write_bytes(body)
    return path


def _save_meta(path: Path, meta: dict[str, Any]) -> None:
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def handle_original_upload(
    *,
    camera_code: str,
    file_id: int,
    file_size: int,
    is_hq: int,
    content_type: str,
    body: bytes,
    infer_fn,
    cfg: UOVisionConfig | None = None,
    upload_route: str = "original",
    on_video_saved=None,
) -> dict[str, Any]:
    """
    处理相机 POST /servlet/original。
    infer_fn: callable(jpeg_bytes) -> dict 识别结果
    """
    cfg = cfg or load_uovision_config()
    ct = (content_type or "").split(";")[0].strip().lower()
    if is_hq not in (1, 2):
        raise ValueError("X-Is-Hq 必须为 1(原图) 或 2(原视频)")
    if file_size != len(body):
        raise ValueError(f"X-File-Size={file_size} 与实际上传 {len(body)} 字节不一致")
    if len(body) > cfg.max_body_bytes:
        raise ValueError(f"文件超过上限 {cfg.max_body_bytes // (1024*1024)}MB")
    if not camera_code.strip():
        raise ValueError("缺少 X-CameraCode")

    camera_code = camera_code.strip()
    touch_heartbeat_activity(camera_code, cfg=cfg, source=f"upload:{upload_route}")
    received_at = time.time()
    ts_iso = datetime.now(timezone.utc).isoformat()

    if is_hq == 1:
        if ct not in ("image/jpeg", "image/jpg"):
            raise ValueError(f"原图 Content-Type 应为 image/jpeg，收到: {content_type}")
        ext = ".jpg"
        saved = _save_payload(cfg, camera_code, file_id, is_hq, ext, body)
        infer_result: dict[str, Any] | None = None
        infer_error = ""
        try:
            infer_result = infer_fn(body)
        except Exception as e:
            infer_error = str(e)
        meta = {
            "camera_code": camera_code,
            "file_id": file_id,
            "file_size": file_size,
            "is_hq": is_hq,
            "content_type": ct,
            "saved_path": str(saved.resolve()),
            "received_at": ts_iso,
            "infer_error": infer_error,
            "infer": infer_result,
        }
        _save_meta(saved, meta)
        event = {
            "camera_code": camera_code,
            "file_id": file_id,
            "kind": "photo",
            "saved_path": str(saved),
            "received_at": ts_iso,
            "tracks": (infer_result or {}).get("tracks", []),
            "infer_error": infer_error,
        }
        _push_recent(event)
        return _upload_ok_response(
            camera_code=camera_code,
            file_id=file_id,
            saved=str(saved),
            extra={
                "inference": infer_result,
                "infer_error": infer_error or None,
            },
        )

    # 原视频：落盘后排队 GPU 识别，供 /watch/ir-{IMEI} 播放
    if ct not in ("video/mpeg4", "video/mp4", "application/octet-stream"):
        raise ValueError(f"原视频 Content-Type 应为 video/mpeg4，收到: {content_type}")
    ext = ".mp4" if "mp4" in ct else ".mpeg4"
    saved = _save_payload(cfg, camera_code, file_id, is_hq, ext, body)
    meta = {
        "camera_code": camera_code,
        "file_id": file_id,
        "file_size": file_size,
        "is_hq": is_hq,
        "content_type": ct,
        "saved_path": str(saved.resolve()),
        "received_at": ts_iso,
        "upload_route": upload_route,
    }
    queue_info: dict[str, Any] | None = None
    queue_error = ""
    if on_video_saved is not None:
        try:
            queue_info = on_video_saved(
                camera_code=camera_code,
                source_path=saved,
                file_id=file_id,
                upload_route=upload_route,
            )
        except Exception as e:
            queue_error = str(e)
    meta["queue"] = queue_info
    meta["queue_error"] = queue_error or None
    _save_meta(saved, meta)
    from uovision_video_pipeline import ir_stream_id

    watch = f"/watch/{ir_stream_id(camera_code)}"
    event = {
        "camera_code": camera_code,
        "file_id": file_id,
        "kind": "video",
        "saved_path": str(saved),
        "received_at": ts_iso,
        "tracks": [],
        "watch_url": watch,
        "queue_status": (queue_info or {}).get("status"),
    }
    _push_recent(event)
    return _upload_ok_response(
        camera_code=camera_code,
        file_id=file_id,
        saved=str(saved),
        extra={
            "inference": None,
            "watch_url": watch,
            "stream_id": ir_stream_id(camera_code),
            "queue": queue_info,
            "queue_error": queue_error or None,
            "note": "视频已保存并排队识别，完成后可在 watch_url 播放最新一段",
        },
    )


def _upload_ok_response(
    *,
    camera_code: str,
    file_id: int,
    saved: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """厂商相机固件通常期望 code=0 / message=OK，否则可能一直重试导致界面卡死。"""
    out: dict[str, Any] = {
        "code": 0,
        "message": "OK",
        "cameraCode": camera_code,
        "ok": True,
        "camera_code": camera_code,
        "file_id": file_id,
    }
    if saved:
        out["saved"] = saved
    if extra:
        out.update(extra)
    return out


def handle_photos_upload(
    *,
    camera_code: str,
    file_id: int,
    file_size: int,
    is_hq: int,
    content_type: str,
    body: bytes,
    cfg: UOVisionConfig | None = None,
    upload_route: str = "photos",
) -> dict[str, Any]:
    """
    SD 卡 profiles 中 PATH_PTHUMB / PATH_VTHUMB 常指向 /servlet/photos。
    手动「发送」时相机可能先传缩略图再传原视频；缺此接口时固件可能长时间重试。
    """
    cfg = cfg or load_uovision_config()
    if not camera_code.strip():
        raise ValueError("缺少 X-CameraCode")
    if file_size != len(body):
        raise ValueError(f"X-File-Size={file_size} 与实际上传 {len(body)} 字节不一致")
    if len(body) > cfg.max_body_bytes:
        raise ValueError(f"文件超过上限 {cfg.max_body_bytes // (1024 * 1024)}MB")

    camera_code = camera_code.strip()
    ts_iso = datetime.now(timezone.utc).isoformat()
    saved = ""
    if body:
        ct = (content_type or "").split(";")[0].strip().lower()
        if is_hq == 1 or "jpeg" in ct or "jpg" in ct:
            ext = ".jpg"
        elif is_hq == 2 or "video" in ct or "mpeg" in ct:
            ext = ".mp4" if "mp4" in ct else ".mpeg4"
        else:
            ext = ".bin"
        saved_path = _save_payload(cfg, camera_code, file_id, is_hq or 1, ext, body)
        saved = str(saved_path)
        _save_meta(
            saved_path,
            {
                "camera_code": camera_code,
                "file_id": file_id,
                "file_size": file_size,
                "is_hq": is_hq,
                "content_type": content_type,
                "saved_path": saved,
                "received_at": ts_iso,
                "upload_route": upload_route,
                "kind": "thumbnail",
            },
        )
    _push_recent(
        {
            "camera_code": camera_code,
            "file_id": file_id,
            "kind": "thumbnail",
            "saved_path": saved or None,
            "received_at": ts_iso,
            "tracks": [],
        }
    )
    return _upload_ok_response(camera_code=camera_code, file_id=file_id, saved=saved)


def forward_acquire_file(payload: dict[str, Any], cfg: UOVisionConfig | None = None) -> dict[str, Any]:
    """转发 POST /acquireFile?cameraCode=&fileId=&fileType= 到 open.uovcloud.com。"""
    from uovision_open_api import forward_acquire_file as _open_acquire

    return _open_acquire(payload, cfg)


def build_modifysettings_template(
    camera_code: str,
    camera_name: str = "M11",
    *,
    video_length_sec: int | None = None,
    remote_control: int | None = None,
) -> dict[str, Any]:
    """
    /modifysettings 必须下发完整 JSON（厂商警告：只改单项可能导致死机）。
    此为模板，请按 GET 接口文档与现场需求修改后再下发。
    """
    from uovision_open_api import default_video_length_sec

    vlen = int(video_length_sec) if video_length_sec is not None else default_video_length_sec()
    rc = int(remote_control if remote_control is not None else os.environ.get("UOVISION_REMOTE_CONTROL", "1"))
    return {
        "cameraCode": camera_code,
        "cameraMode": 2,
        "cameraName": camera_name,
        "flashPower": 1,
        "gpsSwitch": 1,
        "overWrite": 0,
        "photoBurst": 1,
        "photoSize": 0,
        "remoteControl": rc,
        "sendMode": 0,
        "sendPhotoSize": 1,
        "sendVideoSize": 2,
        "sendingOption": 0,
        "sensitivity": 2,
        "start1": "00:00",
        "start1OnOff": "0",
        "start2": "00:00",
        "start2OnOff": "0",
        "start3": "00:00",
        "start3OnOff": "0",
        "start4": "00:00",
        "start4OnOff": "0",
        "stop1": "00:00",
        "stop2": "00:00",
        "stop3": "00:00",
        "stop4": "00:00",
        "timelapse": 1,
        "timelapseUnit": "h",
        "triggerInterval": 5,
        "triggerIntervalUnit": "s",
        "triggerMode": 0,
        "videoLength": vlen,
        "videoSize": 2,
        "week1": "1111111",
        "week2": "1111111",
        "week3": "1111111",
        "week4": "1111111",
    }


def default_uovision_retention_days() -> float:
    """红外原视频/识别结果保留天数，默认 3 天（环境变量 UOVISION_RETENTION_DAYS）。"""
    raw = os.environ.get("UOVISION_RETENTION_DAYS", "3")
    try:
        days = float(raw)
    except ValueError:
        days = 3.0
    return max(0.0, days)


def _uovision_file_age_sec(path: Path, meta: dict[str, Any] | None = None) -> float:
    """优先用 meta.received_at，否则用文件 mtime。"""
    if meta:
        received = meta.get("received_at") or meta.get("updated_at") or ""
        if received:
            try:
                return time.time() - datetime.fromisoformat(
                    str(received).replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                pass
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return 0.0


def _load_sidecar_meta(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(path.suffix + ".json")
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _unlink_file(path: Path) -> int:
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except OSError:
        return 0


def purge_uovision_older_than(
    days: float | None = None,
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    """
    删除超过 retention 天的红外相机文件：原图/原视频、识别 MP4、sidecar JSON。
    保留 heartbeats/（在线状态，体积很小）。
    """
    cfg = cfg or load_uovision_config()
    retention = float(days if days is not None else default_uovision_retention_days())
    if retention <= 0:
        return {"removed_files": 0, "days": retention, "freed_bytes": 0}

    max_age_sec = retention * 86400.0
    root = cfg.data_dir
    if not root.is_dir():
        return {"removed_files": 0, "days": retention, "freed_bytes": 0}

    removed = 0
    freed = 0

    media_suffixes = {".mp4", ".mpeg4", ".jpg", ".jpeg", ".bin"}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] == "heartbeats":
            continue
        if len(rel.parts) >= 2 and rel.parts[0] == "processed" and path.name == "latest.json":
            continue

        is_media = path.suffix.lower() in media_suffixes
        is_json = path.suffix.lower() == ".json"
        if not is_media and not is_json:
            continue

        meta = _load_sidecar_meta(path) if is_media else {}
        if is_json:
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                meta = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                meta = {}

        if _uovision_file_age_sec(path, meta) < max_age_sec:
            continue

        freed += _unlink_file(path)
        removed += 1

        if is_media:
            sidecar = path.with_suffix(path.suffix + ".json")
            if sidecar.is_file():
                freed += _unlink_file(sidecar)
                removed += 1

    processed_root = root / "processed"
    reset_cameras: list[str] = []
    if processed_root.is_dir():
        for cam_dir in processed_root.iterdir():
            if not cam_dir.is_dir():
                continue
            latest_json = cam_dir / "latest.json"
            latest_mp4 = cam_dir / "latest.mp4"
            state: dict[str, Any] = {}
            if latest_json.is_file():
                try:
                    loaded = json.loads(latest_json.read_text(encoding="utf-8"))
                    state = loaded if isinstance(loaded, dict) else {}
                except (OSError, json.JSONDecodeError):
                    state = {}

            if latest_mp4.is_file() and _uovision_file_age_sec(latest_mp4, state) >= max_age_sec:
                freed += _unlink_file(latest_mp4)
                removed += 1

            if latest_json.is_file() and (
                not latest_mp4.is_file()
                or _uovision_file_age_sec(latest_mp4, state) >= max_age_sec
            ):
                state["status"] = "idle"
                state["output_path"] = ""
                state["progress_pct"] = 0.0
                state["error"] = ""
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                latest_json.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                reset_cameras.append(cam_dir.name)

    for dirpath in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if dirpath == root:
            continue
        if dirpath.relative_to(root).parts and dirpath.relative_to(root).parts[0] == "heartbeats":
            continue
        try:
            dirpath.rmdir()
        except OSError:
            pass

    return {
        "removed_files": removed,
        "days": retention,
        "freed_bytes": freed,
        "freed_mb": round(freed / (1024 * 1024), 2),
        "data_dir": str(root.resolve()),
        "camera_codes_reset": reset_cameras,
    }


def forward_modify_settings(settings: dict[str, Any], cfg: UOVisionConfig | None = None) -> dict[str, Any]:
    """按机型转发 POST /modifySettings{Model} 到 open.uovcloud.com。"""
    from uovision_open_api import forward_modify_settings as _open_modify

    return _open_modify(settings, cfg)
