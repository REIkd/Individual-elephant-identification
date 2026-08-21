"""
园区 5 台 UOVision 红外相机注册表与批量配置。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uovision_camera import UOVisionConfig, get_heartbeat_status, load_uovision_config
from uovision_open_api import (
    add_cameras,
    default_video_length_sec,
    forward_get_file,
    forward_modify_settings,
    get_camera_settings,
    merge_settings_for_modify,
    query_model,
    resolve_model,
    set_server_base,
)
from uovision_video_pipeline import get_video_pipeline, ir_stream_id


def default_camera_model() -> str:
    return os.environ.get("UOVISION_CAMERA_MODEL", "UML7").strip().upper() or "UML7"


def normalize_uovision_camera_name(name: str = "", *, camera_id: str = "") -> str:
    """UOVision cameraName 必须恰好 6 位，且仅含 0-9、A-Z。"""
    s = re.sub(r"[^0-9A-Z]", "", (name or "").strip().upper())
    if len(s) == 6:
        return s
    cid = re.sub(r"[^0-9A-Z]", "", str(camera_id or "").strip().upper())
    if cid:
        return f"CAM{cid.zfill(3)}"[:6].ljust(6, "0")
    if s:
        return (s + "0" * 6)[:6]
    return "CAM001"


@dataclass
class IrCameraRecord:
    id: str
    label: str
    sn: str
    imei: str
    iccid: str
    sim: str = ""
    camera_name: str = ""
    model: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "sn": self.sn,
            "imei": self.imei,
            "iccid": self.iccid,
            "sim": self.sim,
            "camera_name": normalize_uovision_camera_name(self.camera_name, camera_id=self.id),
            "model": self.model or default_camera_model(),
            "stream_id": ir_stream_id(self.imei),
            "watch_url": f"/watch/{ir_stream_id(self.imei)}",
            "status_url": f"/api/v1/uovision/ir/{self.imei}/status",
        }


def registry_path() -> Path:
    raw = os.environ.get("UOVISION_CAMERAS_FILE", "uovision_cameras.json")
    return Path(raw)


def load_camera_registry(path: Path | None = None) -> list[IrCameraRecord]:
    p = path or registry_path()
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    out: list[IrCameraRecord] = []
    for row in data:
        out.append(
            IrCameraRecord(
                id=str(row.get("id", "")).strip(),
                label=str(row.get("label", "")).strip(),
                sn=str(row.get("sn", "")).strip(),
                imei=str(row.get("imei", "")).strip(),
                iccid=str(row.get("iccid", "")).strip().lower(),
                sim=str(row.get("sim", "")).strip(),
                camera_name=str(row.get("camera_name", "")).strip(),
                model=str(row.get("model", "")).strip().upper(),
            )
        )
    return [c for c in out if c.imei]


def find_camera(imei_or_id: str, registry: list[IrCameraRecord] | None = None) -> IrCameraRecord | None:
    key = str(imei_or_id).strip()
    items = registry or load_camera_registry()
    for c in items:
        if c.imei == key or c.id == key or c.sn == key:
            return c
    return None


def ir_recognition_overrides(
    camera_name: str,
    *,
    camera_id: str = "",
    video_length_sec: int | None = None,
) -> dict[str, Any]:
    """大象识别项目推荐参数：PIR + 原视频上传 + 远程实时唤醒。"""
    vlen = max(5, min(60, int(video_length_sec or default_video_length_sec())))
    return {
        "cameraName": normalize_uovision_camera_name(camera_name, camera_id=camera_id),
        "cameraMode": 2,
        "videoLength": vlen,
        "videoSize": 2,
        "sendVideoSize": 2,
        "sendPhotoSize": 1,
        "remoteControl": 1,
        "sensitivity": 2,
        "triggerMode": 0,
        "overWrite": 1,
        "photoBurst": 1,
        "sendMode": 0,
    }


def register_all_cameras(
    registry: list[IrCameraRecord] | None = None,
    *,
    model: str = "",
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    """POST /add 将 IMEI 注册到 open.uovcloud.com 当前账号。"""
    cfg = cfg or load_uovision_config()
    items = registry or load_camera_registry()
    if not items:
        return {"ok": False, "error": "未找到 uovision_cameras.json"}
    groups: dict[str, list[str]] = {}
    for rec in items:
        m = _norm_model(rec.model or model or default_camera_model())
        groups.setdefault(m, []).append(rec.imei)
    results: list[dict[str, Any]] = []
    all_ok = True
    for m, imeis in groups.items():
        res = add_cameras(m, imeis, cfg)
        results.append({"model": m, "imei_list": imeis, "result": res})
        if not res.get("ok"):
            all_ok = False
    return {
        "ok": all_ok,
        "groups": results,
        "note": "若仍报「没有该相机」，请确认 UOVISION_CAMERA_MODEL 与机身型号一致，且账号有权限",
    }


def _norm_model(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "")


def verify_camera(
    record: IrCameraRecord,
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_uovision_config()
    model_res = query_model(record.imei, cfg)
    info_res = get_camera_settings(record.imei, cfg)
    iccid_ok = False
    iccid_seen = ""
    if info_res.get("ok"):
        data = info_res.get("data") or {}
        iccid_seen = str(data.get("iccid") or "").strip().lower()
        iccid_ok = iccid_seen == record.iccid
    platform_model = resolve_model(record.imei, cfg)
    registry_model = _norm_model(record.model or default_camera_model())
    model_mismatch = bool(
        platform_model and registry_model and platform_model != registry_model
    )
    return {
        **record.to_public_dict(),
        "gateway_url": cfg.gateway_url or None,
        "query_model": model_res,
        "camera_info": info_res,
        "model": info_res.get("model") or platform_model,
        "platform_model": platform_model,
        "registry_model": registry_model,
        "model_mismatch": model_mismatch,
        "model_mismatch_hint": (
            "平台 queryModel 与本地登记不一致；改参/getFile 以 platform_model 为准。"
            if model_mismatch
            else None
        ),
        "iccid_expected": record.iccid,
        "iccid_seen": iccid_seen,
        "iccid_match": iccid_ok,
        "ok": bool(model_res.get("ok")) and bool(info_res.get("ok")) and (iccid_ok or not iccid_seen),
    }


def configure_camera_for_ir(
    record: IrCameraRecord,
    *,
    video_length_sec: int | None = None,
    trigger_capture: bool = False,
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_uovision_config()
    overrides = ir_recognition_overrides(
        record.camera_name,
        camera_id=record.id,
        video_length_sec=video_length_sec,
    )
    settings = merge_settings_for_modify(record.imei, overrides, cfg)
    modify = forward_modify_settings(settings, cfg)
    trigger = forward_get_file(record.imei, cfg) if trigger_capture else None
    return {
        "camera": record.to_public_dict(),
        "settings_sent": settings,
        "modify": modify,
        "trigger_get_file": trigger,
        "ok": bool(modify.get("ok")),
    }


def setup_all_cameras(
    *,
    data_svr: str = "",
    configure: bool = True,
    register: bool = False,
    trigger_capture: bool = False,
    video_length_sec: int | None = None,
    model: str = "",
    registry: list[IrCameraRecord] | None = None,
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_uovision_config()
    items = registry or load_camera_registry()
    if not items:
        return {"ok": False, "error": "未找到 uovision_cameras.json 或列表为空"}
    register_res = None
    if register:
        register_res = register_all_cameras(items, model=model, cfg=cfg)
    base = (
        data_svr.strip()
        or os.environ.get("UOVISION_CALLBACK_BASE", "").strip()
        or "http://120.196.88.140:9998"
    ).rstrip("/")
    imei_list = [c.imei for c in items]
    base_res = set_server_base(imei_list=imei_list, data_svr=base, cfg=cfg)
    per_camera: list[dict[str, Any]] = []
    for rec in items:
        row: dict[str, Any] = {"camera": rec.to_public_dict(), "verify": verify_camera(rec, cfg)}
        if configure:
            row["configure"] = configure_camera_for_ir(
                rec,
                video_length_sec=video_length_sec,
                trigger_capture=trigger_capture,
                cfg=cfg,
            )
        per_camera.append(row)
    all_ok = (register_res is None or register_res.get("ok")) and bool(base_res.get("ok")) and all(
        (not configure or (row.get("configure") or {}).get("ok")) for row in per_camera
    )
    return {
        "ok": all_ok,
        "data_svr": base,
        "register": register_res,
        "set_server_base": base_res,
        "cameras": per_camera,
        "watch_index": "/watch/ir",
    }


def list_cameras_with_status(registry: list[IrCameraRecord] | None = None) -> list[dict[str, Any]]:
    items = registry or load_camera_registry()
    cfg = load_uovision_config()
    pipe = None
    try:
        pipe = get_video_pipeline()
    except RuntimeError:
        pipe = None
    out: list[dict[str, Any]] = []
    for rec in items:
        row = rec.to_public_dict()
        hb = get_heartbeat_status(rec.imei, cfg)
        row["online"] = hb.get("online", False)
        row["last_heartbeat"] = hb.get("last_heartbeat")
        row["last_heartbeat_age_sec"] = hb.get("last_heartbeat_age_sec")
        row["heartbeat_telemetry"] = hb.get("telemetry") or {}
        if pipe is not None:
            st = pipe.get_state(rec.imei)
            row["pipeline_status"] = st.status
            row["video_ready"] = pipe.get_latest_video_path(rec.imei) is not None
            row["updated_at"] = st.updated_at
            if st.meta:
                proc = st.meta.get("process") or {}
                row["elephant_names"] = proc.get("elephant_names") or []
        out.append(row)
    return out
