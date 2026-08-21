"""
UOVision 开放平台 open.uovcloud.com API 客户端（Swagger 1.0）。

cameraCode = 相机 IMEI（15 位数字，不是 SN、不是 ICCID）。
"""

from __future__ import annotations

import os
from typing import Any

import requests

from uovision_camera import UOVisionConfig, load_uovision_config

# Swagger: getSettings{Model} / modifySettings{Model}
_MODEL_GET_PATH: dict[str, str] = {
    "UML5P": "/getSettingsUML5P",
    "UML6": "/getSettingsUML6",
    "UML7": "/getSettingsUML7",
    "UML8": "/getSettingsUML8",
    "UMQ2": "/getSettingsUMQ2",
    "UMQ6": "/getSettingsUMQ6",
    "UMSW3": "/getSettingsUMSW3",
    "UMW2": "/getSettingsUMW2",
    "UMW5P": "/getSettingsUMW5P",
    "UMW7": "/getSettingsUMW7",
}

_MODEL_MODIFY_PATH: dict[str, str] = {
    "UML5P": "/modifySettingsUML5P",
    "UML6": "/modifySettingsUML6",
    "UML7": "/modifySettingsUML7",
    "UML8": "/modifySettingsUML8",
    "UMQ2": "/modifySettingsUMQ2",
    "UMQ6": "/modifySettingsUMQ6",
    "UMSW3": "/modifySettingsUMSW3",
    "UMW2": "/modifySettingsUMW2",
    "UMW5P": "/modifySettingsUMW5P",
    "UMW7": "/modifySettingsUMW7",
}

# Swagger /add 允许的 model 参数
ADD_CAMERA_MODELS = frozenset(
    {"UML5P", "UMW5P", "UML8", "UML7", "UMW7", "UMW2", "UMQ2", "UMSW3", "UMQ6", "UML6"}
)


def _norm_model(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "")


def _unwrap_api_response(body: Any) -> dict[str, Any]:
    if isinstance(body, dict) and "code" in body:
        return {
            "api_code": body.get("code"),
            "api_msg": body.get("msg"),
            "data": body.get("data"),
            "raw": body,
        }
    return {"api_code": None, "api_msg": None, "data": body, "raw": body}


def _request(
    cfg: UOVisionConfig,
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json_body: Any = None,
    timeout: int = 30,
) -> dict[str, Any]:
    if not cfg.gateway_url:
        return {"ok": False, "error": "未配置 UOVISION_GATEWAY_URL"}
    url = f"{cfg.gateway_url.rstrip('/')}{path}"
    try:
        r = requests.request(
            method.upper(),
            url,
            params=params,
            json=json_body,
            timeout=timeout,
        )
        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text[:4000]}
        ok = 200 <= r.status_code < 300
        if isinstance(body, dict) and body.get("code") not in (None, 0, 200):
            ok = False
        out = {
            "ok": ok,
            "http_status": r.status_code,
            "url": url,
            **_unwrap_api_response(body),
        }
        if not ok and isinstance(body, dict):
            out["error"] = str(body.get("msg") or body)
        return out
    except requests.RequestException as e:
        return {"ok": False, "url": url, "error": str(e)}


def query_model(camera_code: str, cfg: UOVisionConfig | None = None) -> dict[str, Any]:
    """GET /queryModel?cameraCode=IMEI"""
    cfg = cfg or load_uovision_config()
    return _request(
        cfg,
        "GET",
        "/queryModel",
        params={"cameraCode": camera_code.strip()},
    )


def resolve_model(camera_code: str, cfg: UOVisionConfig | None = None) -> str | None:
    """
    解析 modifySettings/getSettings 应使用的机型。

    必须与 open.uovcloud.com 上 /add 登记的机型（queryModel 返回值）一致，
    否则 modifySettings 返回 code=35「不支持该相机型号」。
    """
    code = camera_code.strip()
    cfg = cfg or load_uovision_config()

    res = query_model(code, cfg)
    if res.get("ok"):
        data = res.get("data")
        if isinstance(data, str) and data.strip():
            return _norm_model(data)
        if isinstance(data, dict):
            for k in ("model", "data", "deviceModel"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return _norm_model(v)

    env_model = os.environ.get("UOVISION_CAMERA_MODEL", "").strip()
    if env_model:
        return _norm_model(env_model)
    try:
        from uovision_registry import find_camera, load_camera_registry

        rec = find_camera(code, load_camera_registry())
        if rec and rec.model:
            return _norm_model(rec.model)
    except Exception:
        pass
    return None


def add_cameras(
    model: str,
    imei_list: list[str],
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    """
    POST /add?model=UML8
    body: IMEI 字符串数组（须先注册到开放平台账号，否则 queryModel 返回「没有该相机」）
    """
    cfg = cfg or load_uovision_config()
    m = _norm_model(model)
    if m not in ADD_CAMERA_MODELS:
        return {
            "ok": False,
            "error": f"不支持的 model={m}，允许: {sorted(ADD_CAMERA_MODELS)}",
        }
    imeis = [str(x).strip() for x in imei_list if str(x).strip()]
    if not imeis:
        return {"ok": False, "error": "imei_list 不能为空"}
    return _request(cfg, "POST", "/add", params={"model": m}, json_body=imeis)


def get_camera_settings(camera_code: str, cfg: UOVisionConfig | None = None) -> dict[str, Any]:
    """按机型 GET /getSettings{Model}?cameraCode=IMEI，返回含 iccid、model 等。"""
    cfg = cfg or load_uovision_config()
    model = resolve_model(camera_code, cfg)
    if not model:
        return {
            "ok": False,
            "camera_code": camera_code,
            "error": "无法解析机型，请先确认 IMEI 已添加到平台（/add）",
        }
    path = _MODEL_GET_PATH.get(model)
    if not path:
        return {"ok": False, "camera_code": camera_code, "model": model, "error": f"未知机型 {model}"}
    res = _request(cfg, "GET", path, params={"cameraCode": camera_code.strip()})
    res["camera_code"] = camera_code.strip()
    res["model"] = model
    return res


def probe_imei_by_iccid(
    candidate_imeis: list[str],
    target_iccid: str,
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    """
    用 ICCID 反查 IMEI：对每个候选 IMEI 调 getSettings，比对返回的 iccid。
    """
    cfg = cfg or load_uovision_config()
    want = target_iccid.strip().lower()
    matches: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for imei in candidate_imeis:
        imei = str(imei).strip()
        if not imei:
            continue
        info = get_camera_settings(imei, cfg)
        if not info.get("ok"):
            errors.append({"imei": imei, "error": str(info.get("error") or info.get("api_msg"))})
            continue
        data = info.get("data") or {}
        iccid = str(data.get("iccid") or "").strip()
        if iccid.lower() == want:
            matches.append(
                {
                    "imei": imei,
                    "iccid": iccid,
                    "model": info.get("model") or data.get("model"),
                    "camera_name": data.get("cameraName"),
                    "phone_number": data.get("phoneNumber"),
                }
            )
    return {
        "target_iccid": target_iccid,
        "matches": matches,
        "errors": errors,
        "found": len(matches) > 0,
    }


def forward_acquire_file(payload: dict[str, Any], cfg: UOVisionConfig | None = None) -> dict[str, Any]:
    """
    POST /acquireFile?cameraCode=&fileId=&fileType=
    fileType: 1 图片, 2 视频
    """
    cfg = cfg or load_uovision_config()
    camera_code = str(payload.get("cameraCode") or payload.get("camera_code") or "").strip()
    file_id = payload.get("fileId") or payload.get("file_id")
    file_type = int(payload.get("fileType") or payload.get("file_type") or 1)
    if not camera_code or file_id is None:
        return {
            "ok": False,
            "error": "需要 cameraCode 与 fileId",
            "example": "/acquireFile?cameraCode=868xxx&fileId=123&fileType=2",
        }
    return _request(
        cfg,
        "POST",
        "/acquireFile",
        params={
            "cameraCode": camera_code,
            "fileId": int(file_id),
            "fileType": file_type,
        },
    )


def forward_get_file(camera_code: str, cfg: UOVisionConfig | None = None) -> dict[str, Any]:
    """POST /getFile?cameraCode= 远程取图/触发拍摄。"""
    cfg = cfg or load_uovision_config()
    return _request(cfg, "POST", "/getFile", params={"cameraCode": camera_code.strip()})


def forward_modify_settings(settings: dict[str, Any], cfg: UOVisionConfig | None = None) -> dict[str, Any]:
    """POST /modifySettings{Model}，body 为完整参数 JSON。"""
    cfg = cfg or load_uovision_config()
    camera_code = str(settings.get("cameraCode") or "").strip()
    if not camera_code:
        return {"ok": False, "error": "settings 须包含 cameraCode（IMEI）"}
    model = resolve_model(camera_code, cfg)
    if not model:
        return {"ok": False, "error": f"无法解析 {camera_code} 的机型"}
    path = _MODEL_MODIFY_PATH.get(model)
    if not path:
        return {"ok": False, "error": f"不支持机型 {model}"}
    return _request(cfg, "POST", path, json_body=settings)


def set_server_base(
    *,
    imei_list: list[str],
    data_svr: str = "",
    ctrl_svr: str = "",
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    """
    POST /setServerBase
    dataSvr: 原图/原视频上传前缀，如 http://120.196.88.140:9998/servlet/original 的父路径
    文档要求填「除 api 相对路径外的部分」；通常填 http://host:port
    """
    cfg = cfg or load_uovision_config()
    body = {"imeiList": [str(x).strip() for x in imei_list if str(x).strip()]}
    if data_svr:
        body["dataSvr"] = data_svr.rstrip("/")
    if ctrl_svr:
        body["ctrlSvr"] = ctrl_svr.rstrip("/")
    if not body.get("dataSvr") and not body.get("ctrlSvr"):
        return {"ok": False, "error": "至少设置 dataSvr 或 ctrlSvr 之一"}
    return _request(cfg, "POST", "/setServerBase", json_body=body)


def default_video_length_sec() -> int:
    """Swagger：videoLength 仅 5–60 秒，不是 300。"""
    raw = os.environ.get("UOVISION_VIDEO_LENGTH", "60")
    try:
        v = int(raw)
    except ValueError:
        v = 60
    return max(5, min(60, v))


_READ_ONLY_SETTING_KEYS = frozenset(
    {
        "altitude",
        "batteryLevel",
        "iccid",
        "latitude",
        "latitudeDesc",
        "longitude",
        "longitudeDesc",
        "model",
        "phoneNumber",
        "sdFreeSpace",
        "sdSpace",
        "signalStrength",
        "temperature",
        "version",
        "recordTime",
        "sendTime",
        "start1CustomTimeBegin",
        "start1CustomTimeEnd",
        "start2CustomTimeBegin",
        "start2CustomTimeEnd",
        "start3CustomTimeBegin",
        "start3CustomTimeEnd",
        "start4CustomTimeBegin",
        "start4CustomTimeEnd",
    }
)


def merge_settings_for_modify(
    camera_code: str,
    overrides: dict[str, Any],
    cfg: UOVisionConfig | None = None,
) -> dict[str, Any]:
    """
    先 GET 当前参数，再覆盖 overrides，避免只改单项导致相机异常。
    GET 失败时仅返回 overrides（须含 Swagger 要求的完整字段）。
    """
    base = dict(overrides)
    base["cameraCode"] = camera_code.strip()
    info = get_camera_settings(camera_code, cfg)
    if not info.get("ok"):
        return base
    data = dict(info.get("data") or {})
    for k in _READ_ONLY_SETTING_KEYS:
        data.pop(k, None)
    data["cameraCode"] = camera_code.strip()
    data.update(overrides)
    return data
