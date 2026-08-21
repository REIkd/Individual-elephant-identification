"""
大象识别云端 API 服务（GPU 服务器 / 高配 PC 上运行）。

启动示例:
  set CLOUD_API_KEY=your-secret-key
  python cloud_server.py --host 0.0.0.0 --port 8000

Pi 客户端:
  python pi_cloud_client.py --server http://192.168.1.100:8000 --api-key your-secret-key
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from cloud_inference import CloudInferenceService, set_clip_recorder
from cloud_render import waiting_frame
try:
    from elephant_clip_recorder import ElephantClipRecorder
except ImportError:
    ElephantClipRecorder = None  # type: ignore[misc, assignment]
from predict import parse_allowed_elephants
from uovision_open_api import (
    default_video_length_sec,
    forward_get_file,
    get_camera_settings,
    merge_settings_for_modify,
    probe_imei_by_iccid,
    query_model,
    set_server_base,
)
from uovision_camera import (
    build_modifysettings_template,
    forward_acquire_file,
    forward_modify_settings,
    get_heartbeat_status,
    handle_heartbeat,
    handle_original_upload,
    handle_photos_upload,
    list_heartbeat_status,
    list_recent_events,
    load_uovision_config,
    default_uovision_retention_days,
    purge_uovision_older_than,
)
from uovision_registry import (
    IrCameraRecord,
    configure_camera_for_ir,
    default_camera_model,
    find_camera,
    list_cameras_with_status,
    load_camera_registry,
    normalize_uovision_camera_name,
    register_all_cameras,
    setup_all_cameras,
    verify_camera,
)
from uovision_video_pipeline import (
    default_ir_history_limit,
    get_video_pipeline,
    init_video_pipeline,
    ir_stream_id,
    parse_ir_stream_id,
)

app = FastAPI(title="Elephant Cloud Inference", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _uovision_log_path(path: str) -> bool:
    if path.startswith("/servlet/"):
        return True
    return path in ("/camera/heatbeat", "/camera/heartbeat", "/servlet/heartbeat")


@app.middleware("http")
async def uovision_upload_log(request: Request, call_next):
    path = request.url.path
    if request.method == "POST" and _uovision_log_path(path):
        ip = request.client.host if request.client else "?"
        if path.startswith("/servlet/") and path != "/servlet/heartbeat":
            cam = request.headers.get("x-cameracode") or request.headers.get("X-CameraCode")
            fid = request.headers.get("x-file-id") or request.headers.get("X-File-Id")
            fsize = request.headers.get("x-file-size") or request.headers.get("X-File-Size")
            hq = request.headers.get("x-is-hq") or request.headers.get("X-Is-Hq")
            print(
                f"[UOVision] ← POST {path} from {ip} camera={cam} file_id={fid} size={fsize} is_hq={hq}",
                flush=True,
            )
        else:
            print(f"[UOVision] ← POST {path} from {ip}", flush=True)
    response = await call_next(request)
    if request.method == "POST" and _uovision_log_path(path):
        print(f"[UOVision] → {path} HTTP {response.status_code}", flush=True)
    return response

_service: Optional[CloudInferenceService] = None
_clip_recorder: Optional[object] = None
_api_key: str = ""
_infer_pool = ThreadPoolExecutor(max_workers=1)
_live_stream_enabled: bool = os.environ.get("LIVE_STREAM_ENABLE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_clip_retention_days: float = float(os.environ.get("ELEPHANT_CLIP_RETENTION_DAYS", "3"))
_uovision_retention_days: float = default_uovision_retention_days()


def _enqueue_uovision_video(
    *,
    camera_code: str,
    source_path: Path,
    file_id: int,
    upload_route: str,
) -> dict:
    pipe = get_video_pipeline()
    return pipe.enqueue(
        camera_code=camera_code,
        source_path=source_path,
        file_id=file_id,
        upload_route=upload_route,
    )


def _process_ir_video(input_path: str, output_path: str, progress_cb) -> dict:
    if _service is None:
        raise RuntimeError("推理服务尚未初始化")
    return _service.process_video_file(input_path, output_path, progress_cb)


def get_service() -> CloudInferenceService:
    if _service is None:
        raise HTTPException(status_code=503, detail="推理服务尚未初始化")
    return _service


def get_clip_recorder():
    return _clip_recorder


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not _api_key:
        return
    if x_api_key != _api_key:
        raise HTTPException(status_code=401, detail="无效的 API Key")


@app.get("/health")
def health():
    uov = load_uovision_config()
    return {
        "status": "ok",
        "service": "elephant-cloud",
        "auth_required": bool(_api_key),
        "uovision_gateway": uov.gateway_url or None,
        "uovision_retention_days": _uovision_retention_days,
        "uovision_ir_history_limit": default_ir_history_limit(),
        "clip_retention_days": _clip_retention_days,
        "uovision_data_dir": str(uov.data_dir.resolve()),
    }


@app.post("/api/v1/infer")
async def infer_frame(
    image: UploadFile = File(..., description="JPEG 帧"),
    session_id: Optional[str] = Form(default=None),
    _: None = Depends(verify_api_key),
    svc: CloudInferenceService = Depends(get_service),
):
    t0 = time.perf_counter()
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空图像")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图像过大（上限 8MB）")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _infer_pool,
            lambda: svc.process_jpeg(raw, session_id or None),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败: {e}") from e

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


@app.post("/api/v1/reset")
def reset_session(
    session_id: str = Form(...),
    _: None = Depends(verify_api_key),
    svc: CloudInferenceService = Depends(get_service),
):
    ok = svc.reset_session(session_id)
    return {"session_id": session_id, "reset": ok}


def _load_class_names_for_ui() -> list[str]:
    if _service is not None:
        try:
            tr = _service._ensure_tracker()
            if tr.classifier is not None:
                return list(tr.classifier.class_names)
        except Exception:
            pass
    p = Path("class_names.json")
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


@app.get("/watch/ir", response_class=HTMLResponse)
def watch_ir_index():
    """5 台红外相机识别状态总览。"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>红外相机 · 大象识别</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: "Segoe UI", "PingFang SC", sans-serif; }
    h1 { font-size: 1.15rem; font-weight: 500; padding: 16px; margin: 0; text-align: center; }
    .sub { text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 12px; }
    #list { max-width: 960px; margin: 0 auto; padding: 0 12px 24px; }
    .item {
      display: block; background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
      padding: 12px 14px; margin-bottom: 10px; color: #eee; text-decoration: none;
    }
    .item:hover { border-color: #666; }
    .title { font-weight: 500; }
    .meta { color: #888; font-size: 0.85rem; margin-top: 4px; }
    .ready { color: #81c784; }
    .warn { color: #ffb74d; }
    .idle { color: #888; }
    .online { color: #64b5f6; }
    .offline { color: #888; }
    .empty { text-align: center; color: #666; padding: 40px 16px; }
  </style>
</head>
<body>
  <h1>红外相机 · 最新识别视频</h1>
  <p class="sub">相机上传原视频 → 服务器 GPU 识别打框 → 点击观看（每台可回看最近 3 条）</p>
  <div id="list"><p class="empty">加载中…</p></div>
  <script>
    function statusText(c) {
      const st = c.pipeline_status || "idle";
      if (st === "ready" && c.video_ready) return ["已就绪", "ready"];
      if (st === "processing") return ["识别中", "warn"];
      if (st === "queued") return ["排队中", "warn"];
      if (st === "error") return ["处理失败", "warn"];
      return ["等待上传", "idle"];
    }
    function onlineText(c) {
      if (c.online) return ["在线", "online"];
      if (c.last_heartbeat) return ["离线", "offline"];
      return ["未上报心跳", "offline"];
    }
    function fmtAge(sec) {
      if (sec == null) return "—";
      if (sec < 120) return Math.round(sec) + " 秒前";
      if (sec < 7200) return Math.round(sec / 60) + " 分钟前";
      return Math.round(sec / 3600) + " 小时前";
    }
    function fmtTelemetry(t) {
      t = t || {};
      const parts = [];
      if (t.battery != null) parts.push("电量 " + t.battery + "/10");
      if (t.signal != null) parts.push("信号 " + t.signal + "/5");
      if (t.temperature != null) parts.push("温度 " + t.temperature + "°F");
      if (t.freeSpace != null && t.capacity != null) parts.push("存储 " + t.freeSpace + "/" + t.capacity + "MB");
      return parts.length ? parts.join(" · ") : "—";
    }
    async function load() {
      const box = document.getElementById("list");
      try {
        const r = await fetch("/api/v1/uovision/cameras");
        const data = await r.json();
        const cams = data.cameras || [];
        if (!cams.length) {
          box.innerHTML = '<p class="empty">未配置相机列表（uovision_cameras.json）</p>';
          return;
        }
        box.innerHTML = cams.map(c => {
          const [txt, cls] = statusText(c);
          const [onTxt, onCls] = onlineText(c);
          const names = (c.elephant_names || []).join("、") || "—";
          const hb = fmtTelemetry(c.heartbeat_telemetry);
          const hbAge = fmtAge(c.last_heartbeat_age_sec);
          return '<a class="item" href="' + c.watch_url + '">'
            + '<div class="title">' + (c.label || c.id) + ' · IMEI ' + c.imei + '</div>'
            + '<div class="meta"><span class="' + onCls + '">' + onTxt + '</span>'
            + ' · 心跳 ' + hbAge + ' · ' + hb + '<br>'
            + '<span class="' + cls + '">' + txt + '</span>'
            + ' · 识别：' + names + '<br>SN ' + (c.sn || "—") + '</div></a>';
        }).join("");
      } catch (e) {
        box.innerHTML = '<p class="empty">加载失败</p>';
      }
    }
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>"""


@app.get("/watch/clips", response_class=HTMLResponse)
def watch_clips_gallery():
    """浏览器查看 Pi 识别到大象后自动保存的历史片段。"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>大象识别录像库</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: "Segoe UI", "PingFang SC", sans-serif; }
    h1 { font-size: 1.15rem; font-weight: 500; padding: 16px; margin: 0; text-align: center; }
    .sub { text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 12px; }
    #list { max-width: 960px; margin: 0 auto; padding: 0 12px 24px; }
    .item {
      display: block; background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
      padding: 12px 14px; margin-bottom: 10px; color: #eee; text-decoration: none;
    }
    .item:hover { border-color: #666; }
    .names { color: #81c784; font-weight: 500; }
    .meta { color: #888; font-size: 0.85rem; margin-top: 4px; }
    .empty { text-align: center; color: #666; padding: 40px 16px; }
  </style>
</head>
<body>
  <h1>Pi 大象识别 · 录像库</h1>
  <p class="sub">树莓派检测到大象 → 本地 1920 录制 → 自动上传；网页直播已关闭</p>
  <div id="list"><p class="empty">加载中…</p></div>
  <script>
    async function load() {
      const box = document.getElementById("list");
      try {
        const r = await fetch("/api/v1/clips?limit=80");
        const data = await r.json();
        const clips = data.clips || [];
        if (!clips.length) {
          box.innerHTML = '<p class="empty">暂无录像。请保持 Pi 客户端运行，画面中出现大象后会自动保存。</p>';
          return;
        }
        box.innerHTML = clips.map(c => {
          const names = (c.elephant_names || []).join("、") || "检测到大象（未命名）";
          const dur = c.duration_sec != null ? c.duration_sec + " 秒" : "—";
          return '<a class="item" href="/watch/clips/' + encodeURIComponent(c.clip_id) + '">'
            + '<div class="names">' + names + '</div>'
            + '<div class="meta">' + (c.started_at || "") + ' · ' + dur
            + ' · session ' + (c.session_id || "") + '</div></a>';
        }).join("");
      } catch (e) {
        box.innerHTML = '<p class="empty">加载失败</p>';
      }
    }
    load();
    setInterval(load, 15000);
  </script>
</body>
</html>"""


@app.get("/watch/clips/{clip_id}", response_class=HTMLResponse)
def watch_clip_player(clip_id: str):
    safe = clip_id.replace("<", "").replace(">", "").replace('"', "")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>录像 · {safe}</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: "Segoe UI", "PingFang SC", sans-serif; }}
    .page {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1rem; padding: 14px 16px 8px; margin: 0; }}
    a {{ color: #64b5f6; }}
    video {{ width: 100%; background: #000; display: block; }}
    .meta {{ padding: 12px 16px 20px; color: #aaa; font-size: 0.9rem; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="page">
    <h1><a href="/watch/clips">← 录像库</a></h1>
    <video controls playsinline preload="auto" src="/api/v1/clips/{safe}/video.mp4"></video>
    <div class="meta" id="meta">加载信息…</div>
  </div>
  <script>
    fetch("/api/v1/clips/{safe}").then(r => r.json()).then(c => {{
      const names = (c.elephant_names || []).join("、") || "—";
      document.getElementById("meta").innerHTML =
        "时间：" + (c.started_at || "—") + "<br>时长：" + (c.duration_sec || "—") + " 秒<br>象名：" + names;
    }}).catch(() => {{}});
  </script>
</body>
</html>"""


@app.get("/api/v1/clips")
def list_elephant_clips(
    limit: int = 50,
    session_id: Optional[str] = None,
):
    """Pi 直播自动保存的大象片段列表（无需 API Key，便于网页浏览）。"""
    rec = get_clip_recorder()
    if rec is None:
        return {"enabled": False, "clips": [], "note": "未启用 ELEPHANT_CLIP_ENABLE"}
    return {"enabled": True, "clips": rec.list_clips(limit=limit, session_id=session_id)}


@app.get("/api/v1/clips/{clip_id}")
def get_elephant_clip(clip_id: str):
    rec = get_clip_recorder()
    if rec is None:
        raise HTTPException(status_code=503, detail="录像功能未启用")
    meta = rec.get_clip(clip_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="录像不存在")
    return meta


@app.get("/api/v1/clips/{clip_id}/video.mp4")
def get_elephant_clip_video(clip_id: str):
    rec = get_clip_recorder()
    if rec is None:
        raise HTTPException(status_code=503, detail="录像功能未启用")
    path = rec.clip_video_path(clip_id)
    if path is None:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/v1/clips/upload")
async def upload_elephant_clip(
    video: UploadFile = File(..., description="Pi 上传的标注 MP4"),
    metadata: str = Form(..., description="JSON 元数据"),
    _: None = Depends(verify_api_key),
):
    """Pi 检测到大象后上传本地录像，供 /watch/clips 浏览下载。"""
    rec = get_clip_recorder()
    if rec is None:
        raise HTTPException(status_code=503, detail="录像功能未启用")
    try:
        meta_in = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"metadata 须为合法 JSON: {e}") from e
    raw = await video.read()
    if not raw:
        raise HTTPException(status_code=400, detail="视频为空")
    max_mb = int(os.environ.get("ELEPHANT_CLIP_UPLOAD_MAX_MB", "512"))
    if len(raw) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"视频超过 {max_mb}MB 上限")

    names = meta_in.get("elephant_names") or []
    if isinstance(names, str):
        names = [x.strip() for x in names.split(",") if x.strip()]
    started_at: float | None = None
    started_raw = meta_in.get("started_at") or ""
    if started_raw:
        try:
            from datetime import datetime

            started_at = datetime.fromisoformat(
                str(started_raw).replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            started_at = None

    try:
        saved = rec.import_uploaded_clip(
            raw,
            session_id=str(meta_in.get("session_id") or ""),
            elephant_names=list(names),
            started_at=started_at,
            duration_sec=float(meta_in.get("duration_sec") or 0),
            source=str(meta_in.get("source") or "pi"),
            device_id=str(meta_in.get("device_id") or meta_in.get("session_id") or ""),
            frame_count=int(meta_in.get("frame_count") or 0),
            width=int(meta_in.get("width") or 0),
            height=int(meta_in.get("height") or 0),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {"ok": True, **saved}


@app.get("/watch/{stream_id}", response_class=HTMLResponse)
def watch_page(stream_id: str):
    """浏览器观看：ir-{IMEI} 为红外相机；Pi 直播已关闭，普通 stream 跳转录像库。"""
    camera = parse_ir_stream_id(stream_id)
    if camera:
        return _watch_ir_page(camera, stream_id)
    if not _live_stream_enabled:
        return _watch_clips_redirect_page()
    return _watch_mjpeg_page(stream_id)


def _watch_clips_redirect_page() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="0;url=/watch/clips"/>
  <title>录像库</title>
</head>
<body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;padding:48px;">
  <p>网页直播已关闭。Pi 检测到大象后会自动上传录像。</p>
  <p><a href="/watch/clips" style="color:#64b5f6;">前往录像库 →</a></p>
</body>
</html>"""


def _watch_ir_page(camera_code: str, stream_id: str) -> str:
    safe_id = stream_id.replace("<", "").replace(">", "").replace('"', "")
    safe_cam = camera_code.replace("<", "").replace(">", "").replace('"', "")
    hist_limit = default_ir_history_limit()
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>红外相机识别回放 · {safe_cam}</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
    .page {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1.1rem; font-weight: 500; padding: 14px 16px 8px; margin: 0; text-align: center; }}
    .sub {{ text-align: center; color: #888; font-size: 0.85rem; padding: 0 16px 12px; }}
    .back {{ display: block; text-align: center; color: #64b5f6; font-size: 0.9rem; margin-bottom: 8px; text-decoration: none; }}
    .video-wrap {{ background: #000; line-height: 0; }}
    video {{ width: 100%; height: auto; display: block; background: #000; }}
    .status-box {{
      margin: 12px 16px 16px; padding: 12px 14px; background: #1a1a1a;
      border: 1px solid #333; border-radius: 8px; font-size: 0.9rem; line-height: 1.5;
    }}
    .status-box .label {{ color: #888; }}
    .ok {{ color: #81c784; }}
    .warn {{ color: #ffb74d; }}
    .err {{ color: #ef5350; }}
    .history {{ margin: 0 16px 20px; }}
    .history h2 {{ font-size: 0.95rem; font-weight: 500; margin: 0 0 10px; color: #ccc; }}
    .hist-item {{
      display: block; background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
      padding: 10px 12px; margin-bottom: 8px; color: #eee; cursor: pointer; text-align: left; width: 100%;
      font-family: inherit; font-size: 0.9rem;
    }}
    .hist-item:hover, .hist-item.active {{ border-color: #64b5f6; }}
    .hist-names {{ color: #81c784; }}
    .hist-meta {{ color: #888; font-size: 0.82rem; margin-top: 4px; }}
    .hint {{ text-align: center; color: #666; font-size: 0.82rem; padding: 0 16px 20px; }}
  </style>
</head>
<body>
  <div class="page">
    <a class="back" href="/watch/ir">← 全部红外相机</a>
    <h1>红外相机 · 识别视频</h1>
    <p class="sub">相机 {safe_cam} · 最近 {hist_limit} 条（最新一段自动播放）</p>
    <div class="video-wrap">
      <video id="player" controls playsinline preload="auto"></video>
    </div>
    <div class="status-box" id="statusBox">
      <div><span class="label">状态：</span><span id="stText">加载中…</span></div>
      <div><span class="label">进度：</span><span id="stProg">—</span></div>
      <div><span class="label">识别到：</span><span id="stNames">—</span></div>
    </div>
    <div class="history" id="historyBox" style="display:none;">
      <h2>最近 {hist_limit} 条</h2>
      <div id="historyList"></div>
    </div>
    <p class="hint">新视频处理完成后自动刷新；点击下方条目可切换播放。</p>
  </div>
  <script>
    const cameraCode = "{safe_cam}";
    const histLimit = {hist_limit};
    let lastVersion = "";
    let activeFileId = null;

    function setStatus(text, cls) {{
      const el = document.getElementById("stText");
      el.textContent = text;
      el.className = cls || "";
    }}

    function fmtTime(iso) {{
      if (!iso) return "—";
      try {{
        const d = new Date(iso);
        if (isNaN(d.getTime())) return iso;
        return d.toLocaleString("zh-CN", {{ hour12: false }});
      }} catch (e) {{ return iso; }}
    }}

    function playVideo(url, fileId, names) {{
      activeFileId = fileId;
      const ver = String(fileId || "") + ":" + url;
      if (ver !== lastVersion) {{
        lastVersion = ver;
        const v = document.getElementById("player");
        v.src = url + "?v=" + encodeURIComponent(String(fileId || Date.now()));
        v.load();
        v.play().catch(() => {{}});
      }}
      document.getElementById("stNames").textContent =
        names && names.length ? names.join("、") : "（未识别到个体名）";
      document.querySelectorAll(".hist-item").forEach(btn => {{
        btn.classList.toggle("active", String(btn.dataset.fileId) === String(fileId));
      }});
    }}

    async function loadHistory() {{
      try {{
        const resp = await fetch("/api/v1/uovision/ir/" + encodeURIComponent(cameraCode)
          + "/videos?limit=" + histLimit);
        const data = await resp.json();
        const videos = data.videos || [];
        const box = document.getElementById("historyBox");
        const list = document.getElementById("historyList");
        if (!videos.length) {{
          box.style.display = "none";
          return;
        }}
        box.style.display = "block";
        list.innerHTML = videos.map((v, i) => {{
          const names = (v.elephant_names || []).join("、") || "（未命名）";
          const label = i === 0 ? "最新" : ("第 " + (i + 1) + " 条");
          return '<button type="button" class="hist-item" data-file-id="' + v.file_id + '">'
            + '<div class="hist-names">' + label + " · " + names + '</div>'
            + '<div class="hist-meta">file #' + v.file_id + " · " + fmtTime(v.updated_at) + '</div>'
            + '</button>';
        }}).join("");
        list.querySelectorAll(".hist-item").forEach(btn => {{
          btn.addEventListener("click", () => {{
            const fid = btn.dataset.fileId;
            const item = videos.find(x => String(x.file_id) === String(fid));
            if (item && item.video_url) {{
              setStatus("正在播放 file #" + fid, "ok");
              document.getElementById("stProg").textContent = "100%";
              playVideo(item.video_url, fid, item.elephant_names || []);
            }}
          }});
        }});
      }} catch (e) {{}}
    }}

    async function poll() {{
      try {{
        const resp = await fetch("/api/v1/uovision/ir/" + encodeURIComponent(cameraCode) + "/status");
        const data = await resp.json();
        const st = data.status || "idle";
        if (st === "ready" && data.video_url) {{
          setStatus("已就绪，正在播放最新一段", "ok");
          document.getElementById("stProg").textContent = "100%";
          const names = (data.meta && data.meta.process && data.meta.process.elephant_names) || [];
          if (activeFileId == null || String(activeFileId) === String(data.file_id)) {{
            playVideo(data.video_url, data.file_id, names);
          }}
        }} else if (st === "processing") {{
          setStatus("服务器正在识别打框…", "warn");
          document.getElementById("stProg").textContent = (data.progress_pct || 0) + "%";
          document.getElementById("stNames").textContent = "—";
        }} else if (st === "queued") {{
          setStatus("已上传，排队等待 GPU…", "warn");
          document.getElementById("stProg").textContent = "0%";
        }} else if (st === "error") {{
          setStatus("处理失败：" + (data.error || ""), "err");
        }} else {{
          setStatus("等待相机上传原视频…", "warn");
          document.getElementById("stProg").textContent = "—";
          if (activeFileId == null) document.getElementById("stNames").textContent = "—";
        }}
        await loadHistory();
      }} catch (e) {{
        setStatus("无法连接服务器", "err");
      }}
    }}
    poll();
    setInterval(poll, 3000);
  </script>
</body>
</html>"""


def _watch_mjpeg_page(stream_id: str) -> str:
    """Pi 实时 MJPEG 直播页。"""
    safe_id = stream_id.replace("<", "").replace(">", "").replace('"', "")
    names_json = json.dumps(_load_class_names_for_ui(), ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>大象识别直播 · {safe_id}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #111; color: #eee; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
    .page {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1.15rem; font-weight: 500; padding: 14px 16px 8px; margin: 0; text-align: center; }}
    .video-wrap {{ background: #000; line-height: 0; }}
    .video-wrap img {{ width: 100%; height: auto; display: block; }}
    .legend {{
      padding: 14px 16px 12px;
      background: #181818;
      border-top: 1px solid #2a2a2a;
      min-height: 58px;
      display: flex; flex-wrap: wrap; gap: 10px;
      justify-content: center; align-items: center;
    }}
    .legend-title {{ width: 100%; text-align: center; font-size: 0.85rem; color: #888; margin-bottom: 2px; }}
    .legend-empty {{ color: #666; font-size: 0.95rem; }}
    .chip {{
      display: inline-flex; align-items: center; gap: 8px;
      padding: 7px 16px 7px 12px; border-radius: 999px;
      background: #242424; border: 2px solid var(--chip-color, #666);
      color: #f5f5f5; font-size: 1.05rem;
    }}
    .chip-dot {{
      width: 14px; height: 14px; border-radius: 50%;
      background: var(--chip-color, #666); flex-shrink: 0;
    }}
    .chip.pending {{ border-color: #555; color: #aaa; }}
    .panel {{
      margin: 0 12px 16px; padding: 14px 16px;
      background: #1a1a1a; border: 1px solid #333; border-radius: 10px;
    }}
    .panel h2 {{ margin: 0 0 10px; font-size: 1rem; font-weight: 600; }}
    .panel .sub {{ color: #888; font-size: 0.82rem; margin-bottom: 12px; line-height: 1.45; }}
    .key-row {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
    .key-row input {{
      flex: 1; min-width: 180px; padding: 8px 10px; border-radius: 6px;
      border: 1px solid #444; background: #111; color: #eee;
    }}
    .checks {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(88px, 1fr));
      gap: 8px; margin-bottom: 12px;
    }}
    .checks label {{
      display: flex; align-items: center; gap: 6px; font-size: 0.95rem;
      padding: 6px 8px; background: #242424; border-radius: 6px; cursor: pointer;
    }}
    .checks input {{ width: 16px; height: 16px; accent-color: #4caf50; }}
    .btn-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    button {{
      padding: 8px 14px; border-radius: 6px; border: none; cursor: pointer;
      font-size: 0.9rem; background: #2e7d32; color: #fff;
    }}
    button.secondary {{ background: #444; }}
    button:hover {{ filter: brightness(1.08); }}
    .status {{ margin-top: 10px; font-size: 0.85rem; min-height: 1.2em; }}
    .status.ok {{ color: #81c784; }}
    .status.err {{ color: #ef5350; }}
    .hint {{ text-align: center; color: #666; font-size: 0.85rem; padding: 0 16px 16px; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>大象识别直播 · {safe_id}</h1>
    <div class="video-wrap">
      <img src="/stream/{safe_id}" alt="live stream"/>
    </div>
    <div class="legend" id="legend">
      <div class="legend-title">当前识别</div>
      <span class="legend-empty">等待 Pi 推流…</span>
    </div>
    <div class="panel">
      <h2>今日出场大象（候选名单）</h2>
      <p class="sub">只勾选今天在园区的大象，识别时<strong>仅在这些名字里选择</strong>，可显著减少认错。首次使用请在下方填写 API Key（与云端一致），浏览器会记住。</p>
      <div class="key-row">
        <input type="password" id="apiKey" placeholder="API Key（如 elephant-demo-2026）"/>
        <button type="button" class="secondary" id="btnSaveKey">记住 Key</button>
      </div>
      <div class="checks" id="nameChecks"></div>
      <div class="btn-row">
        <button type="button" id="btnAll">全选</button>
        <button type="button" class="secondary" id="btnNone">全不选</button>
        <button type="button" id="btnApplyStream">应用到此直播</button>
        <button type="button" class="secondary" id="btnApplyGlobal">应用为全局默认</button>
      </div>
      <div class="status" id="cfgStatus"></div>
    </div>
    <p class="hint">画面由树莓派推流，云端检测标注。可在 URL 加 <code>?key=你的密钥</code> 自动填入 Key。</p>
  </div>
  <script>
    const streamId = "{safe_id}";
    const allNames = {names_json};
    const pendingNames = new Set(["识别中...", "大象"]);
    const LS_KEY = "elephant_api_key";

    function escapeHtml(s) {{
      return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }}

    function getApiKey() {{
      return (document.getElementById("apiKey").value || localStorage.getItem(LS_KEY) || "").trim();
    }}

    function initApiKey() {{
      const p = new URLSearchParams(location.search);
      if (p.get("key")) {{
        localStorage.setItem(LS_KEY, p.get("key"));
      }}
      const k = localStorage.getItem(LS_KEY) || "";
      document.getElementById("apiKey").value = k;
    }}

    function renderChecks(selectedSet) {{
      const box = document.getElementById("nameChecks");
      box.innerHTML = allNames.map((n) => {{
        const on = !selectedSet || selectedSet.has(n);
        return (
          '<label><input type="checkbox" data-name="' + escapeHtml(n) + '" ' +
          (on ? "checked" : "") + " /> " + escapeHtml(n) + "</label>"
        );
      }}).join("");
    }}

    function getCheckedNames() {{
      return Array.from(document.querySelectorAll("#nameChecks input:checked"))
        .map((el) => el.getAttribute("data-name"));
    }}

    async function loadAllowed() {{
      try {{
        const url = "/api/v1/allowed-elephants?session_id=" + encodeURIComponent(streamId);
        const resp = await fetch(url);
        const data = await resp.json();
        const allowed = data.allowed_elephants;
        if (!allowed || allowed.length === 0) {{
          renderChecks(null);
          setStatus("当前：未限制（全部 " + allNames.length + " 头）", "ok");
        }} else {{
          renderChecks(new Set(allowed));
          setStatus("当前候选：" + allowed.join("、"), "ok");
        }}
      }} catch (e) {{
        renderChecks(null);
      }}
    }}

    function setStatus(msg, cls) {{
      const el = document.getElementById("cfgStatus");
      el.textContent = msg;
      el.className = "status " + (cls || "");
    }}

    async function saveAllowed(scope) {{
      const key = getApiKey();
      if (!key) {{
        setStatus("请先填写 API Key", "err");
        return;
      }}
      const names = getCheckedNames();
      if (names.length === 0) {{
        setStatus("请至少勾选 1 头大象", "err");
        return;
      }}
      const body = names.length >= allNames.length ? {{ names: null }} : {{ names: names }};
      const url = scope === "global"
        ? "/api/v1/allowed-elephants"
        : "/api/v1/session/" + encodeURIComponent(streamId) + "/allowed-elephants";
      try {{
        const resp = await fetch(url, {{
          method: "PUT",
          headers: {{ "Content-Type": "application/json", "X-Api-Key": key }},
          body: JSON.stringify(body),
        }});
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || resp.statusText);
        localStorage.setItem(LS_KEY, key);
        if (body.names === null) {{
          setStatus("已保存：全部 " + allNames.length + " 头（不限制）", "ok");
          renderChecks(null);
        }} else {{
          setStatus("已保存候选 " + names.length + " 头：" + names.join("、"), "ok");
        }}
      }} catch (e) {{
        setStatus("保存失败：" + e.message, "err");
      }}
    }}

    async function refreshLegend() {{
      const el = document.getElementById("legend");
      try {{
        const resp = await fetch("/api/v1/stream/" + encodeURIComponent(streamId) + "/tracks");
        const data = await resp.json();
        if (!data.active || !data.tracks || data.tracks.length === 0) {{
          el.innerHTML = '<div class="legend-title">当前识别</div><span class="legend-empty">暂无检测到的大象</span>';
          return;
        }}
        const chips = data.tracks.map((t) => {{
          const rgb = (t.color_rgb || [128, 128, 128]).join(",");
          const color = "rgb(" + rgb + ")";
          const pending = pendingNames.has(t.name);
          const cls = pending ? "chip pending" : "chip";
          return (
            '<span class="' + cls + '" style="--chip-color:' + color + '">' +
            '<span class="chip-dot"></span>' + escapeHtml(t.name) + "</span>"
          );
        }}).join("");
        el.innerHTML = '<div class="legend-title">当前识别</div>' + chips;
      }} catch (e) {{
        el.innerHTML = '<div class="legend-title">当前识别</div><span class="legend-empty">连接中断…</span>';
      }}
    }}

    document.getElementById("btnSaveKey").onclick = () => {{
      const k = document.getElementById("apiKey").value.trim();
      if (k) {{ localStorage.setItem(LS_KEY, k); setStatus("API Key 已保存", "ok"); }}
    }};
    document.getElementById("btnAll").onclick = () => {{
      document.querySelectorAll("#nameChecks input").forEach((el) => {{ el.checked = true; }});
    }};
    document.getElementById("btnNone").onclick = () => {{
      document.querySelectorAll("#nameChecks input").forEach((el) => {{ el.checked = false; }});
    }};
    document.getElementById("btnApplyStream").onclick = () => saveAllowed("stream");
    document.getElementById("btnApplyGlobal").onclick = () => saveAllowed("global");

    initApiKey();
    loadAllowed();
    refreshLegend();
    setInterval(refreshLegend, 350);
  </script>
</body>
</html>"""


@app.get("/stream/{stream_id}")
async def mjpeg_stream(
    stream_id: str,
    svc: CloudInferenceService = Depends(get_service),
):
    """MJPEG 视频流（默认已关闭，改用 /watch/clips 录像库）。"""
    if not _live_stream_enabled:
        raise HTTPException(
            status_code=404,
            detail="网页直播已关闭。请访问 /watch/clips 查看 Pi 上传的大象录像。",
        )
    boundary = "frame"

    async def generate():
        interval = 1.0 / max(8.0, svc._stream_fps)
        while True:
            jpeg = svc.render_stream_jpeg(stream_id)
            if jpeg is not None:
                yield (
                    b"--"
                    + boundary.encode()
                    + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                await asyncio.sleep(interval)
            else:
                wf = waiting_frame("等待 Pi 推流...")
                yield (
                    b"--"
                    + boundary.encode()
                    + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                    + wf
                    + b"\r\n"
                )
                await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/snapshot/{stream_id}")
def snapshot(
    stream_id: str,
    svc: CloudInferenceService = Depends(get_service),
):
    """单张最新截图（公开只读）。"""
    if not _live_stream_enabled:
        raise HTTPException(status_code=404, detail="网页直播已关闭")
    jpeg = svc.get_latest_jpeg(stream_id)
    if jpeg is None:
        jpeg = waiting_frame("等待 Pi 推流...")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/v1/stream/{stream_id}/status")
def stream_status(
    stream_id: str,
    svc: CloudInferenceService = Depends(get_service),
):
    return svc.get_stream_status(stream_id)


@app.get("/api/v1/stream/{stream_id}/tracks")
def stream_tracks(
    stream_id: str,
    svc: CloudInferenceService = Depends(get_service),
):
    """网页直播页轮询：当前识别到的大象名字与框颜色。"""
    return svc.get_stream_tracks(stream_id)


# ---------- UOVision 红外相机（厂商 2.2 / 2.3 / 1.3）----------


async def _uovision_photos_handler(
    request: Request,
    upload_route: str,
    x_cameracode: Optional[str],
    x_file_id: Optional[int],
    x_file_size: Optional[int],
    x_is_hq: Optional[int],
    content_type: Optional[str],
):
    """缩略图/预览图上传（profiles 中 PATH_PTHUMB / PATH_VTHUMB → servlet/photos）。"""
    if x_cameracode is None or x_file_id is None or x_file_size is None:
        raise HTTPException(
            status_code=400,
            detail="缺少 Header: X-CameraCode, X-File-Id, X-File-Size",
        )
    body = await request.body()
    cfg = load_uovision_config()

    def _run():
        return handle_photos_upload(
            camera_code=x_cameracode,
            file_id=int(x_file_id),
            file_size=int(x_file_size),
            is_hq=int(x_is_hq or 1),
            content_type=content_type or "",
            body=body,
            cfg=cfg,
            upload_route=upload_route,
        )

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_infer_pool, _run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _uovision_upload_handler(
    request: Request,
    upload_route: str,
    x_cameracode: Optional[str],
    x_file_id: Optional[int],
    x_file_size: Optional[int],
    x_is_hq: Optional[int],
    content_type: Optional[str],
    svc: CloudInferenceService,
):
    """
    original (2.2)：平台 acquirefile 后相机回传，或相机按 2.2 规范 POST。
    original2 (1.3)：相机主动上报原视频/原图（Header 名可能为小写，FastAPI 已兼容）。
    二者 Body 与识别流程相同，故共用本处理函数。
    """
    if x_cameracode is None or x_file_id is None or x_file_size is None or x_is_hq is None:
        raise HTTPException(
            status_code=400,
            detail="缺少 Header: X-CameraCode, X-File-Id, X-File-Size, X-Is-Hq",
        )
    body = await request.body()
    cfg = load_uovision_config()

    def _run():
        return handle_original_upload(
            camera_code=x_cameracode,
            file_id=int(x_file_id),
            file_size=int(x_file_size),
            is_hq=int(x_is_hq),
            content_type=content_type or "",
            body=body,
            infer_fn=svc.infer_still_jpeg,
            cfg=cfg,
            upload_route=upload_route,
            on_video_saved=(
                (lambda **kw: _enqueue_uovision_video(
                    camera_code=kw["camera_code"],
                    source_path=Path(kw["source_path"]),
                    file_id=kw["file_id"],
                    upload_route=kw.get("upload_route", upload_route),
                ))
                if int(x_is_hq) == 2
                else None
            ),
        )

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_infer_pool, _run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/servlet/photos")
async def uovision_photos(
    request: Request,
    x_cameracode: Optional[str] = Header(default=None, alias="X-CameraCode"),
    x_file_id: Optional[int] = Header(default=None, alias="X-File-Id"),
    x_file_size: Optional[int] = Header(default=None, alias="X-File-Size"),
    x_is_hq: Optional[int] = Header(default=None, alias="X-Is-Hq"),
    content_type: Optional[str] = Header(default=None, alias="Content-Type"),
):
    """UOVision 缩略图/预览图（SD profiles PATH_PTHUMB / PATH_VTHUMB）。"""
    return await _uovision_photos_handler(
        request,
        "photos",
        x_cameracode,
        x_file_id,
        x_file_size,
        x_is_hq,
        content_type,
    )


@app.post("/servlet/original")
async def uovision_original(
    request: Request,
    x_cameracode: Optional[str] = Header(default=None, alias="X-CameraCode"),
    x_file_id: Optional[int] = Header(default=None, alias="X-File-Id"),
    x_file_size: Optional[int] = Header(default=None, alias="X-File-Size"),
    x_is_hq: Optional[int] = Header(default=None, alias="X-Is-Hq"),
    content_type: Optional[str] = Header(default=None, alias="Content-Type"),
    svc: CloudInferenceService = Depends(get_service),
):
    """UOVision 2.2：原图/原视频上传（平台拉取或相机回调）。"""
    return await _uovision_upload_handler(
        request,
        "original",
        x_cameracode,
        x_file_id,
        x_file_size,
        x_is_hq,
        content_type,
        svc,
    )


@app.post("/servlet/original2")
async def uovision_original2(
    request: Request,
    x_cameracode: Optional[str] = Header(default=None, alias="X-CameraCode"),
    x_file_id: Optional[int] = Header(default=None, alias="X-File-Id"),
    x_file_size: Optional[int] = Header(default=None, alias="X-File-Size"),
    x_is_hq: Optional[int] = Header(default=None, alias="X-Is-Hq"),
    content_type: Optional[str] = Header(default=None, alias="Content-Type"),
    svc: CloudInferenceService = Depends(get_service),
):
    """UOVision 1.3：相机主动上报原图/原视频（与 original 处理相同）。"""
    return await _uovision_upload_handler(
        request,
        "original2",
        x_cameracode,
        x_file_id,
        x_file_size,
        x_is_hq,
        content_type,
        svc,
    )


async def _parse_heartbeat_payload(request: Request) -> dict[str, Any]:
    ct = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ct in ("application/x-www-form-urlencoded", "multipart/form-data"):
        form = await request.form()
        return {str(k): form.get(k) for k in form.keys()}
    body = await request.body()
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (UnicodeDecodeError, json.JSONDecodeError):
        text = body.decode("utf-8", errors="replace").strip()
        if "=" in text and "&" in text:
            out: dict[str, Any] = {}
            for part in text.split("&"):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                out[k] = v
            return out
        return {"raw": text[:500]} if text else {}


def _client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    if request.client:
        return request.client.host or ""
    return ""


async def _uovision_heartbeat_handler(
    request: Request,
    payload: dict[str, Any],
    *,
    require_json: bool = True,
):
    """UOVision 1.4：相机定时 POST JSON 上报状态（/camera/heatbeat）。"""
    _ = require_json  # 兼容旧调用；实际按 Body 解析，不强制 Content-Type
    cfg = load_uovision_config()
    try:
        return handle_heartbeat(payload=payload, cfg=cfg, client_ip=_client_ip(request))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/camera/heatbeat")
@app.post("/camera/heartbeat")
@app.post("/servlet/heartbeat")
async def uovision_heartbeat_post(request: Request):
    """
    UOVision 1.4 心跳（对接文档路径 /camera/heatbeat）。
    POST application/json，Body.cameraCode 必填。
    /servlet/heartbeat 为同文档 curl 示例中的兼容路径。
    """
    payload = await _parse_heartbeat_payload(request)
    return await _uovision_heartbeat_handler(request, payload, require_json=True)


@app.get("/servlet/heartbeat")
async def uovision_heartbeat_get(
    request: Request,
    cameraCode: Optional[str] = None,
    camera_code: Optional[str] = None,
):
    """联调备用：GET 携带 cameraCode 与状态字段。"""
    code = (cameraCode or camera_code or "").strip()
    params = dict(request.query_params)
    if code:
        params.setdefault("cameraCode", code)
    return await _uovision_heartbeat_handler(request, params, require_json=False)


@app.get("/api/v1/uovision/ir/{camera_code}/heartbeat")
def uovision_ir_heartbeat(camera_code: str):
    """单台相机最近心跳与在线状态（网页/运维，无需 API Key）。"""
    return get_heartbeat_status(camera_code)


@app.get("/api/v1/uovision/heartbeats")
def uovision_heartbeats(_: None = Depends(verify_api_key)):
    """全部已注册相机的心跳摘要（需 API Key）。"""
    items = load_camera_registry()
    codes = [c.imei for c in items] if items else None
    return {"heartbeats": list_heartbeat_status(codes)}


@app.get("/api/v1/uovision/ir/{camera_code}/status")
def uovision_ir_status(camera_code: str):
    """红外相机最新视频处理状态（网页轮询，无需 API Key）。"""
    pipe = get_video_pipeline()
    st = pipe.get_state(camera_code)
    video_path = pipe.get_latest_video_path(camera_code)
    out = {
        "camera_code": st.camera_code,
        "stream_id": ir_stream_id(st.camera_code),
        "watch_url": st.watch_url or f"/watch/{ir_stream_id(st.camera_code)}",
        "status": st.status,
        "file_id": st.file_id,
        "progress_pct": st.progress_pct,
        "frames_done": st.frames_done,
        "frames_total": st.frames_total,
        "error": st.error or None,
        "updated_at": st.updated_at,
        "upload_route": st.upload_route or None,
        "meta": st.meta or None,
        "video_ready": video_path is not None,
    }
    if video_path is not None and st.status == "ready":
        out["video_url"] = f"/api/v1/uovision/ir/{camera_code}/latest.mp4"
    return out


@app.get("/api/v1/uovision/ir/{camera_code}/videos")
def uovision_ir_list_videos(camera_code: str, limit: int = 0):
    """单台相机最近若干条识别视频（网页浏览，无需 API Key）。"""
    lim = int(limit) if limit > 0 else default_ir_history_limit()
    pipe = get_video_pipeline()
    videos = pipe.list_videos(camera_code, limit=lim)
    return {
        "camera_code": camera_code.strip(),
        "limit": lim,
        "videos": videos,
    }


@app.get("/api/v1/uovision/ir/{camera_code}/videos/{file_id}.mp4")
def uovision_ir_video_by_id(camera_code: str, file_id: int):
    """播放指定 file_id 的识别 MP4。"""
    pipe = get_video_pipeline()
    path = pipe.get_video_path(camera_code, file_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="视频不存在或已过期删除")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"ir_{camera_code}_{file_id}.mp4",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/v1/uovision/ir/{camera_code}/latest.mp4")
def uovision_ir_latest_video(camera_code: str):
    """播放最新一段识别后的 MP4（公开只读）。"""
    pipe = get_video_pipeline()
    path = pipe.get_latest_video_path(camera_code)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="尚无已处理的视频")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"ir_{camera_code}_latest.mp4",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/v1/uovision/ir/{camera_code}/start-recording")
def uovision_ir_start_recording(
    camera_code: str,
    camera_name: str = "",
    video_length_sec: int = 0,
    trigger_capture: bool = True,
    _: None = Depends(verify_api_key),
):
    """
    1) 按机型 POST /modifySettings{Model}（videoLength 官方仅 5–60 秒）
    2) 可选 POST /getFile 远程触发拍摄
    """
    rec = find_camera(camera_code)
    imei = rec.imei if rec else camera_code.strip()
    cname = normalize_uovision_camera_name(
        camera_name or (rec.camera_name if rec else ""),
        camera_id=rec.id if rec else "",
    )
    vlen = video_length_sec or default_video_length_sec()
    vlen = max(5, min(60, int(vlen)))
    if rec:
        return configure_camera_for_ir(
            rec,
            video_length_sec=vlen,
            trigger_capture=trigger_capture,
        )
    overrides = {
        "cameraMode": 2,
        "cameraName": cname,
        "videoLength": vlen,
        "sendVideoSize": 2,
        "sendPhotoSize": 1,
        "remoteControl": 1,
    }
    settings = merge_settings_for_modify(imei, overrides)
    modify_result = forward_modify_settings(settings)
    trigger_result = forward_get_file(imei) if trigger_capture else None
    return {
        "camera_code": imei,
        "stream_id": ir_stream_id(imei),
        "watch_url": f"/watch/{ir_stream_id(imei)}",
        "video_length_sec": vlen,
        "settings_sent": settings,
        "modify": modify_result,
        "trigger_get_file": trigger_result,
    }


@app.post("/acquirefile")
@app.post("/acquireFile")
async def uovision_acquire_file(
    payload: dict,
    _: None = Depends(verify_api_key),
):
    """向 open.uovcloud.com 下发 POST /acquireFile?cameraCode=&fileId=&fileType=。"""
    return forward_acquire_file(payload)


@app.post("/modifysettings")
async def uovision_modify_settings(
    settings: dict,
    _: None = Depends(verify_api_key),
):
    """按机型转发 POST /modifySettings{Model}（须完整 JSON，建议先 GET 再 merge）。"""
    if "cameraCode" not in settings:
        raise HTTPException(status_code=400, detail="settings 须包含 cameraCode（IMEI）且为完整参数集")
    return forward_modify_settings(settings)


@app.get("/api/v1/uovision/query-model")
def uovision_query_model(camera_code: str, _: None = Depends(verify_api_key)):
    """GET open.uovcloud.com/queryModel?cameraCode=IMEI"""
    return query_model(camera_code)


@app.get("/api/v1/uovision/camera-info")
def uovision_camera_info(camera_code: str, _: None = Depends(verify_api_key)):
    """GET /getSettings{Model}，返回 iccid、电量、GPS 等。"""
    return get_camera_settings(camera_code)


@app.get("/api/v1/uovision/probe-iccid")
def uovision_probe_iccid(
    iccid: str,
    candidate_imeis: str,
    _: None = Depends(verify_api_key),
):
    """
    用 ICCID 反查 IMEI：对每个候选 IMEI 调 getSettings，比对 iccid 字段。
    candidate_imeis 逗号分隔，如 868274062084688,868274062084689
    """
    imeis = [x.strip() for x in candidate_imeis.split(",") if x.strip()]
    if not imeis:
        raise HTTPException(status_code=400, detail="candidate_imeis 不能为空")
    return probe_imei_by_iccid(imeis, iccid)


@app.get("/api/v1/uovision/cameras")
def uovision_list_cameras():
    """注册表中的红外相机及当前识别流水线状态（网页总览用）。"""
    return {"cameras": list_cameras_with_status()}


@app.get("/api/v1/uovision/verify-all")
def uovision_verify_all(_: None = Depends(verify_api_key)):
    """逐台 queryModel + getSettings，核对 ICCID。"""
    items = load_camera_registry()
    if not items:
        raise HTTPException(status_code=404, detail="未找到 uovision_cameras.json")
    results = [verify_camera(c) for c in items]
    return {
        "count": len(results),
        "all_ok": all(r.get("ok") for r in results),
        "cameras": results,
    }


@app.post("/api/v1/uovision/register-all")
def uovision_register_all(body: dict | None = None, _: None = Depends(verify_api_key)):
    """
    POST /add 将 uovision_cameras.json 中的 IMEI 注册到开放平台。
    body: { "model": "UML7" }  机型须与机身一致，默认读 UOVISION_CAMERA_MODEL
    """
    body = body or {}
    model = str(body.get("model") or "").strip()
    return register_all_cameras(model=model)


@app.post("/api/v1/uovision/setup-all")
def uovision_setup_all(body: dict, _: None = Depends(verify_api_key)):
    """
    批量 setServerBase + 可选下发识别参数。
    body: {
      "dataSvr": "http://120.196.88.140:9998",
      "register": true,
      "model": "UML7",
      "configure": true,
      "trigger_capture": false
    }
    """
    return setup_all_cameras(
        data_svr=str(body.get("dataSvr") or body.get("data_svr") or "").strip(),
        configure=bool(body.get("configure", True)),
        register=bool(body.get("register", False)),
        trigger_capture=bool(body.get("trigger_capture", False)),
        video_length_sec=int(body["video_length_sec"]) if body.get("video_length_sec") else None,
        model=str(body.get("model") or "").strip(),
    )


@app.post("/api/v1/uovision/ir/{camera_code}/configure")
def uovision_ir_configure(
    camera_code: str,
    trigger_capture: bool = False,
    video_length_sec: int = 0,
    _: None = Depends(verify_api_key),
):
    """单台相机：merge 当前参数 + 大象识别推荐值 + modifySettings。"""
    rec = find_camera(camera_code)
    if rec is None:
        rec = IrCameraRecord(
            id="",
            label=camera_code,
            sn="",
            imei=camera_code.strip(),
            iccid="",
            camera_name="CAM001",
        )
    vlen = video_length_sec or None
    return configure_camera_for_ir(rec, video_length_sec=vlen, trigger_capture=trigger_capture)


@app.post("/api/v1/uovision/set-server-base")
def uovision_set_server_base(body: dict, _: None = Depends(verify_api_key)):
    """
    POST /setServerBase：配置相机原图/原视频上传前缀（dataSvr）。
    body: { "imeiList": ["868..."], "dataSvr": "http://120.196.88.140:9998" }
    """
    imei_list = body.get("imeiList") or body.get("imei_list") or []
    if isinstance(imei_list, str):
        imei_list = [x.strip() for x in imei_list.split(",") if x.strip()]
    data_svr = str(body.get("dataSvr") or body.get("data_svr") or os.environ.get("UOVISION_CALLBACK_BASE", "")).strip()
    ctrl_svr = str(body.get("ctrlSvr") or body.get("ctrl_svr") or "").strip()
    return set_server_base(imei_list=imei_list, data_svr=data_svr, ctrl_svr=ctrl_svr)


@app.post("/api/v1/uovision/ir/{camera_code}/trigger-capture")
def uovision_trigger_capture(camera_code: str, _: None = Depends(verify_api_key)):
    """POST /getFile?cameraCode= 远程取图/触发拍摄。"""
    return forward_get_file(camera_code)


@app.get("/api/v1/uovision/settings-template")
def uovision_settings_template(
    camera_code: str,
    camera_name: str = "M11",
    video_length_sec: int = 0,
    _: None = Depends(verify_api_key),
):
    """返回 modifySettings 完整 JSON 模板（勿直接套用 GET 返回值）。"""
    vlen = video_length_sec or default_video_length_sec()
    return build_modifysettings_template(
        camera_code, camera_name, video_length_sec=vlen
    )


@app.get("/api/v1/uovision/events")
def uovision_events(
    limit: int = 50,
    camera_code: Optional[str] = None,
    _: None = Depends(verify_api_key),
):
    """最近相机上传与识别结果（内存索引，重启清空）。"""
    return {"events": list_recent_events(limit=limit, camera_code=camera_code)}


@app.get("/api/v1/class-names")
def list_class_names(svc: CloudInferenceService = Depends(get_service)):
    """全部象名（直播页勾选，无需 API Key）。"""
    tr = svc._ensure_tracker()
    if tr.classifier is None:
        return {"names": _load_class_names_for_ui()}
    return {"names": tr.classifier.class_names}


@app.get("/api/v1/allowed-elephants")
def get_allowed_elephants(
    session_id: Optional[str] = None,
    svc: CloudInferenceService = Depends(get_service),
):
    """查询候选象（直播页读取，无需 API Key）。"""
    if session_id:
        names = svc.get_session_allowed(session_id)
        return {
            "session_id": session_id,
            "allowed_elephants": names,
            "restricted": names is not None,
        }
    default = svc.get_default_allowed()
    return {
        "allowed_elephants": default,
        "restricted": default is not None,
        "all_classes": default is None,
    }


@app.put("/api/v1/allowed-elephants")
def set_allowed_elephants_global(
    body: dict,
    _: None = Depends(verify_api_key),
    svc: CloudInferenceService = Depends(get_service),
):
    """
    设置全局候选象（园区当日出场象）。
    body: {"names": ["印东", "凯恩"]} 或 {"names": null} 恢复 17 类全开
    """
    raw = body.get("names")
    if raw is None:
        svc.set_default_allowed(None)
        return {"ok": True, "allowed_elephants": None, "all_classes": True}
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="names 应为数组或 null")
    names = [str(x).strip() for x in raw if str(x).strip()]
    svc.set_default_allowed(names)
    return {"ok": True, "allowed_elephants": names, "count": len(names)}


@app.put("/api/v1/session/{session_id}/allowed-elephants")
def set_allowed_elephants_session(
    session_id: str,
    body: dict,
    _: None = Depends(verify_api_key),
    svc: CloudInferenceService = Depends(get_service),
):
    """为某一路 Pi 直播单独设候选象；body.names=null 表示跟随全局。"""
    raw = body.get("names")
    if raw is None:
        svc.set_session_allowed(session_id, None)
    elif isinstance(raw, list):
        names = [str(x).strip() for x in raw if str(x).strip()]
        svc.set_session_allowed(session_id, names)
    else:
        raise HTTPException(status_code=400, detail="names 应为数组或 null")
    return {
        "ok": True,
        "session_id": session_id,
        "allowed_elephants": svc.get_session_allowed(session_id),
    }


def main():
    global _service, _api_key, _clip_recorder

    parser = argparse.ArgumentParser(description="大象识别云端 API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="best_elephant_model.pth")
    parser.add_argument("--classes", default="class_names.json")
    parser.add_argument("--yolo-weights", default="yolov8m.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--infer-max-width", type=int, default=1280)
    parser.add_argument("--detect-interval", type=int, default=1)
    parser.add_argument("--recog-interval", type=int, default=1)
    parser.add_argument("--min-conf", type=float, default=42.0, help="分类置信度阈值(%)")
    parser.add_argument("--min-margin", type=float, default=12.0, help="top1-top2 间隔阈值(%)")
    parser.add_argument(
        "--freeze-locked",
        action="store_true",
        default=True,
        help="名字锁定后不再改（默认开启）",
    )
    parser.add_argument(
        "--no-freeze-locked",
        action="store_false",
        dest="freeze_locked",
        help="锁定后仍允许换名（易闪名，不推荐）",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="云端只做检测，不做个体分类",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CLOUD_API_KEY", ""),
        help="请求头 X-Api-Key；不设则不做鉴权（仅内网调试）",
    )
    parser.add_argument(
        "--stream-max-width",
        type=int,
        default=854,
        help="网页 MJPEG 最大宽度（越小越流畅）",
    )
    parser.add_argument(
        "--stream-jpeg-quality",
        type=int,
        default=68,
        help="网页 MJPEG JPEG 质量",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=int(os.environ.get("ELEPHANT_GPU", "0")),
        help="使用的 GPU 编号（0 或 1，双卡默认用 0）",
    )
    parser.add_argument(
        "--stream-fps",
        type=float,
        default=20.0,
        help="网页 MJPEG 目标帧率（推理慢时用外推补帧）",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="低显存：YOLO 用 predict、分类器放 CPU（共享 GPU 推荐）",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="全部用 CPU 推理（最省显存，较慢）",
    )
    parser.add_argument(
        "--allowed-elephants",
        default=os.environ.get("ELEPHANT_ALLOWED", ""),
        help='候选象，逗号分隔，如 "印东,凯恩,威风"；或 JSON 文件路径；空=17类全开',
    )
    args = parser.parse_args()

    allowed = parse_allowed_elephants(args.allowed_elephants or None)

    import torch

    print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mark = " <-- use" if i == args.gpu else ""
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}{mark}")
    else:
        print("  [WARN] 当前为 CPU 版 PyTorch，请在服务器运行: bash install_gpu_pytorch.sh")

    _api_key = (args.api_key or "").strip()
    print("正在加载云端推理模型（首次较慢）…")
    _service = CloudInferenceService(
        model_path=args.model,
        class_names_path=args.classes,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        infer_max_width=args.infer_max_width,
        detect_interval=args.detect_interval,
        recog_interval=args.recog_interval,
        classify=not args.no_classify,
        min_confidence=args.min_conf,
        min_margin=args.min_margin,
        freeze_when_locked=args.freeze_locked,
        stream_max_width=args.stream_max_width,
        stream_jpeg_quality=args.stream_jpeg_quality,
        gpu_id=args.gpu,
        stream_fps=args.stream_fps,
        low_vram=args.low_vram,
        force_cpu=args.force_cpu,
        allowed_elephants=allowed,
    )
    _service.warmup()
    _clip_recorder = None
    if ElephantClipRecorder is not None:
        _clip_recorder = ElephantClipRecorder.from_env()
        if _clip_recorder is None:
            root = Path(os.environ.get("ELEPHANT_CLIP_DIR", "data/clips"))
            _clip_recorder = ElephantClipRecorder(data_dir=root)
        set_clip_recorder(_clip_recorder)
        purged = _clip_recorder.purge_older_than(_clip_retention_days)
        if purged.get("removed"):
            print(f"  录像清理: 删除过期 {purged['removed']} 条（>{_clip_retention_days} 天）")
        print(f"  录像库: {_clip_recorder.data_dir.resolve()} | 保留 {_clip_retention_days} 天")
        infer_rec = os.environ.get("ELEPHANT_CLIP_RECORD_INFER", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        print(f"  服务端随流录制: {'开' if infer_rec else '关（由 Pi 上传）'}")
    else:
        print("  大象录像: 跳过（未上传 elephant_clip_recorder.py）")

    def _clip_purge_loop() -> None:
        while True:
            time.sleep(3600)
            rec = get_clip_recorder()
            if rec is not None:
                try:
                    rec.purge_older_than(_clip_retention_days)
                except Exception:
                    pass

    def _run_uovision_purge() -> dict[str, Any]:
        result = purge_uovision_older_than(_uovision_retention_days)
        try:
            pipe = get_video_pipeline()
            for code in result.get("camera_codes_reset") or []:
                pipe.reload_state(str(code))
        except RuntimeError:
            pass
        return result

    threading.Thread(target=_clip_purge_loop, daemon=True, name="clip-purge").start()

    def _uovision_purge_loop() -> None:
        while True:
            time.sleep(3600)
            try:
                purged = _run_uovision_purge()
                if purged.get("removed_files"):
                    print(
                        f"  红外清理: 删除 {purged['removed_files']} 个文件，"
                        f"释放 {purged.get('freed_mb', 0)} MB（>{_uovision_retention_days} 天）"
                    )
            except Exception:
                pass

    init_video_pipeline(_process_ir_video, load_uovision_config())
    uov_purged = _run_uovision_purge()
    if uov_purged.get("removed_files"):
        print(
            f"  红外清理: 删除过期 {uov_purged['removed_files']} 个文件，"
            f"释放 {uov_purged.get('freed_mb', 0)} MB（>{_uovision_retention_days} 天）"
        )
    uov_cfg = load_uovision_config()
    print(
        f"  红外存储: {uov_cfg.data_dir.resolve()} | 保留 {_uovision_retention_days} 天"
        f"（UOVISION_RETENTION_DAYS）"
    )
    threading.Thread(target=_uovision_purge_loop, daemon=True, name="uovision-purge").start()
    print(f"  UOVision 网关: {uov_cfg.gateway_url or '(未配置)'}")
    print(f"  红外回调地址: {os.environ.get('UOVISION_CALLBACK_BASE', '(见 setServerBase)')}")
    if allowed:
        print("候选象模式: " + ", ".join(allowed))
    else:
        print("候选象: 未限制（全部类别）")
    print(f"API: http://{args.host}:{args.port}")
    print(f"  网页直播: {'开' if _live_stream_enabled else '关 → 使用 /watch/clips'}")
    print("  GET  /health")
    print("  GET  /watch/clips             Pi 大象录像库（上传后在此观看）")
    print("  POST /api/v1/clips/upload     Pi 上传标注 MP4")
    print("  GET  /api/v1/clips           录像列表 JSON")
    if _live_stream_enabled:
        print("  GET  /watch/{{stream_id}}   MJPEG 直播页")
        print("  GET  /stream/{{stream_id}}  MJPEG 流")
    print("  POST /api/v1/infer  (multipart: image=JPEG, session_id=流 ID)")
    print("  POST /api/v1/reset  (form: session_id)")
    print("  POST /servlet/photos      UOVision 缩略图/预览（发送视频前可能先走此路径）")
    print("  POST /servlet/original   UOVision 原图/原视频（2.2）")
    print("  POST /servlet/original2  UOVision 相机主动上报（1.3，与 original 相同处理）")
    print("  POST /camera/heatbeat       UOVision 相机心跳（1.4，文档路径）")
    print("  POST /servlet/heartbeat     UOVision 相机心跳（1.4，curl 示例兼容）")
    print("  GET  /watch/ir                 5 台红外相机总览")
    print("  GET  /api/v1/uovision/ir/{{IMEI}}/videos     最近 N 条识别视频列表")
    print("  GET  /watch/ir-{{IMEI}}        单台识别视频（含最近历史）")
    print("  POST /api/v1/uovision/ir/{{IMEI}}/start-recording  下发录像参数+触发 getFile")
    print("  POST /acquireFile         转发 open.uovcloud.com（需 UOVISION_GATEWAY_URL）")
    print("  GET  /api/v1/uovision/camera-info?camera_code=IMEI  查 ICCID/状态")
    print("  GET  /api/v1/uovision/events  最近相机识别记录")
    print("  GET/PUT /api/v1/allowed-elephants  候选象名单（提高园区识别率）")
    if _api_key:
        print("  鉴权: 请求头 X-Api-Key")
    else:
        print("  警告: 未设置 API Key，请勿暴露到公网")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
