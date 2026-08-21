"""云端：在推理帧上绘制检测框与中文标签，供网页 MJPEG 直播。"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    for p in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
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


def _draw_label(frame_bgr: np.ndarray, x: int, y: int, text: str, color_bgr, font_size=22):
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


def render_annotated_frame(
    frame_bgr: np.ndarray,
    tracks: list[dict],
    info_line: str = "",
) -> np.ndarray:
    out = frame_bgr.copy()
    for tr in tracks:
        x, y, w, h = tr["bbox"]
        color = tuple(tr.get("color_bgr", [80, 255, 100]))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        _draw_label(out, x, y, str(tr.get("name", "识别中...")), color)
    if info_line:
        cv2.putText(
            out,
            info_line,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
    return out


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80, max_width: int = 0) -> bytes:
    img = frame_bgr
    if max_width > 0:
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(
                img,
                (max_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return buf.tobytes()


def waiting_frame(text: str = "等待 Pi 推流...") -> bytes:
    img = np.zeros((480, 854, 3), dtype=np.uint8)
    img[:] = (32, 32, 32)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    font = _get_cjk_font(28)
    l, t, r, b = dr.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    x = max(10, (854 - tw) // 2)
    y = (480 - th) // 2
    dr.text((x - l, y - t), text, font=font, fill=(200, 200, 200))
    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return encode_jpeg(out)


class StreamTrackSmoother:
    """网页 MJPEG：在两次推理之间外推框位置，提高观感帧率与跟手度。"""

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
