"""YOLO 裁象体（import_new_photos 用，与线上一致）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def normalize_image_path(raw: str) -> Path:
    s = raw.strip().strip('"').strip("'")
    for ch in ("\u201c", "\u201d", "\u2018", "\u2019"):
        s = s.strip(ch)
    return Path(s)


def imread_bgr(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def pad_roi_bgr(
    frame_bgr: np.ndarray, x: int, y: int, w: int, h: int, pad_ratio: float = 0.12
) -> np.ndarray | None:
    pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(w + 2 * pad_w, frame_bgr.shape[1] - x)
    h = min(h + 2 * pad_h, frame_bgr.shape[0] - y)
    if w < 24 or h < 24:
        return None
    roi = frame_bgr[y : y + h, x : x + w]
    return roi if roi.size else None


def yolo_elephant_roi_bgr(
    frame_bgr: np.ndarray,
    yolo_weights: str = "yolov8n.pt",
    cuda_device: int = -1,
    pad_ratio: float = 0.12,
) -> tuple[np.ndarray | None, str]:
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "未安装 ultralytics，无法 YOLO 裁剪"

    if int(cuda_device) >= 0 and torch.cuda.is_available():
        device = max(0, min(int(cuda_device), torch.cuda.device_count() - 1))
    else:
        device = "cpu"

    model = YOLO(yolo_weights)
    results = model.predict(
        frame_bgr,
        verbose=False,
        imgsz=640,
        device=device,
        half=(device != "cpu"),
    )
    names = model.names
    best = None
    best_area = 0
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cname = str(names[cls_id]).lower()
            if cname not in {"elephant", "大象"}:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2 - x1, y2 - y1, float(box.conf[0]))

    if best is None:
        return None, "YOLO 未检测到大象"

    x, y, w, h, conf = best
    roi = pad_roi_bgr(frame_bgr, x, y, w, h, pad_ratio=pad_ratio)
    if roi is None:
        return None, "检测框过小"
    return roi, f"YOLO ROI (conf={conf:.2f})"
