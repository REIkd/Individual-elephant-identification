"""
基于 YOLO 的大象检测与跟踪 + 个体分类器。
COCO 中 elephant 类别 id 为 20（原代码误用 21 会检成 bear）。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from predict import ElephantClassifier

# 缓存字体，避免每帧重复加载
_CJK_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """OpenCV 无法绘制中文；使用系统 TrueType 字体（Windows 常见雅黑/黑体）。"""
    if size in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[size]

    chosen: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    custom = os.environ.get("ELEPHANT_FONT", "").strip()
    if custom and Path(custom).is_file():
        try:
            chosen = ImageFont.truetype(custom, size)
        except OSError:
            pass
    if chosen is None and os.name == "nt":
        candidates = [
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\simsun.ttc"),
        ]
    elif chosen is None:
        candidates = [
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/System/Library/Fonts/PingFang.ttc"),
        ]
    else:
        candidates = []

    if chosen is None:
        for p in candidates:
            if p.is_file():
                try:
                    chosen = ImageFont.truetype(str(p), size)
                    break
                except OSError:
                    continue

    if chosen is None:
        chosen = ImageFont.load_default()

    _CJK_FONT_CACHE[size] = chosen
    return chosen


def _draw_label_chinese_bgr(
    frame_bgr: np.ndarray,
    x: int,
    y_box_top: int,
    text: str,
    color_bgr: tuple[int, int, int],
    font_size: int = 22,
) -> None:
    """在 bbox 上沿绘制中文标签（小图块贴回原图，避免整帧 PIL 缩放拖慢预览 FPS）。"""
    font = _get_cjk_font(font_size)
    pad_x, pad_y = 6, 4
    fill_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    scratch = Image.new("RGB", (4, 4))
    dr0 = ImageDraw.Draw(scratch)
    left, top, right, bot = dr0.textbbox((0, 0), text, font=font)
    tw, th = right - left, bot - top
    pw = tw + 2 * pad_x
    ph = th + 2 * pad_y

    pil_label = Image.new("RGB", (pw, ph), fill_rgb)
    dr = ImageDraw.Draw(pil_label)
    dr.text((pad_x - left, pad_y - top), text, font=font, fill=(255, 255, 255))

    y_top = int(y_box_top) - ph - 2
    if y_top < 0:
        y_top = int(y_box_top) + 4
    x_left = int(x) + 4

    fh, fw = frame_bgr.shape[:2]
    iy1 = max(0, min(y_top, fh - 1))
    ix1 = max(0, min(x_left, fw - 1))
    iy2 = min(fh, y_top + ph)
    ix2 = min(fw, x_left + pw)
    if iy2 <= iy1 or ix2 <= ix1:
        return

    ly1 = iy1 - y_top
    lx1 = ix1 - x_left
    ly2 = ly1 + (iy2 - iy1)
    lx2 = lx1 + (ix2 - ix1)

    arr = np.asarray(pil_label)
    patch_rgb = arr[ly1:ly2, lx1:lx2]
    frame_bgr[iy1:iy2, ix1:ix2] = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

# 高区分度 BGR 调色板（按类别名顺序分配）
_BGR_PALETTE = [
    (80, 80, 255),
    (80, 255, 80),
    (255, 80, 80),
    (0, 255, 255),
    (255, 0, 255),
    (255, 180, 60),
    (180, 100, 255),
    (100, 200, 255),
    (42, 180, 130),
    (200, 150, 100),
    (100, 100, 220),
    (50, 200, 200),
    (220, 100, 200),
    (150, 220, 80),
    (90, 130, 255),
    (255, 140, 140),
]


class YOLOElephantTracker:
    def __init__(
        self,
        model_path="best_elephant_model.pth",
        class_names_path="class_names.json",
        use_yolo=True,
        yolo_weights="yolov8m.pt",
        min_confidence: float = 30.0,
        min_margin: float = 4.0,
        recognition_interval: int = 4,
        ema_alpha: float = 0.12,
        min_det_conf: float = 0.42,
        freeze_recognition_when_locked: bool = True,
        show_overlay_confidence: bool = False,
        # ---- 实时性（高分辨率 USB 摄像头）----
        # YOLO 输入短边缩放目标；越小越快，过大的话可配合 infer_max_width
        yolo_imgsz: int = 640,
        # >0 时先将画面缩放到不超过该宽度再检测，再把框映射回原始分辨率（省算力）
        infer_max_width: int = 0,
        # 每 N 帧跑一次检测/跟踪更新；中间帧只沿用上一帧框（提高观感帧率，框略滞后）
        detect_interval: int = 1,
        # 连续若干帧未再被检测到则删除轨迹，避免画面上残留「幻影框」（拿开照片后仍显示框）
        max_lost_frames: int = 30,
        # 连续 N 次「真实跑完 YOLO 且大象框数量为 0」则立刻清空所有轨迹；0 表示关闭（处理长视频建议关）
        clear_after_empty_detects: int = 0,
        # USB 摄像头常用 MJPEG 才能在 1080p 下达到 30fps+；若花屏可改 False
        camera_use_mjpeg: bool = True,
        # >0 时在打开摄像头后请求采集分辨率（画面清晰度；检测仍可用 infer_max_width 缩小）
        camera_width: int = 0,
        camera_height: int = 0,
        # False 时不加载 ResNet、不做个体分类，仅 YOLO 检大象框（树莓派更易接近实时）
        classify_elephants: bool = True,
        cuda_device: int = 0,
        classifier_cuda_device: int | None = None,
        yolo_use_predict: bool = False,
    ):
        print("正在初始化大象跟踪系统...")

        self.classify_elephants = bool(classify_elephants)
        cls_dev = (
            int(classifier_cuda_device)
            if classifier_cuda_device is not None
            else int(cuda_device)
        )
        if self.classify_elephants:
            self.classifier = ElephantClassifier(
                model_path, class_names_path, cuda_device=cls_dev
            )
            print("✓ 个体识别模型已加载")
        else:
            self.classifier = None
            print("✓ 仅检测模式（不加载个体分类器，帧率更高，框标签为「大象」）")

        self.use_yolo = use_yolo
        self._yolo_elephant_aliases = {"elephant", "大象"}

        if use_yolo:
            try:
                from ultralytics import YOLO

                if torch.cuda.is_available():
                    dev = max(0, min(int(cuda_device), torch.cuda.device_count() - 1))
                    torch.cuda.set_device(dev)
                self.yolo_model = YOLO(yolo_weights)
                print(f"✓ YOLO 检测模型: {yolo_weights}")
            except ImportError:
                print("⚠️  未安装 ultralytics，将使用背景分割")
                self.use_yolo = False

        if not self.use_yolo:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=25, detectShadows=False
            )
            print("✓ 使用背景分割检测器")

        self.trackers = {}
        self.next_id = 0

        self.name_to_color = {
            n: _BGR_PALETTE[i % len(_BGR_PALETTE)]
            for i, n in enumerate(
                self.classifier.class_names
                if self.classifier is not None
                else ["大象"]
            )
        }

        self.recognition_interval = max(1, recognition_interval)
        self.frame_count = 0
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.ema_alpha = float(ema_alpha)
        self.min_det_conf = float(min_det_conf)
        self.freeze_recognition_when_locked = bool(freeze_recognition_when_locked)
        self.show_overlay_confidence = bool(show_overlay_confidence)
        self.yolo_imgsz = int(yolo_imgsz)
        self.infer_max_width = max(0, int(infer_max_width))
        self.detect_interval = max(1, int(detect_interval))
        self.max_lost_frames = max(3, int(max_lost_frames))
        self.clear_after_empty_detects = max(0, int(clear_after_empty_detects))
        self._empty_yolo_streak = 0
        self.camera_use_mjpeg = bool(camera_use_mjpeg)
        self.camera_width = max(0, int(camera_width))
        self.camera_height = max(0, int(camera_height))
        self._smooth_fps = 0.0
        self._fps_prev_t = 0.0

        self._yolo_device: int | str = "cpu"
        self._yolo_half = False
        self._yolo_use_predict = bool(yolo_use_predict)
        if self.use_yolo:
            if torch.cuda.is_available():
                dev = max(0, min(int(cuda_device), torch.cuda.device_count() - 1))
                self._yolo_device = dev
                self._yolo_half = True
                print(f"✓ YOLO 使用 GPU cuda:{dev} ({torch.cuda.get_device_name(dev)}) + FP16")
            else:
                self._yolo_device = "cpu"
                print("✓ YOLO 使用 CPU（未检测到 CUDA）")
        if self._yolo_use_predict:
            print("✓ YOLO 低显存模式: predict（分类器可放 CPU，省 GPU 显存）")

        # 身份已稳定后，换名需连续多帧 + EMA 上 top1-top2 间隔足够大，抑制「错象闪一下」
        self._switch_frames_loose = 8
        self._switch_margin_loose_pct = 12.0
        self._switch_frames_locked = 999
        self._switch_margin_locked_pct = 99.0
        self._lock_after_stable_frames = 8

        if self.clear_after_empty_detects > 0:
            print(
                f"✓ 零目标快清: 连续 {self.clear_after_empty_detects} 次检测无大象则移除所有框"
            )

        print("✓ 初始化完成\n")
    
    def _detect_yolo_on_bgr(self, frame_bgr: np.ndarray) -> list:
        """在已有分辨率的 BGR 图上跑 YOLO（不进行缩放）。"""
        detections: list = []
        yolo_kw = dict(
            verbose=False,
            imgsz=self.yolo_imgsz,
            device=self._yolo_device,
            half=self._yolo_half,
        )
        if self._yolo_use_predict:
            results = self.yolo_model.predict(frame_bgr, **yolo_kw)
        else:
            results = self.yolo_model.track(frame_bgr, persist=True, **yolo_kw)
        names = self.yolo_model.names
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cname = str(names[cls_id]).lower()
                if cname not in self._yolo_elephant_aliases:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = (
                    int(box.id[0]) if (not self._yolo_use_predict and box.id is not None) else None
                )
                bbox = (x1, y1, x2 - x1, y2 - y1)
                if bbox[2] > 32 and bbox[3] > 32:
                    detections.append(
                        {"bbox": bbox, "conf": conf, "track_id": track_id}
                    )
        return detections

    def detect_objects(self, frame):
        """
        检测帧中的大象；若启用 infer_max_width 则先在缩小图上检测再映射回原分辨率。
        背景分割模式始终在原分辨率上运算。
        """
        if not self.use_yolo:
            detections = []
            fg_mask = self.bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 50 and h > 50:
                        bbox = (x, y, w, h)
                        detections.append(
                            {"bbox": bbox, "conf": 1.0, "track_id": None}
                        )
            return detections

        h, w = frame.shape[:2]
        scale = 1.0
        blob = frame
        if self.infer_max_width > 0 and w > self.infer_max_width:
            scale = self.infer_max_width / float(w)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            blob = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        detections = self._detect_yolo_on_bgr(blob)
        if scale != 1.0 and detections:
            inv = 1.0 / scale
            for d in detections:
                x, y, bw, bh = d["bbox"]
                d["bbox"] = (
                    int(round(x * inv)),
                    int(round(y * inv)),
                    int(round(bw * inv)),
                    int(round(bh * inv)),
                )
        return detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IOU（交并比）"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _active_names(self) -> list[str]:
        if self.classifier is None:
            return []
        return self.classifier.active_class_names()

    def _probs_dict_to_vec(self, probs: dict) -> np.ndarray:
        names = self._active_names()
        v = np.array(
            [float(probs.get(c, 0.0)) for c in names],
            dtype=np.float64,
        )
        v = np.clip(v, 0.0, 100.0) / 100.0
        s = v.sum()
        if s < 1e-6:
            v[:] = 1.0 / max(len(v), 1)
        else:
            v /= s
        return v

    def recognize_elephant(self, frame, bbox):
        """融合主体+特征区识别；返回 (名字, 置信度, top1-top2 间隔%, probs字典)。"""
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w < 24 or h < 24:
            return None, 0.0, 0.0, None
        try:
            name, confidence, margin, probs, _ = self.classifier.predict_fused_from_bbox(
                frame, x, y, w, h, tta=False
            )
            return name, float(confidence), float(margin), probs
        except Exception as e:
            print(f"识别错误: {e}")
            return None, 0.0, 0.0, None

    def _ema_update(self, info: dict, probs: dict, raw_margin: float) -> None:
        vec = self._probs_dict_to_vec(probs)
        # 单帧很含糊时减小步长，减轻错象一帧把 EMA 拉飞
        trust = float(np.clip(raw_margin / 14.0, 0.35, 1.0))
        a = self.ema_alpha * trust
        if info.get("prob_ema") is None:
            info["prob_ema"] = vec.copy()
        else:
            pe = (1.0 - a) * info["prob_ema"] + a * vec
            s = pe.sum()
            info["prob_ema"] = pe / (s + 1e-8)

    def _hysteresis_set_display(self, info: dict) -> None:
        pe = info.get("prob_ema")
        if pe is None:
            return
        if info.get("identity_locked") and info.get("display_name"):
            dn = info["display_name"]
            info["name"] = dn
            info["hint_name"] = dn
            info["color"] = self.name_to_color.get(dn, (80, 255, 100))
            return
        names = self._active_names()
        if not names:
            return
        order = np.argsort(pe)[::-1]
        i1 = int(order[0])
        i2 = int(order[1]) if len(order) > 1 else i1
        top1p, top2p = float(pe[i1]), float(pe[i2])
        ema_margin_pct = (top1p - top2p) * 100.0
        candidate = names[i1]

        disp = info.get("display_name")
        info["display_conf"] = top1p * 100.0
        if disp is None:
            # 首名必须 EMA 上足够自信，避免第一帧误分（如全判成同一象）被立刻展示
            if (
                top1p * 100.0 < self.min_confidence
                or ema_margin_pct < self.min_margin
            ):
                return
            info["display_name"] = candidate
            info["alt_pending"] = None
            info["alt_streak"] = 0
            info["stable_frames"] = 1
            info["identity_locked"] = False
        elif candidate == disp:
            info["alt_pending"] = None
            info["alt_streak"] = 0
            info["stable_frames"] = int(info.get("stable_frames", 0)) + 1
            if info["stable_frames"] >= self._lock_after_stable_frames:
                info["identity_locked"] = True
        else:
            if info.get("identity_locked"):
                dn = info.get("display_name")
                if dn:
                    info["name"] = dn
                    info["hint_name"] = dn
                    info["color"] = self.name_to_color.get(dn, (80, 255, 100))
                return
            info["stable_frames"] = 0
            if candidate == info.get("alt_pending"):
                info["alt_streak"] = int(info.get("alt_streak", 0)) + 1
            else:
                info["alt_pending"] = candidate
                info["alt_streak"] = 1
            if (
                info["alt_streak"] >= self._switch_frames_loose
                and ema_margin_pct >= self._switch_margin_loose_pct
            ):
                info["display_name"] = candidate
                info["alt_pending"] = None
                info["alt_streak"] = 0
                info["identity_locked"] = False
                info["stable_frames"] = 0

        dn = info.get("display_name")
        if dn:
            info["name"] = dn
            info["hint_name"] = dn
            info["hint_conf"] = float(info["display_conf"])
            info["color"] = self.name_to_color.get(dn, (80, 255, 100))

    def _track_identity_strength(self, info: dict) -> float:
        return float(info.get("display_conf", 0.0)) + 0.5 * int(
            info.get("stable_frames", 0)
        )

    def _set_track_display_name(self, info: dict, name: str, prob_pct: float) -> None:
        info["display_name"] = name
        info["name"] = name
        info["display_conf"] = prob_pct
        info["hint_name"] = name
        info["hint_conf"] = prob_pct
        info["color"] = self.name_to_color.get(name, (80, 255, 100))

    def _resolve_unique_names_in_frame(self, current_frame_ids: set) -> None:
        """仅在同帧出现重名时按置信度消歧；已锁定轨迹不再被全局重分配。"""
        if not self.classify_elephants or self.classifier is None:
            return
        if len(current_frame_ids) < 2:
            return

        all_names = self._active_names()
        if len(all_names) < 2:
            return

        entries: list[tuple[int, dict]] = []
        for tid in current_frame_ids:
            info = self.trackers.get(tid)
            if info is not None:
                entries.append((tid, info))
        if len(entries) < 2:
            return

        by_name: dict[str, list[tuple[int, dict]]] = {}
        for tid, info in entries:
            dn = info.get("display_name")
            if dn:
                by_name.setdefault(str(dn), []).append((tid, info))
        duplicates = {n: tids for n, tids in by_name.items() if len(tids) > 1}
        if not duplicates:
            return

        taken: set[str] = set()
        for tid, info in entries:
            if info.get("identity_locked") and info.get("display_name"):
                taken.add(str(info["display_name"]))

        def _ema_prob(info: dict, name: str) -> float:
            pe = info.get("prob_ema")
            if pe is None or name not in all_names:
                return 0.0
            return float(pe[all_names.index(name)])

        for name, group in duplicates.items():
            ranked = sorted(
                group,
                key=lambda x: self._track_identity_strength(x[1]),
                reverse=True,
            )
            winner_tid, winner_info = ranked[0]
            if winner_info.get("identity_locked") or str(
                winner_info.get("display_name")
            ) == name:
                taken.add(name)
            for tid, info in ranked[1:]:
                if info.get("identity_locked"):
                    continue
                avail = [n for n in all_names if n not in taken]
                if not avail:
                    info["display_name"] = None
                    info["name"] = None
                    info["display_conf"] = 0.0
                    continue
                chosen = max(avail, key=lambda n: _ema_prob(info, n))
                self._set_track_display_name(
                    info, chosen, _ema_prob(info, chosen) * 100.0
                )
                taken.add(chosen)

    def _ensure_track_id(self, bbox, track_id):
        if track_id is not None:
            return int(track_id)
        best_match_id = None
        best_iou = 0.3
        for tid, tinfo in self.trackers.items():
            iou = self._calculate_iou(bbox, tinfo["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match_id = tid
        if best_match_id is not None:
            return best_match_id
        tid = self.next_id
        self.next_id += 1
        return tid

    def _remove_stale_trackers(self, current_frame_ids: set) -> None:
        """未被本帧检测结果包含的轨迹，若太久未命中则删除。"""
        stale: list = []
        for tid, tinfo in self.trackers.items():
            if tid not in current_frame_ids:
                if (
                    self.frame_count - tinfo.get("last_seen", 0)
                    > self.max_lost_frames
                ):
                    stale.append(tid)
        for tid in stale:
            del self.trackers[tid]

    def _update_tracks_for_frame(self, frame, detections):
        # 检测跳帧：未跑 YOLO。不能在此时刷新 last_seen，否则画面里已无大象时框会永远不消失
        if detections is None:
            self._remove_stale_trackers(set())
            return

        # 画面里已无大象但仍残留框时：只靠 max_lost 要等很多帧；可启用 clear_after_empty_detects 快进清
        if self.clear_after_empty_detects > 0 and len(detections) == 0:
            self._empty_yolo_streak += 1
            if self._empty_yolo_streak >= self.clear_after_empty_detects:
                self.trackers.clear()
                self._empty_yolo_streak = 0
                return
        else:
            self._empty_yolo_streak = 0

        current_frame_ids = set()
        for det in detections:
            bbox = det["bbox"]
            track_id = self._ensure_track_id(bbox, det["track_id"])
            current_frame_ids.add(track_id)

            if track_id not in self.trackers:
                self.trackers[track_id] = {
                    "bbox": bbox,
                    "name": None,
                    "confidence": 0.0,
                    "display_conf": 0.0,
                    "color": (140, 140, 140),
                    "last_seen": self.frame_count,
                    "birth_frame": self.frame_count,
                    "hint_name": None,
                    "hint_conf": 0.0,
                    "prob_ema": None,
                    "display_name": None,
                    "alt_pending": None,
                    "alt_streak": 0,
                    "stable_frames": 0,
                    "identity_locked": False,
                }

            info = self.trackers[track_id]
            info["bbox"] = bbox
            info["last_seen"] = self.frame_count
            if not self.classify_elephants and info.get("display_name") is None:
                info["display_name"] = "大象"
                info["name"] = "大象"
                info["color"] = self.name_to_color.get("大象", (80, 255, 100))
            track_age = self.frame_count - info["birth_frame"]
            unconfirmed = info.get("display_name") is None
            det_conf = float(det.get("conf", 1.0))
            locked = bool(info.get("identity_locked"))
            if self.freeze_recognition_when_locked and locked:
                should_recognize = unconfirmed
            else:
                should_recognize = unconfirmed or (
                    track_age < 22
                ) or (not locked) or (
                    self.frame_count % self.recognition_interval == 0
                )

            if should_recognize and det_conf >= self.min_det_conf and self.classify_elephants:
                out = self.recognize_elephant(frame, bbox)
                name, confidence, margin, probs = out
                if probs is not None:
                    if (
                        float(confidence) >= self.min_confidence
                        and float(margin) >= self.min_margin
                    ):
                        self._ema_update(info, probs, margin)
                        self._hysteresis_set_display(info)
                    info["confidence"] = float(confidence)

        self._resolve_unique_names_in_frame(current_frame_ids)
        self._remove_stale_trackers(current_frame_ids)

    def _draw_tracked_elephants(self, frame, show_track_dot=True, label_font_size=22):
        for track_id, info in self.trackers.items():
            x, y, w, h = info["bbox"]
            dn = info.get("display_name")
            color = info["color"]
            if dn:
                if self.show_overlay_confidence:
                    conf_show = float(info.get("display_conf", 0.0))
                    text = f"{dn} {conf_show:.0f}%"
                else:
                    text = dn
            else:
                text = "识别中..."
                color = (120, 120, 120)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            _draw_label_chinese_bgr(frame, x, y, text, color, font_size=label_font_size)
            if show_track_dot:
                cv2.circle(frame, (x + 14, y + 14), 9, color, -1)
                cv2.putText(
                    frame,
                    str(track_id),
                    (x + 7, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    2,
                )

    def process_video(self, video_path, output_path=None, show_live=True):
        """
        处理视频
        
        参数:
            video_path: 输入视频路径
            output_path: 输出视频路径
            show_live: 是否显示实时画面
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return

        self.frame_count = 0
        self.trackers.clear()
        self.next_id = 0

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = (total_frames / fps) if fps and total_frames > 0 else 0.0

        print(f"\n视频信息:")
        print(f"  文件: {video_path}")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {duration_s:.1f} 秒")
        print(
            f"  检测配置: yolo_imgsz={self.yolo_imgsz}, "
            f"infer_max_width={self.infer_max_width or '关'}, "
            f"detect_interval={self.detect_interval}\n"
        )
        
        # 创建视频写入器
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"输出: {output_path}\n")
        
        print("开始处理...")
        if show_live:
            print("提示: 按 'q' 退出, 'p' 暂停")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                self.frame_count += 1

                if (self.frame_count - 1) % self.detect_interval == 0:
                    detections = self.detect_objects(frame)
                    self._update_tracks_for_frame(frame, detections)
                else:
                    self._update_tracks_for_frame(frame, None)
                self._draw_tracked_elephants(frame)
                
                # 显示进度信息
                info_text = f"Frame: {self.frame_count}/{total_frames} | Elephants: {len(self.trackers)}"
                cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 无GUI模式下打印进度
                if not show_live and self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"进度: {progress:.1f}% ({self.frame_count}/{total_frames}) | 大象: {len(self.trackers)}")
                
                # 写入输出
                if out:
                    out.write(frame)
            
            # 显示画面
            if show_live:
                try:
                    cv2.imshow('Elephant Tracking', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\n用户中断")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        status = "暂停" if paused else "继续"
                        print(f"{status}...")
                except cv2.error:
                    # GUI不可用
                    show_live = False
                    print("检测到无GUI环境，切换到后台模式")
        
        # 清理
        cap.release()
        if out:
            out.release()
        
        if show_live:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print(f"\n处理完成!")
        print(f"  处理帧数: {self.frame_count}")
        print(f"  识别到: {len(self.trackers)} 头大象")
        if output_path:
            print(f"  输出文件: {output_path}")

    def process_video_to_file(
        self,
        video_path: str | Path,
        output_path: str | Path,
        progress_cb=None,
    ) -> dict:
        """
        离线视频识别并写出带框 MP4（无 GUI）。
        progress_cb(percent, frames_done, frames_total)
        """
        video_path = str(video_path)
        output_path = str(output_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        self.frame_count = 0
        self.trackers.clear()
        self.next_id = 0

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"无法创建输出视频: {output_path}")

        names_seen: set[str] = set()
        try:
            label_font_size = int(os.environ.get("ELEPHANT_LABEL_FONT_SIZE", "40"))
        except ValueError:
            label_font_size = 40
        label_font_size = max(16, min(96, label_font_size))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_count += 1
                if (self.frame_count - 1) % self.detect_interval == 0:
                    detections = self.detect_objects(frame)
                    self._update_tracks_for_frame(frame, detections)
                else:
                    self._update_tracks_for_frame(frame, None)
                self._draw_tracked_elephants(
                    frame, show_track_dot=False, label_font_size=label_font_size
                )
                out.write(frame)
                for _, info in self.trackers.items():
                    dn = info.get("display_name")
                    if dn and dn not in ("识别中...", "大象"):
                        names_seen.add(str(dn))
                if progress_cb and total_frames > 0 and self.frame_count % 5 == 0:
                    progress_cb(
                        100.0 * self.frame_count / total_frames,
                        self.frame_count,
                        total_frames,
                    )
        finally:
            cap.release()
            out.release()

        if progress_cb and total_frames > 0:
            progress_cb(100.0, self.frame_count, total_frames)

        duration_s = (self.frame_count / fps) if fps else 0.0
        return {
            "frames": self.frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": round(duration_s, 2),
            "elephant_names": sorted(names_seen),
            "output_path": output_path,
        }
    
    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        """降低缓冲、尽量开启 MJPEG，并可请求采集分辨率。"""
        if self.camera_use_mjpeg:
            try:
                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG"),
                )
            except Exception:
                pass
        if self.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.camera_width))
        if self.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.camera_height))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _update_fps_overlay(self) -> float:
        now = time.perf_counter()
        if self._fps_prev_t <= 0:
            self._fps_prev_t = now
            return 0.0
        dt = now - self._fps_prev_t
        self._fps_prev_t = now
        if dt > 1e-6:
            inst = 1.0 / dt
            self._smooth_fps = (
                0.9 * self._smooth_fps + 0.1 * inst
                if self._smooth_fps > 0
                else inst
            )
        return self._smooth_fps

    def _print_webcam_console_status(self) -> None:
        """无窗口模式下周期性打印跟踪结果。"""
        if not self.trackers:
            print(f"[帧 {self.frame_count}] 未检测到大象")
            return
        parts = []
        for tid, info in self.trackers.items():
            dn = info.get("display_name") or "识别中"
            parts.append(f"#{tid}:{dn}")
        print(f"[帧 {self.frame_count}] " + " | ".join(parts))

    def process_webcam(self, camera_id=0, headless: bool = False):
        """处理摄像头。headless=True 不写窗口（适合 opencv-python-headless 或服务端）。"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return

        self._configure_capture(cap)
        self.frame_count = 0
        self.trackers.clear()
        self.next_id = 0
        self._smooth_fps = 0.0
        self._fps_prev_t = 0.0
        
        print("\n摄像头已启动")
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS)
        print(
            f"采集约: {aw}x{ah} @ {afps:.1f} FPS (驱动上报) | "
            f"classify={'开' if self.classify_elephants else '关'} | "
            f"yolo_imgsz={self.yolo_imgsz}, infer_max_width="
            f"{self.infer_max_width or '关'}, detect_interval={self.detect_interval}"
        )
        gui_ok = not headless
        if headless:
            print("无窗口模式: 不进行 imshow（Ctrl+C 退出），每 30 帧打印一次识别状态")
        else:
            print("按 'q' 退出, 'p' 暂停, 's' 截图, 'r' 重置")
        
        paused = False
        screenshot_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                self.frame_count += 1

                if (self.frame_count - 1) % self.detect_interval == 0:
                    detections = self.detect_objects(frame)
                    self._update_tracks_for_frame(frame, detections)
                else:
                    self._update_tracks_for_frame(frame, None)
                self._draw_tracked_elephants(frame, show_track_dot=False)

                fps_disp = self._update_fps_overlay()
                info = f"FPS:{fps_disp:.0f} | Trk:{len(self.trackers)} | d={self.detect_interval}"
                cv2.putText(frame, info, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            key = 0
            if gui_ok:
                try:
                    cv2.imshow("Elephant Tracking - Webcam", frame)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    gui_ok = False
                    print(
                        "\n[cv2.imshow 不可用] 当前 OpenCV 为无 GUI 版本（常见于 opencv-python-headless）。\n"
                        "若要弹出摄像头窗口，请在当前环境执行：\n"
                        "  pip uninstall opencv-python-headless -y\n"
                        "  pip install opencv-python\n"
                        "然后重新运行。\n"
                        "---\n"
                        "已自动切换到无窗口模式（Ctrl+C 退出），仍会继续识别…\n"
                    )
            else:
                if self.frame_count % 30 == 0:
                    self._print_webcam_console_status()
                time.sleep(0.001)

            if key == ord('q'):
                break
            elif gui_ok and key == ord('p'):
                paused = not paused
            elif gui_ok and key == ord('s'):
                path = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(path, frame)
                print(f"截图保存: {path}")
                screenshot_count += 1
            elif gui_ok and key == ord('r'):
                self.trackers.clear()
                print("已重置跟踪")
        
        cap.release()
        if gui_ok:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

def main():
    parser = argparse.ArgumentParser(description='大象视频跟踪系统')
    parser.add_argument('--mode', type=str, default='video',
                       choices=['video', 'webcam'],
                       help='模式: video 或 webcam')
    parser.add_argument('--input', type=str,
                       help='输入视频路径')
    parser.add_argument('--output', type=str,
                       help='输出视频路径')
    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头ID')
    parser.add_argument('--no-yolo', action='store_true',
                       help='不使用YOLO（使用背景分割）')
    parser.add_argument('--model', type=str, default='best_elephant_model.pth',
                       help='识别模型路径')
    parser.add_argument('--classes', type=str, default='class_names.json',
                       help='类别文件路径')
    parser.add_argument('--yolo-weights', type=str, default='yolov8m.pt',
                       help='YOLOv8 权重（建议 m/l 提升检测准确率）')
    parser.add_argument('--min-conf', type=float, default=30.0,
                       help='参与投票的最低分类置信度(%%)，默认 30')
    parser.add_argument('--min-margin', type=float, default=4.0,
                       help='top1-top2 概率差(%%)下限，默认 4')
    parser.add_argument('--recog-interval', type=int, default=4,
                       help='已确认目标每隔多少帧再识别一次，默认 4')
    parser.add_argument('--ema-alpha', type=float, default=0.12,
                       help='分类概率 EMA 步长(0~1)，越小越平滑、换名越慢，默认 0.12')
    parser.add_argument('--min-det-conf', type=float, default=0.42,
                       help='YOLO 检测置信度低于此值时不更新分类(减少框很差时的乱闪)，默认 0.42')
    
    parser.add_argument('--no-freeze-locked', action='store_true',
                       help='身份锁定后仍周期性重识别（画面更少「锁死错名」，但更易跳名）')
    parser.add_argument('--overlay-conf', action='store_true',
                       help='在框上叠加置信度百分比（默认关闭，仅显示姓名）')
    parser.add_argument('--no-overlay-conf', action='store_true',
                       help='不叠置信度（与默认一致，保留兼容）')
    parser.add_argument(
        '--yolo-imgsz',
        type=int,
        default=640,
        help='YOLO 推理尺寸(短边 letterbox)，如 480/416 可明显加速，默认 640',
    )
    parser.add_argument(
        '--infer-max-width',
        type=int,
        default=0,
        help='>0 时先将画面缩放到不超过该宽度再检测(例 1080p 用 960)，再映射框回原图，默认 0 关闭',
    )
    parser.add_argument(
        '--detect-interval',
        type=int,
        default=1,
        help='每 N 帧跑一次检测；中间帧沿用上一框，提高画面流畅度(框略滞后)，默认 1',
    )
    parser.add_argument(
        '--max-lost-frames',
        type=int,
        default=30,
        help='连续多少帧没有再检测到该轨迹就删框；与 --snap-empty 二选一兜底，默认30',
    )
    parser.add_argument(
        '--snap-empty',
        type=int,
        default=None,
        metavar='N',
        help=(
            '连续 N 次「跑完检测且画面中零个大象框」就立刻清空全部框（拿开照片/目标离开很快生效）；'
            '不写则普通模式为0；见 --responsive-cam'
        ),
    )
    parser.add_argument(
        '--no-camera-mjpeg',
        action='store_true',
        help='不强制 MJPEG(若摄像头花屏或与驱动不兼容可加此项)',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='不显示摄像头窗口（无 highgui / SSH 时使用；需 Ctrl+C 结束）',
    )
    parser.add_argument(
        '--usb-fast-profile',
        action='store_true',
        help='高帧率 USB 摄像头推荐组合(yolov8n + imgsz480 + infer960 + detect每2帧)，需你已使用轻量权重',
    )
    parser.add_argument(
        '--max-preview-fps',
        action='store_true',
        help='进一步拉高预览FPS：更小 yolo-imgsz、缩略检测、detect≥3、分类更稀疏（框/名略滞后）',
    )
    parser.add_argument(
        '--responsive-cam',
        action='store_true',
        help='低滞后摄像头：强制每帧检测、收紧丢目标帧数，且默认启用 snap-empty=2（可用 --snap-empty 覆盖）',
    )

    parser.add_argument(
        '--video-fast',
        action='store_true',
        help='仅 --mode video 时：轻量化检测+跳帧+稀疏分类以加快离线导出',
    )
    parser.add_argument(
        '--detect-only',
        action='store_true',
        help='只检大象框、不跑个体分类（树莓派上帧率更高，标签为「大象」）',
    )
    parser.add_argument(
        '--camera-width',
        type=int,
        default=0,
        help='请求摄像头采集宽度，0=不指定（例 1280）',
    )
    parser.add_argument(
        '--camera-height',
        type=int,
        default=0,
        help='请求摄像头采集高度，0=不指定（例 720）',
    )
    parser.add_argument(
        '--pi-webcam',
        action='store_true',
        help='树莓派 webcam 推荐：720p 清晰预览 + 小图检测 + 默认仅检测框；要名字加 --with-classify',
    )
    parser.add_argument(
        '--with-classify',
        action='store_true',
        help='与 --pi-webcam 联用：仍加载个体分类（更慢）',
    )

    args = parser.parse_args()

    if args.snap_empty is None:
        args.snap_empty = 2 if (args.responsive_cam and args.mode == "webcam") else 0

    if args.usb_fast_profile:
        args.yolo_imgsz = min(args.yolo_imgsz, 480)
        if args.infer_max_width <= 0:
            args.infer_max_width = 960
        if args.detect_interval <= 1:
            args.detect_interval = 2
        if Path(args.yolo_weights).name == "yolov8m.pt":
            args.yolo_weights = "yolov8n.pt"
            print("[usb-fast-profile] 已切换到 yolov8n.pt ，可用 --yolo-weights 覆盖")

    if args.max_preview_fps:
        args.yolo_imgsz = min(args.yolo_imgsz, 384)
        if args.infer_max_width <= 0:
            args.infer_max_width = 848
        args.detect_interval = max(args.detect_interval, 3)
        if args.recog_interval < 10:
            args.recog_interval = 10
        wname = Path(args.yolo_weights).name
        if wname in ("yolov8m.pt", "yolov8s.pt", "yolov8l.pt", "yolov8x.pt"):
            args.yolo_weights = "yolov8n.pt"
            print("[max-preview-fps] 已切换到 yolov8n.pt ，可用 --yolo-weights 覆盖")

    # 离线视频导出加速（不改变 webcam 默认值）
    if args.mode == 'video' and args.video_fast:
        args.yolo_imgsz = min(args.yolo_imgsz, 480)
        if args.infer_max_width <= 0:
            args.infer_max_width = 960
        if args.detect_interval <= 1:
            args.detect_interval = 2
        if args.recog_interval < 8:
            args.recog_interval = 8
        wvf = Path(args.yolo_weights).name
        if wvf in ("yolov8m.pt", "yolov8s.pt", "yolov8l.pt", "yolov8x.pt"):
            args.yolo_weights = "yolov8n.pt"
            print("[video-fast] 已切换到 yolov8n.pt ，可用 --yolo-weights 覆盖")
        print(
            "[video-fast] imgsz≤480 infer_max960 detect≥2 recog≥8，"
            "画框仍为每帧（略滞后于检测间隔）"
        )

    # 仅 webcam：抵消 usb-fast 的跳帧，保证跟手 / 快清框
    if args.responsive_cam:
        if args.mode != 'webcam':
            print("提示: --responsive-cam 仅在与 --mode webcam 同时使用才有意义，已跳过")
        else:
            args.detect_interval = 1
            args.max_lost_frames = min(args.max_lost_frames, 10)
            print(
                "[responsive-cam] detect-interval=1, max-lost-frames≤10, "
                f"snap-empty={args.snap_empty}"
            )

    if args.pi_webcam and args.mode == "webcam":
        if Path(args.yolo_weights).name == "yolov8m.pt":
            args.yolo_weights = "yolov8n.pt"
        args.yolo_imgsz = min(args.yolo_imgsz, 320)
        if args.infer_max_width <= 0:
            args.infer_max_width = 640
        if args.detect_interval <= 1:
            args.detect_interval = 2
        if args.camera_width <= 0:
            args.camera_width = 1280
        if args.camera_height <= 0:
            args.camera_height = 720
        if not args.with_classify:
            args.detect_only = True
        else:
            args.recog_interval = max(args.recog_interval, 16)
        print(
            "[pi-webcam] 采集 1280x720、检测缩至宽640、detect-interval≥2；"
            + (
                "默认仅检测框(加 --with-classify 才识别个体)"
                if args.detect_only
                else "已启用个体分类(较慢)"
            )
        )

    tracker = YOLOElephantTracker(
        model_path=args.model,
        class_names_path=args.classes,
        use_yolo=not args.no_yolo,
        yolo_weights=args.yolo_weights,
        min_confidence=args.min_conf,
        min_margin=args.min_margin,
        recognition_interval=args.recog_interval,
        ema_alpha=args.ema_alpha,
        min_det_conf=args.min_det_conf,
        freeze_recognition_when_locked=not args.no_freeze_locked,
        show_overlay_confidence=(args.overlay_conf and not args.no_overlay_conf),
        yolo_imgsz=args.yolo_imgsz,
        infer_max_width=args.infer_max_width,
        detect_interval=args.detect_interval,
        max_lost_frames=args.max_lost_frames,
        clear_after_empty_detects=args.snap_empty,
        camera_use_mjpeg=not args.no_camera_mjpeg,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        classify_elephants=not args.detect_only,
    )
    
    if args.mode == 'video':
        if not args.input:
            print("错误: 请指定 --input <视频路径>")
            return
        
        if not Path(args.input).exists():
            print(f"错误: 文件不存在 {args.input}")
            return
        
        tracker.process_video(args.input, args.output, show_live=True)
    
    elif args.mode == 'webcam':
        tracker.process_webcam(args.camera, headless=args.headless)

if __name__ == '__main__':
    main()
