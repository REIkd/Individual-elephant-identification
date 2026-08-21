"""
大象个体识别推理：自动读取 checkpoint 中的 arch / image_size，与 train.py 一致。
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from elephant_net import build_model


def normalize_image_path(raw: str) -> Path:
    """去掉引号并解析路径（Windows 终端/smart quotes 兼容）。"""
    s = raw.strip().strip('"').strip("'")
    for ch in ("\u201c", "\u201d", "\u2018", "\u2019"):
        s = s.strip(ch)
    return Path(s)


def imread_bgr(path: str | Path) -> np.ndarray | None:
    """OpenCV 在 Windows 上无法直接读中文路径，用 imdecode 绕过。"""
    p = Path(path)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_allowed_elephants(raw: str | None) -> list[str] | None:
    """
    解析候选象名单。None/空/「全部」= 不限制。
    支持逗号分隔、JSON 数组文件、每行一个名字的文本文件。
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"*", "all", "全部", "none"}:
        return None
    p = Path(s)
    if p.is_file():
        text = p.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            names = json.loads(text)
            if not isinstance(names, list):
                raise ValueError(f"候选象文件应为 JSON 数组: {p}")
            return [str(x).strip() for x in names if str(x).strip()]
        return [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    return [x.strip() for x in s.replace("，", ",").split(",") if x.strip()]


class ElephantClassifier:
    def __init__(
        self,
        model_path: str = "best_elephant_model.pth",
        class_names_path: str = "class_names.json",
        cuda_device: int = 0,
        allowed_elephants: list[str] | None = None,
    ):
        if int(cuda_device) < 0 or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            dev = max(0, min(int(cuda_device), torch.cuda.device_count() - 1))
            self.device = torch.device(f"cuda:{dev}")
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            self.class_names = checkpoint["class_names"]
        else:
            with open(class_names_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)

        arch = "resnet50"
        image_size = 224
        if isinstance(checkpoint, dict):
            arch = checkpoint.get("arch", arch)
            image_size = int(checkpoint.get("image_size", image_size))

        self.arch = arch
        self.image_size = image_size
        self.model = build_model(arch, len(self.class_names), pretrained=False)
        state = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.to(self.device)

        va = checkpoint.get("val_acc", 0.0) if isinstance(checkpoint, dict) else 0.0
        print(f"模型已加载 arch={arch} 输入={image_size} (checkpoint val_acc={va:.2f}%)")
        if self.device.type == "cuda":
            print(f"分类器设备: {self.device} ({torch.cuda.get_device_name(self.device.index)})")
        else:
            print("分类器设备: cpu（未检测到 CUDA，请安装 GPU 版 PyTorch）")
        print("可识别: " + ", ".join(self.class_names))
        self._allowed: list[str] | None = None
        if allowed_elephants:
            self.set_allowed_elephants(allowed_elephants)

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # CPU：限制线程避免与 OpenCV/YOLO 争抢（可调 ELEPHANT_TORCH_THREADS）
        if self.device.type == "cpu":
            try:
                default_nt = max(1, min(8, (os.cpu_count() or 4)))
                _nt = int(os.environ.get("ELEPHANT_TORCH_THREADS", str(default_nt)))
                torch.set_num_threads(max(1, _nt))
                torch.set_num_interop_threads(1)
            except Exception:
                pass

    def set_allowed_elephants(self, names: list[str] | None) -> None:
        """限制只在这几头象里选（园区当日出场象），显著提高准确率。"""
        if not names:
            self._allowed = None
            return
        valid: list[str] = []
        name_set = set(self.class_names)
        for n in names:
            t = str(n).strip()
            if t in name_set and t not in valid:
                valid.append(t)
            elif t and t not in name_set:
                print(f"警告: 候选象 '{t}' 不在模型 {len(self.class_names)} 类中，已忽略")
        if not valid:
            self._allowed = None
            return
        self._allowed = valid
        print("候选象(仅此范围内识别): " + ", ".join(self._allowed))

    def get_allowed_elephants(self) -> list[str] | None:
        return list(self._allowed) if self._allowed else None

    def active_class_names(self) -> list[str]:
        return self._allowed if self._allowed else self.class_names

    def _restrict_probs(self, probs: dict[str, float]) -> dict[str, float]:
        if not self._allowed:
            return probs
        active = self._allowed
        masked = {n: float(probs.get(n, 0.0)) for n in active}
        s = sum(masked.values())
        if s < 1e-8:
            u = 100.0 / len(active)
            return {n: u for n in active}
        return {n: v / s * 100.0 for n, v in masked.items()}

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        return self._predict_pil(image)

    def predict_from_frame(self, frame: np.ndarray):
        """兼容旧接口：整帧 BGR。"""
        return self.predict_from_bgr(frame)[:2]

    def _predict_pil(self, image: Image.Image, tta: bool = False):
        if not tta:
            return self._predict_pil_once(image)
        _, _, p0 = self._predict_pil_once(image)
        _, _, p1 = self._predict_pil_once(image.transpose(Image.FLIP_LEFT_RIGHT))
        merged = {k: (p0[k] + p1[k]) / 2.0 for k in p0}
        merged = self._restrict_probs(merged)
        best = max(merged, key=merged.get)
        return best, merged[best], merged

    def _predict_pil_once(self, image: Image.Image):
        tensor = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            probs = torch.softmax(logits.float(), dim=1)
            confidence, predicted = torch.max(probs, 1)

        idx = predicted.item()
        elephant_name = self.class_names[idx]
        confidence_score = confidence.item() * 100
        all_probs = {
            self.class_names[i]: probs[0, i].item() * 100
            for i in range(len(self.class_names))
        }
        all_probs = self._restrict_probs(all_probs)
        elephant_name = max(all_probs, key=all_probs.get)
        confidence_score = all_probs[elephant_name]
        return elephant_name, confidence_score, all_probs

    def _probs_from_bgr(self, bgr: np.ndarray, tta: bool = False) -> dict[str, float]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _, _, probs = self._predict_pil(Image.fromarray(rgb), tta=tta)
        return probs

    def merge_weighted_probs(
        self,
        parts: list[tuple[dict[str, float], float]],
    ) -> tuple[str, float, float, dict[str, float]]:
        class_names = self.active_class_names()
        merged = {n: 0.0 for n in class_names}
        wsum = 0.0
        for probs, w in parts:
            if w <= 0:
                continue
            restricted = self._restrict_probs(probs)
            for n in class_names:
                merged[n] += float(restricted.get(n, 0.0)) * w
            wsum += w
        if wsum < 1e-6:
            return class_names[0], 0.0, 0.0, merged
        merged = {k: v / wsum for k, v in merged.items()}
        best = max(merged, key=merged.get)
        vals = sorted(merged.values(), reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else vals[0]
        return best, merged[best], margin, merged

    @staticmethod
    def _merge_weighted_probs(
        class_names: list[str],
        parts: list[tuple[dict[str, float], float]],
    ) -> tuple[str, float, float, dict[str, float]]:
        """兼容旧调用；新代码请用 instance.merge_weighted_probs。"""
        merged = {n: 0.0 for n in class_names}
        wsum = 0.0
        for probs, w in parts:
            if w <= 0:
                continue
            for n in class_names:
                merged[n] += float(probs.get(n, 0.0)) * w
            wsum += w
        if wsum < 1e-6:
            return class_names[0], 0.0, 0.0, merged
        merged = {k: v / wsum for k, v in merged.items()}
        best = max(merged, key=merged.get)
        vals = sorted(merged.values(), reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else vals[0]
        return best, merged[best], margin, merged

    def predict_fused_from_bbox(
        self,
        frame_bgr: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        tta: bool = False,
    ) -> tuple[str, float, float, dict[str, float], str]:
        """实地/YOLO：主体 + 头/中部近似特征区，加权融合输出象名。"""
        parts: list[tuple[dict[str, float], float]] = []
        body = _pad_roi_bgr(frame_bgr, x, y, w, h, pad_ratio=0.12)
        if body is None:
            return "识别中...", 0.0, 0.0, {}, "框过小"
        body = _upsample_roi_if_small(body, min_side=384)

        bh, bw = body.shape[:2]
        parts: list[tuple[dict[str, float], float]] = []
        # 框较大时才做头/中部伪特征融合；远距离小框只用主体，避免噪声区拉错象
        if min(bh, bw) >= 120:
            parts.append((self._probs_from_bgr(body, tta=tta), 0.45))
            head = body[0 : max(1, int(bh * 0.52)), :]
            parts.append((self._probs_from_bgr(head, tta=tta), 0.30))
            y0, y1 = int(bh * 0.18), int(bh * 0.72)
            x0, x1 = int(bw * 0.20), int(bw * 0.80)
            mid = body[y0:y1, x0:x1]
            if mid.size > 0:
                parts.append((self._probs_from_bgr(mid, tta=tta), 0.25))
        else:
            parts.append((self._probs_from_bgr(body, tta=tta), 1.0))

        name, conf, margin, probs = self.merge_weighted_probs(parts)
        tag = f"融合({len(parts)}区)" if len(parts) > 1 else "远距离主体"
        if self._allowed:
            tag += f" 候选{len(self._allowed)}头"
        return name, conf, margin, probs, tag

    def predict_fused_from_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        xml_path: Path | None = None,
        yolo_bbox: tuple[int, int, int, int] | None = None,
        tta: bool = False,
    ) -> tuple[str, float, float, dict[str, float], str]:
        """有 XML：主体+全部特征框融合；否则 YOLO 主体+伪特征区。"""
        if xml_path is not None and xml_path.is_file():
            regions = _parse_fusion_regions_xml(xml_path, self.class_names)
            if regions:
                parts: list[tuple[dict[str, float], float]] = []
                fh, fw = frame_bgr.shape[:2]
                for kind, box, weight in regions:
                    pad = 0.12 if kind == "body" else 0.18
                    xmin, ymin, xmax, ymax = _pad_box(box, fw, fh, pad)
                    crop = frame_bgr[ymin:ymax, xmin:xmax]
                    if crop.size == 0:
                        continue
                    parts.append((self._probs_from_bgr(crop, tta=tta), weight))
                if parts:
                    name, conf, margin, probs = self.merge_weighted_probs(parts)
                    nb = sum(1 for k, _, _ in regions if k == "body")
                    nf = len(regions) - nb
                    return name, conf, margin, probs, f"融合(XML 主体{nb}+特征{nf})"

        if yolo_bbox is not None:
            x, y, bw, bh = yolo_bbox
            return self.predict_fused_from_bbox(frame_bgr, x, y, bw, bh, tta=tta)

        return "识别中...", 0.0, 0.0, {}, "无有效区域"

    def predict_from_bgr(self, bgr: np.ndarray, *, fuse: bool = True, tta: bool = False):
        """ROI 识别；fuse=True 时对 ROI 做多区融合（与工业化训练一致）。"""
        if fuse and bgr is not None and bgr.size > 0:
            h, w = bgr.shape[:2]
            name, conf, _, probs, _ = self.predict_fused_from_bbox(
                bgr, 0, 0, w, h, tta=tta
            )
            return name, conf, probs
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._predict_pil(Image.fromarray(rgb), tta=tta)


def elephant_name_from_folder(folder_name: str) -> str:
    name = folder_name.strip()
    if "（" in name:
        name = name.split("（", 1)[0].strip()
    if "(" in name:
        name = name.split("(", 1)[0].strip()
    return name


def _parse_fusion_regions_xml(
    xml_path: Path, class_names: list[str]
) -> list[tuple[str, tuple[int, int, int, int], float]]:
    """
    解析 XML 中主体框 + 特征框，返回 [(kind, (xmin,ymin,xmax,ymax), weight), ...]
    kind: body | feature
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    name_set = set(class_names)
    bodies: list[tuple[int, int, int, int]] = []
    features: list[tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        tag_el = obj.find("name")
        tag = (tag_el.text or "").strip() if tag_el is not None else ""
        bnd = obj.find("bndbox")
        if not tag or bnd is None:
            continue
        try:
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))
        except (TypeError, ValueError, AttributeError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        box = (xmin, ymin, xmax, ymax)
        if tag in name_set or tag == "大象":
            bodies.append(box)
        else:
            prefix = tag.split("-", 1)[0].strip()
            if prefix in name_set:
                features.append(box)
    if not bodies and not features:
        return []
    if not bodies and features:
        w = 1.0 / len(features)
        return [("feature", b, w) for b in features]
    if len(bodies) > 1:
        bodies = [max(bodies, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]
    out: list[tuple[str, tuple[int, int, int, int], float]] = [
        ("body", bodies[0], 0.40)
    ]
    if features:
        fw = 0.60 / len(features)
        out.extend(("feature", b, fw) for b in features)
    else:
        out[0] = ("body", bodies[0], 1.0)
    return out


def _parse_voc_boxes(xml_path: Path) -> list[tuple[int, int, int, int]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))
        except (TypeError, ValueError, AttributeError):
            continue
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def _pad_box(
    box: tuple[int, int, int, int], w: int, h: int, pad_ratio: float = 0.12
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    bw, bh = xmax - xmin, ymax - ymin
    pad_w, pad_h = int(bw * pad_ratio), int(bh * pad_ratio)
    return (
        max(0, xmin - pad_w),
        max(0, ymin - pad_h),
        min(w, xmax + pad_w),
        min(h, ymax + pad_h),
    )


def _load_crop_rgb(img_path: Path, pad_ratio: float = 0.12) -> Image.Image | None:
    image = Image.open(img_path).convert("RGB")
    xml_path = img_path.with_suffix(".xml")
    if xml_path.is_file():
        boxes = _parse_voc_boxes(xml_path)
        if boxes:
            w, h = image.size
            xmin, ymin, xmax, ymax = _pad_box(boxes[0], w, h, pad_ratio)
            image = image.crop((xmin, ymin, xmax, ymax))
            return image
    return None


def _pad_roi_bgr(
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


def _upsample_roi_if_small(roi: np.ndarray, min_side: int = 384) -> np.ndarray:
    """远距离小框：放大 ROI 再送分类器，减轻「像素太少」导致的误识别。"""
    bh, bw = roi.shape[:2]
    m = max(bh, bw)
    if m >= min_side:
        return roi
    scale = min_side / float(m)
    nw = max(1, int(round(bw * scale)))
    nh = max(1, int(round(bh * scale)))
    return cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_CUBIC)


def _yolo_elephant_roi_bgr(
    frame_bgr: np.ndarray,
    yolo_weights: str = "yolov8n.pt",
    cuda_device: int = -1,
    pad_ratio: float = 0.12,
) -> tuple[np.ndarray | None, str]:
    """与线上一致：YOLO 检大象框再裁剪 ROI。"""
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
    roi = _pad_roi_bgr(frame_bgr, x, y, w, h, pad_ratio=pad_ratio)
    if roi is None:
        return None, "检测框过小"
    return roi, f"YOLO ROI (conf={conf:.2f})"


def _yolo_elephant_bbox(
    frame_bgr: np.ndarray,
    yolo_weights: str = "yolov8n.pt",
    cuda_device: int = -1,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """返回 (x, y, w, h) 与说明。"""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "未安装 ultralytics"

    device = (
        max(0, min(int(cuda_device), torch.cuda.device_count() - 1))
        if int(cuda_device) >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    model = YOLO(yolo_weights)
    results = model.predict(
        frame_bgr, verbose=False, imgsz=640, device=device, half=(device != "cpu")
    )
    names = model.names
    best = None
    best_area = 0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if str(names[cls_id]).lower() not in {"elephant", "大象"}:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2 - x1, y2 - y1, float(box.conf[0]))
    if best is None:
        return None, "YOLO 未检测到大象"
    x, y, w, h, conf = best
    return (x, y, w, h), f"YOLO conf={conf:.2f}"


def fused_predict_path(
    classifier: ElephantClassifier,
    image_path: str | Path,
    *,
    full_image: bool = False,
    use_yolo: bool = True,
    yolo_weights: str = "yolov8n.pt",
    cuda_device: int = -1,
    tta: bool = True,
) -> tuple[str, float, float, dict[str, float], str]:
    """单张图：主体+特征融合，返回 (象名, 置信度%, 间隔%, probs, 说明)。"""
    path = Path(image_path).resolve()
    frame = imread_bgr(path)
    if frame is None:
        raise ValueError(f"无法读取: {path}")

    if full_image:
        h, w = frame.shape[:2]
        return classifier.predict_fused_from_bbox(frame, 0, 0, w, h, tta=tta)

    xml_path = path.with_suffix(".xml")
    if xml_path.is_file():
        return classifier.predict_fused_from_frame(
            frame, xml_path=xml_path, tta=tta
        )

    if use_yolo:
        bbox, ydetail = _yolo_elephant_bbox(
            frame, yolo_weights=yolo_weights, cuda_device=cuda_device
        )
        if bbox is not None:
            x, y, w, h = bbox
            name, conf, margin, probs, detail = classifier.predict_fused_from_bbox(
                frame, x, y, w, h, tta=tta
            )
            return name, conf, margin, probs, f"{detail} | {ydetail}"

    h, w = frame.shape[:2]
    return classifier.predict_fused_from_bbox(frame, 0, 0, w, h, tta=tta)


def prepare_classify_input(
    image_path: str | Path,
    *,
    full_image: bool = False,
    use_yolo: bool = True,
    yolo_weights: str = "yolov8n.pt",
    cuda_device: int = -1,
) -> tuple[Image.Image, str]:
    """
    准备分类器输入。训练/评测用 VOC 象体裁剪；新照片默认 YOLO 检框再识别（与 Pi+云端一致）。
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片: {path}")

    if full_image:
        return Image.open(path).convert("RGB"), "整图（不推荐，与训练分布不一致）"

    voc = _load_crop_rgb(path, pad_ratio=0.12)
    if voc is not None:
        return voc, "VOC 标注框裁剪（与训练一致）"

    if use_yolo:
        frame = imread_bgr(path)
        if frame is None:
            raise ValueError(f"无法读取图片（路径或格式有误）: {path}")
        roi, detail = _yolo_elephant_roi_bgr(
            frame, yolo_weights=yolo_weights, cuda_device=cuda_device
        )
        if roi is not None:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb), detail

    img = Image.open(path).convert("RGB")
    return img, "整图（无 XML 且未检出大象，结果可能不准）"


def batch_eval_dataset(
    dataset_dir: str | Path = "dataset",
    model_path: str = "best_elephant_model.pth",
    class_names_path: str = "class_names.json",
    val_only: bool = True,
    val_split: float = 0.2,
    seed: int = 42,
    max_per_class: int | None = None,
    cuda_device: int = 0,
) -> dict:
    """对 21 类大象做批量识别评测（优先 VOC 框裁剪，与 train.py 一致）。"""
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    samples: list[tuple[Path, str]] = []
    for folder in sorted(dataset_path.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        for img_path in folder.iterdir():
            if img_path.suffix in exts:
                samples.append((img_path, folder.name))

    if not samples:
        raise RuntimeError(f"{dataset_path} 下没有图片")

    if val_only:
        paths = [str(p) for p, _ in samples]
        labels = [name for _, name in samples]
        _, val_paths, _, val_labels = train_test_split(
            paths, labels, test_size=val_split, random_state=seed, stratify=labels
        )
        samples = [(Path(p), lbl) for p, lbl in zip(val_paths, val_labels)]
        print(f"评测集: 验证集 {len(samples)} 张 (val_split={val_split}, seed={seed})")
    else:
        print(f"评测集: 全部 {len(samples)} 张")

    classifier = ElephantClassifier(model_path, class_names_path, cuda_device=cuda_device)

    by_class: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for img_path, true_name in samples:
        by_class[true_name].append((img_path, true_name))

    stats = {
        "total": 0,
        "correct": 0,
        "per_class": defaultdict(lambda: {"total": 0, "correct": 0, "errors": []}),
    }

    class_names = sorted(by_class.keys())
    print(f"类别数: {len(class_names)}")
    if max_per_class:
        print(f"每类最多评测 {max_per_class} 张")

    for cls_idx, true_name in enumerate(class_names):
        items = sorted(by_class[true_name], key=lambda x: str(x[0]))
        if max_per_class and len(items) > max_per_class:
            rng = __import__("random").Random(seed + cls_idx * 9973)
            rng.shuffle(items)
            items = items[:max_per_class]

        for img_path, _ in tqdm(items, desc=true_name, leave=False):
            try:
                frame = imread_bgr(img_path)
                if frame is None:
                    raise ValueError(f"无法读取: {img_path}")
                true_elephant = elephant_name_from_folder(true_name)
                xml_path = img_path.with_suffix(".xml")
                if xml_path.is_file():
                    pred_name, conf, _, _, _ = classifier.predict_fused_from_frame(
                        frame, xml_path=xml_path, tta=False
                    )
                else:
                    h, w = frame.shape[:2]
                    pred_name, conf, _, _, _ = classifier.predict_fused_from_bbox(
                        frame, 0, 0, w, h, tta=False
                    )
            except Exception as e:
                stats["per_class"][true_name]["errors"].append(
                    {"file": str(img_path), "error": str(e)}
                )
                continue

            stats["total"] += 1
            stats["per_class"][true_name]["total"] += 1
            ok = pred_name == true_elephant
            if ok:
                stats["correct"] += 1
                stats["per_class"][true_name]["correct"] += 1
            else:
                stats["per_class"][true_name]["errors"].append(
                    {
                        "file": img_path.name,
                        "predicted": pred_name,
                        "confidence": round(conf, 2),
                    }
                )

    overall = stats["correct"] / stats["total"] * 100 if stats["total"] else 0.0
    print("\n" + "=" * 58)
    print("大象识别评测结果 (主体+特征融合, predict.py)")
    print("=" * 58)
    print(f"总体准确率: {overall:.2f}% ({stats['correct']}/{stats['total']})")
    print("\n各类别:")
    print("-" * 58)
    for name in sorted(stats["per_class"].keys()):
        cs = stats["per_class"][name]
        if cs["total"] == 0:
            continue
        acc = cs["correct"] / cs["total"] * 100
        flag = "OK" if acc >= 95 else ("~" if acc >= 80 else "!")
        print(f"  [{flag}] {name:6s}  {acc:6.2f}%  ({cs['correct']}/{cs['total']})")
        for err in cs["errors"][:3]:
            if "predicted" in err:
                print(
                    f"        误: {err['file']} -> {err['predicted']} "
                    f"({err['confidence']:.1f}%)"
                )

    out = {
        "overall_accuracy_pct": round(overall, 4),
        "total": stats["total"],
        "correct": stats["correct"],
        "val_only": val_only,
        "max_per_class": max_per_class,
        "per_class": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy_pct": round(v["correct"] / v["total"] * 100, 4)
                if v["total"]
                else 0,
                "errors": v["errors"][:10],
            }
            for k, v in stats["per_class"].items()
        },
    }
    out_path = Path("predict_eval_21.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {out_path.resolve()}")
    return out


def predict_image(
    model_path,
    class_names_path,
    image_path,
    *,
    full_image: bool = False,
    use_yolo: bool = True,
    yolo_weights: str = "yolov8n.pt",
    cuda_device: int = -1,
    tta: bool = True,
    fuse: bool = True,
    allowed_elephants: list[str] | None = None,
):
    clf = ElephantClassifier(
        model_path,
        class_names_path,
        cuda_device=cuda_device,
        allowed_elephants=allowed_elephants,
    )
    print(f"\n识别: {image_path}")
    if fuse:
        name, conf, margin, all_probs, how = fused_predict_path(
            clf,
            image_path,
            full_image=full_image,
            use_yolo=use_yolo,
            yolo_weights=yolo_weights,
            cuda_device=cuda_device,
            tta=tta,
        )
        print(f"输入方式: {how}")
    else:
        image, how = prepare_classify_input(
            image_path,
            full_image=full_image,
            use_yolo=use_yolo,
            yolo_weights=yolo_weights,
            cuda_device=cuda_device,
        )
        print(f"输入方式: {how}  尺寸: {image.size[0]}x{image.size[1]}")
        name, conf, all_probs = clf._predict_pil(image, tta=tta)
        sorted_p = sorted(all_probs.values(), reverse=True)
        margin = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)
    print(f"结果: {name}  置信度: {conf:.2f}%  top1-top2间隔: {margin:.2f}%")
    if conf < 42 or margin < 12:
        print("提示: 置信度/间隔偏低，建议换角度、拉近大象或确保框住整头象")
    for n, p in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(p / 5)
        print(f"  {n:8s} {p:6.2f}% {bar}")


def predict_camera(model_path, class_names_path, camera_id=0):
    clf = ElephantClassifier(model_path, class_names_path)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    print("按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        name, conf, _ = clf.predict_from_bgr(frame)
        color = (0, 255, 0) if conf > 80 else (0, 255, 255) if conf > 60 else (0, 0, 255)
        cv2.rectangle(frame, (10, 10), (520, 72), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{name} ({conf:.1f}%)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )
        cv2.imshow("Elephant Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["image", "camera", "batch"],
        default="image",
        help="image=单张 | camera=摄像头 | batch=21类数据集评测",
    )
    parser.add_argument("--image", type=str)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="best_elephant_model.pth")
    parser.add_argument("--classes", type=str, default="class_names.json")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="batch 模式评测全部图片（默认只用与 train 相同的 20%% 验证集）",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="batch 模式每类最多评测张数（加速抽样）",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA 设备编号，-1 为 CPU")
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="整图识别（不推荐；训练用的是象体裁剪，整图常会错）",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="无 VOC 标注时不使用 YOLO 检框",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolov8n.pt",
        help="单张测试时 YOLO 权重（新照片默认先检大象再识别）",
    )
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="单张模式只用单框识别，不做主体+特征融合",
    )
    parser.add_argument(
        "--allowed",
        type=str,
        default="",
        help='候选象，逗号分隔，如 "印东,凯恩"；空=全部类别',
    )
    args = parser.parse_args()
    allowed = parse_allowed_elephants(args.allowed or None)
    if args.mode == "image":
        if not args.image:
            print("请指定 --image")
            return
        predict_image(
            args.model,
            args.classes,
            str(normalize_image_path(args.image)),
            full_image=args.full_image,
            use_yolo=not args.no_yolo,
            yolo_weights=args.yolo_weights,
            cuda_device=args.gpu,
            fuse=not args.no_fuse,
            allowed_elephants=allowed,
        )
    elif args.mode == "batch":
        batch_eval_dataset(
            dataset_dir=args.dataset,
            model_path=args.model,
            class_names_path=args.classes,
            val_only=not args.all_data,
            max_per_class=args.max_per_class,
            cuda_device=args.gpu,
        )
    else:
        predict_camera(args.model, args.classes, args.camera)


if __name__ == "__main__":
    main()
