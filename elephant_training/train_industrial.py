"""
工业化大象个体识别训练：主体框 + 特征部位框联合训练，提升实地/YOLO 裁剪泛化。

数据约定（与当前 dataset 一致）:
  dataset/印东（身体特征框）/印东-350.jpg + .xml
  - 第 1 个框: 象名主体（如 印东）
  - 后续框: 特征（如 印东-鼻子、印东-左前腿）

训练: 主体 + 特征裁剪都参与（同一大象同一类别）
验证: 仅主体框（贴近 Pi/云端 YOLO 检整象后再识别）

用法:
  python train_industrial.py
  python train_industrial.py --epochs-head 10 --epochs-ft 30
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from elephant_net import build_model, set_backbone_trainable
from paths import CLASS_NAMES_PATH, DATASET_DIR, MODEL_PATH, ensure_torch_home
from voc_name_aliases import is_body_tag, is_feature_tag

ensure_torch_home()

ARCH = os.environ.get("ELEPHANT_ARCH", "efficientnet_v2_s")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed()


def elephant_name_from_folder(folder_name: str) -> str:
    """印东（身体特征框） -> 印东"""
    name = folder_name.strip()
    if "（" in name:
        name = name.split("（", 1)[0].strip()
    if "(" in name:
        name = name.split("(", 1)[0].strip()
    return name


def _parse_box(obj: ET.Element) -> tuple[int, int, int, int] | None:
    bnd = obj.find("bndbox")
    if bnd is None:
        return None
    try:
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
    except (TypeError, ValueError, AttributeError):
        return None
    if xmax > xmin and ymax > ymin:
        return xmin, ymin, xmax, ymax
    return None


def parse_voc_crops(
    xml_path: Path, elephant_name: str
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    """返回 (主体框, [特征框...])。"""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None, []
    body: tuple[int, int, int, int] | None = None
    features: list[tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        tag = (name_el.text or "").strip() if name_el is not None else ""
        box = _parse_box(obj)
        if box is None:
            continue
        if is_body_tag(tag, elephant_name):
            body = box
        elif is_feature_tag(tag, elephant_name):
            features.append(box)
    if body is None and features:
        # 兜底：最大框当主体
        body = max(features, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return body, features


def pad_box(
    box: tuple[int, int, int, int], w: int, h: int, pad_ratio: float
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


@dataclass(frozen=True)
class CropSample:
    img_path: Path
    label: int
    box: tuple[int, int, int, int]
    kind: str  # body | feature


@dataclass
class ImageRecord:
    img_path: Path
    label: int
    elephant_name: str
    body: tuple[int, int, int, int] | None
    features: list[tuple[int, int, int, int]]


def collect_image_records() -> tuple[list[ImageRecord], list[str]]:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"缺少数据目录: {DATASET_DIR}")

    folders = sorted(
        [d for d in DATASET_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    class_names = [elephant_name_from_folder(d.name) for d in folders]
    if len(set(class_names)) != len(class_names):
        raise RuntimeError("文件夹解析出的象名有重复，请检查 dataset 目录命名")

    class_to_idx = {n: i for i, n in enumerate(class_names)}
    records: list[ImageRecord] = []

    for folder in folders:
        elephant = elephant_name_from_folder(folder.name)
        label = class_to_idx[elephant]
        for img_path in folder.iterdir():
            if img_path.suffix not in IMAGE_EXT:
                continue
            xml_path = img_path.with_suffix(".xml")
            body, features = (
                parse_voc_crops(xml_path, elephant) if xml_path.is_file() else (None, [])
            )
            records.append(
                ImageRecord(
                    img_path=img_path,
                    label=label,
                    elephant_name=elephant,
                    body=body,
                    features=features,
                )
            )

    if not records:
        raise RuntimeError("未发现任何图片")

    with open("class_names.json", "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, ensure_ascii=False, indent=2)

    return records, class_names


def expand_crops(
    records: list[ImageRecord],
    *,
    use_body: bool = True,
    use_features: bool = True,
) -> list[CropSample]:
    out: list[CropSample] = []
    for rec in records:
        if use_body and rec.body is not None:
            out.append(CropSample(rec.img_path, rec.label, rec.body, "body"))
        if use_features:
            for fb in rec.features:
                out.append(CropSample(rec.img_path, rec.label, fb, "feature"))
    return out


class IndustrialCropDataset(Dataset):
    def __init__(
        self,
        crops: list[CropSample],
        transform,
        body_pad: float = 0.12,
        feature_pad: float = 0.18,
    ):
        self.crops = crops
        self.transform = transform
        self.body_pad = body_pad
        self.feature_pad = feature_pad

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int):
        item = self.crops[idx]
        image = Image.open(item.img_path).convert("RGB")
        w, h = image.size
        pad = self.body_pad if item.kind == "body" else self.feature_pad
        xmin, ymin, xmax, ymax = pad_box(item.box, w, h, pad)
        image = image.crop((xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)
        return image, item.label


@dataclass
class TrainConfig:
    image_size: int = 384 if ARCH == "efficientnet_v2_s" else 224
    batch_size: int = int(os.environ.get("ELEPHANT_BATCH", "16"))
    num_workers: int = min(4, os.cpu_count() or 1)
    num_epochs_head: int = 10
    num_epochs_finetune: int = 35
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.02
    val_split: float = 0.2
    label_smoothing: float = 0.1
    body_pad: float = 0.12
    feature_pad: float = 0.18
    model_save_path: str = str(MODEL_PATH)
    backup_old_path: str = str(MODEL_PATH.with_name("best_elephant_model_body_only_backup.pth"))
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(image_size: int):
    sz = image_size
    train_tf = transforms.Compose(
        [
            transforms.Resize(int(sz * 1.15)),
            transforms.RandomResizedCrop(sz, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.35, 0.35, 0.25, 0.08),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(sz * 1.1)),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return train_tf, val_tf


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        pred = logits.argmax(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        pbar.set_postfix(loss=loss_sum / len(loader), acc=100.0 * correct / total)
    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Val"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.item()
        pred = logits.argmax(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1)


def _optimizer_for_phase(model, arch: str, finetune_full: bool, cfg: TrainConfig):
    arch_l = arch.lower().strip()
    if not finetune_full:
        params = (
            list(model.fc.parameters())
            if arch_l == "resnet50"
            else list(model.classifier.parameters())
        )
        return optim.AdamW(params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    if arch_l == "resnet50":
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc")]
        head = [p for n, p in model.named_parameters() if n.startswith("fc")]
    else:
        backbone = [p for n, p in model.named_parameters() if n.startswith("features")]
        head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    return optim.AdamW(
        [
            {"params": backbone, "lr": cfg.lr_backbone},
            {"params": head, "lr": cfg.lr_head * 0.25},
        ],
        weight_decay=cfg.weight_decay,
    )


def save_checkpoint(model, class_names, val_acc, cfg: TrainConfig, arch: str, tag: str):
    path = cfg.model_save_path
    torch.save(
        {
            "arch": arch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "class_names": class_names,
            "image_size": cfg.image_size,
            "train_mode": tag,
        },
        path,
    )


def plot_curves(tl, vl, ta, va, out: str = "training_curves_industrial.png"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tl, label="train")
    plt.plot(vl, label="val(body)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(ta, label="train")
    plt.plot(va, label="val(body)")
    plt.xlabel("epoch")
    plt.ylabel("acc %")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"曲线已保存: {out}")


def print_dataset_stats(records: list[ImageRecord], class_names: list[str]) -> None:
    n_body = sum(1 for r in records if r.body is not None)
    n_feat = sum(len(r.features) for r in records)
    print(f"类别数: {len(class_names)} | 图片: {len(records)}")
    print(f"有主体框: {n_body} | 特征框总数: {n_feat}")
    print("象名: " + ", ".join(class_names))


def main():
    parser = argparse.ArgumentParser(description="工业化训练：主体+特征联合")
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-ft", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--output",
        default="best_elephant_model.pth",
        help="输出权重路径",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份旧 best_elephant_model.pth",
    )
    args = parser.parse_args()

    set_seed(42)
    cfg = TrainConfig(
        num_epochs_head=args.epochs_head,
        num_epochs_finetune=args.epochs_ft,
        model_save_path=args.output,
    )
    if args.batch_size > 0:
        cfg.batch_size = args.batch_size

    print("=" * 60)
    print("工业化大象识别训练（主体 + 特征部位联合）")
    print("=" * 60)
    print(f"设备: {cfg.device} | 骨干: {ARCH} | 输入: {cfg.image_size}")

    records, class_names = collect_image_records()
    print_dataset_stats(records, class_names)

    labels = [r.label for r in records]
    train_rec, val_rec, _, _ = train_test_split(
        records, labels, test_size=cfg.val_split, random_state=42, stratify=labels
    )
    train_crops = expand_crops(train_rec, use_body=True, use_features=True)
    val_crops = expand_crops(val_rec, use_body=True, use_features=False)
    print(
        f"训练裁剪样本: {len(train_crops)} (主体+特征) | "
        f"验证裁剪样本: {len(val_crops)} (仅主体)"
    )

    train_tf, val_tf = get_transforms(cfg.image_size)
    train_loader = DataLoader(
        IndustrialCropDataset(train_crops, train_tf, cfg.body_pad, cfg.feature_pad),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        IndustrialCropDataset(val_crops, val_tf, cfg.body_pad, cfg.feature_pad),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    out_path = Path(cfg.model_save_path)
    if out_path.is_file() and not args.no_backup:
        import shutil

        bak = Path(cfg.backup_old_path)
        if not bak.is_file():
            shutil.copy2(out_path, bak)
            print(f"已备份旧模型 -> {bak}")

    model = build_model(ARCH, len(class_names), pretrained=True).to(cfg.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_acc = 0.0
    tl_hist, vl_hist, ta_hist, va_hist = [], [], [], []

    set_backbone_trainable(model, ARCH, trainable=False)
    optimizer = _optimizer_for_phase(model, ARCH, False, cfg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.num_epochs_head, 1)
    )

    print("\n--- 阶段 1：分类头（主体+特征样本）---")
    for epoch in range(cfg.num_epochs_head):
        print(f"\nHead {epoch + 1}/{cfg.num_epochs_head}")
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        vl, va = validate(model, val_loader, criterion, cfg.device)
        scheduler.step()
        tl_hist.append(tl)
        vl_hist.append(vl)
        ta_hist.append(ta)
        va_hist.append(va)
        print(f"  train acc={ta:.2f}% | val(body) acc={va:.2f}%")
        if va > best_acc:
            best_acc = va
            save_checkpoint(
                model, class_names, va, cfg, ARCH, "industrial_body_feature"
            )
            print(f"  [√] 保存 {cfg.model_save_path} val(body)={va:.2f}%")

    print("\n--- 阶段 2：全模型微调 ---")
    set_backbone_trainable(model, ARCH, trainable=True)
    optimizer = _optimizer_for_phase(model, ARCH, True, cfg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.num_epochs_finetune, 1)
    )
    for epoch in range(cfg.num_epochs_finetune):
        print(f"\nFinetune {epoch + 1}/{cfg.num_epochs_finetune}")
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        vl, va = validate(model, val_loader, criterion, cfg.device)
        scheduler.step()
        tl_hist.append(tl)
        vl_hist.append(vl)
        ta_hist.append(ta)
        va_hist.append(va)
        print(f"  train acc={ta:.2f}% | val(body) acc={va:.2f}%")
        if va > best_acc:
            best_acc = va
            save_checkpoint(
                model, class_names, va, cfg, ARCH, "industrial_body_feature"
            )
            print(f"  [√] 保存 {cfg.model_save_path} val(body)={va:.2f}%")

    plot_curves(tl_hist, vl_hist, ta_hist, va_hist)
    print("\n" + "=" * 60)
    print(f"完成。最佳 val(主体框) 准确率: {best_acc:.2f}%")
    print(f"权重: {cfg.model_save_path}")
    print("部署: 替换云端/Pi 的 best_elephant_model.pth 后重启服务")
    print("=" * 60)


if __name__ == "__main__":
    main()
