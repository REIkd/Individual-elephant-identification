"""
大象个体识别训练：仅使用 dataset/<象名>/ 下图片；优先按 VOC XML 裁剪象体区域（工业场景更利于细粒度识别）。
默认骨干为 EfficientNet-V2-S，冻结预热后再全量微调。
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

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

ensure_torch_home()


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed()

# 可在此切换骨干：efficientnet_v2_s（推荐，准确率优先）或 resnet50（更快、更省显存）
ARCH = os.environ.get("ELEPHANT_ARCH", "efficientnet_v2_s")


class Config:
    image_size = 384 if ARCH == "efficientnet_v2_s" else 224
    batch_size = int(os.environ.get("ELEPHANT_BATCH", "16"))
    num_workers = min(4, os.cpu_count() or 1)
    num_epochs_head = int(os.environ.get("ELEPHANT_EPOCH_HEAD", "12"))
    num_epochs_finetune = int(os.environ.get("ELEPHANT_EPOCH_FT", "40"))
    lr_head = 1e-3
    lr_backbone = 3e-5
    weight_decay = 0.02
    val_split = 0.2
    label_smoothing = 0.08
    model_save_path = str(MODEL_PATH)
    class_names_path = str(CLASS_NAMES_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bbox_pad_ratio = 0.12


config = Config()


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
    box: tuple[int, int, int, int], w: int, h: int, pad_ratio: float
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    bw, bh = xmax - xmin, ymax - ymin
    pad_w, pad_h = int(bw * pad_ratio), int(bh * pad_ratio)
    xmin = max(0, xmin - pad_w)
    ymin = max(0, ymin - pad_h)
    xmax = min(w, xmax + pad_w)
    ymax = min(h, ymax + pad_h)
    return xmin, ymin, xmax, ymax


class ElephantCropDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        xml_path = img_path.with_suffix(".xml")
        if xml_path.is_file():
            boxes = _parse_voc_boxes(xml_path)
            if boxes:
                w, h = image.size
                xmin, ymin, xmax, ymax = _pad_box(
                    boxes[0], w, h, config.bbox_pad_ratio
                )
                image = image.crop((xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)
        return image, label


def collect_samples() -> tuple[list[tuple[Path, int]], list[str]]:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"缺少数据目录: {DATASET_DIR}")

    elephant_folders = sorted(
        [d for d in DATASET_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    class_names = [d.name for d in elephant_folders]
    class_to_idx = {n: i for i, n in enumerate(class_names)}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    samples: list[tuple[Path, int]] = []
    for folder in elephant_folders:
        label = class_to_idx[folder.name]
        for f in folder.iterdir():
            if f.suffix in exts:
                samples.append((f, label))

    if not samples:
        raise RuntimeError(f"在 {DATASET_DIR} 下未发现图片文件")

    with open(config.class_names_path, "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, ensure_ascii=False, indent=2)

    return samples, class_names


def get_transforms():
    sz = config.image_size
    train_tf = transforms.Compose(
        [
            transforms.Resize(int(sz * 1.15)),
            transforms.RandomCrop(sz),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.25, 0.25, 0.2, 0.05),
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


def _optimizer_for_phase(model, arch: str, finetune_full: bool):
    arch_l = arch.lower().strip()
    if not finetune_full:
        if arch_l == "resnet50":
            params = list(model.fc.parameters())
        else:
            params = list(model.classifier.parameters())
        return optim.AdamW(params, lr=config.lr_head, weight_decay=config.weight_decay)

    if arch_l == "resnet50":
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc")]
        head = [p for n, p in model.named_parameters() if n.startswith("fc")]
    else:
        backbone = [p for n, p in model.named_parameters() if n.startswith("features")]
        head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    return optim.AdamW(
        [
            {"params": backbone, "lr": config.lr_backbone},
            {"params": head, "lr": config.lr_head * 0.25},
        ],
        weight_decay=config.weight_decay,
    )


def main():
    print("=" * 56)
    print("大象个体识别训练（VOC 裁剪 + 迁移学习）")
    print("=" * 56)
    print(f"设备: {config.device} | 骨干: {ARCH} | 输入边长: {config.image_size}")

    samples, class_names = collect_samples()
    labels = [s[1] for s in samples]
    print(f"类别数: {len(class_names)} | 样本数: {len(samples)}")

    train_s, val_s, _, _ = train_test_split(
        samples,
        labels,
        test_size=config.val_split,
        random_state=42,
        stratify=labels,
    )
    print(f"训练 {len(train_s)} | 验证 {len(val_s)}")

    train_tf, val_tf = get_transforms()
    train_loader = DataLoader(
        ElephantCropDataset(train_s, train_tf),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ElephantCropDataset(val_s, val_tf),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(ARCH, len(class_names), pretrained=True)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 阶段 1：仅分类头
    set_backbone_trainable(model, ARCH, trainable=False)
    optimizer = _optimizer_for_phase(model, ARCH, finetune_full=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.num_epochs_head, 1)
    )

    print("\n--- 阶段 1：训练分类头（骨干冻结）---")
    for epoch in range(config.num_epochs_head):
        print(f"\nHead Epoch {epoch + 1}/{config.num_epochs_head}")
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, config.device)
        vl, va = validate(model, val_loader, criterion, config.device)
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        scheduler.step()
        print(f"  train loss={tl:.4f} acc={ta:.2f}% | val loss={vl:.4f} acc={va:.2f}%")
        if va > best_acc:
            best_acc = va
            _save_checkpoint(model, class_names, va, ARCH)
            print(f"  [√] 保存最佳模型 val_acc={va:.2f}%")

    # 阶段 2：全模型微调
    print("\n--- 阶段 2：微调骨干网络 ---")
    set_backbone_trainable(model, ARCH, trainable=True)
    optimizer = _optimizer_for_phase(model, ARCH, finetune_full=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.num_epochs_finetune, 1)
    )

    for epoch in range(config.num_epochs_finetune):
        print(f"\nFinetune Epoch {epoch + 1}/{config.num_epochs_finetune}")
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, config.device)
        vl, va = validate(model, val_loader, criterion, config.device)
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        scheduler.step()
        print(f"  train loss={tl:.4f} acc={ta:.2f}% | val loss={vl:.4f} acc={va:.2f}%")
        if va > best_acc:
            best_acc = va
            _save_checkpoint(model, class_names, va, ARCH)
            print(f"  [√] 保存最佳模型 val_acc={va:.2f}%")

    _plot_curves(train_losses, val_losses, train_accs, val_accs)
    print("\n" + "=" * 56)
    print(f"完成。最佳验证准确率: {best_acc:.2f}%")
    print(f"权重: {config.model_save_path} | 类别: {config.class_names_path}")
    print("=" * 56)


def _save_checkpoint(model, class_names, val_acc, arch: str):
    torch.save(
        {
            "arch": arch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "class_names": class_names,
            "image_size": config.image_size,
        },
        config.model_save_path,
    )


def _plot_curves(tl, vl, ta, va):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tl, label="train")
    plt.plot(vl, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(ta, label="train")
    plt.plot(va, label="val")
    plt.xlabel("epoch")
    plt.ylabel("acc %")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    out = "training_curves.png"
    plt.savefig(out, dpi=150)
    print(f"曲线已保存: {out}")


if __name__ == "__main__":
    main()
