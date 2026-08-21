"""
在新照片加入 dataset/ 后，从已有 checkpoint 继续微调（适应新相机/新场景）。

用法:
  python import_new_photos.py --src 新拍照片
  python finetune.py
  python finetune.py --epochs 25 --lr 2e-5
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from elephant_net import build_model, set_backbone_trainable
from paths import MODEL_PATH
from train import (
    Config,
    ElephantCropDataset,
    collect_samples,
    config,
    get_transforms,
    train_epoch,
    validate,
    _save_checkpoint,
    ARCH,
)


def finetune(resume: str, epochs: int, lr: float, batch_size: int | None):
    if not os.path.isfile(resume):
        raise FileNotFoundError(f"找不到权重: {resume}")

    ckpt = torch.load(resume, map_location=config.device, weights_only=False)
    class_names = ckpt["class_names"]
    arch = ckpt.get("arch", ARCH)
    image_size = int(ckpt.get("image_size", config.image_size))

    print("=" * 56)
    print("微调：适应新拍照片")
    print("=" * 56)
    print(f"设备: {config.device} | 骨干: {arch} | 输入: {image_size}")
    print(f"从 checkpoint 恢复: {resume} (val_acc={ckpt.get('val_acc', 0):.2f}%)")

    samples, names = collect_samples()
    if names != class_names:
        print("[警告] dataset 类别与 checkpoint 不完全一致，以 dataset 为准")
    labels = [s[1] for s in samples]
    print(f"样本数: {len(samples)} | 类别: {len(names)}")

    train_s, val_s, _, _ = train_test_split(
        samples, labels, test_size=config.val_split, random_state=42, stratify=labels
    )
    print(f"训练 {len(train_s)} | 验证 {len(val_s)}")

    train_tf, val_tf = get_transforms()
    bs = batch_size or max(4, config.batch_size // 2)
    train_loader = DataLoader(
        ElephantCropDataset(train_s, train_tf),
        batch_size=bs,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ElephantCropDataset(val_s, val_tf),
        batch_size=bs,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(arch, len(names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(config.device)

    set_backbone_trainable(model, arch, trainable=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    best_acc = float(ckpt.get("val_acc", 0.0))
    backup_path = "best_elephant_model_before_finetune.pth"
    if os.path.isfile(config.model_save_path) and not os.path.isfile(backup_path):
        import shutil

        shutil.copy2(config.model_save_path, backup_path)
        print(f"已备份原模型 -> {backup_path}")

    print(f"\n--- 微调 {epochs} 轮 (lr={lr}) ---")
    for epoch in range(epochs):
        print(f"\nFinetune {epoch + 1}/{epochs}")
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, config.device)
        vl, va = validate(model, val_loader, criterion, config.device)
        scheduler.step()
        print(f"  train acc={ta:.2f}% | val acc={va:.2f}%")
        if va > best_acc:
            best_acc = va
            _save_checkpoint(model, names, va, arch)
            print(f"  [√] 保存 best_elephant_model.pth val_acc={va:.2f}%")

    print("\n完成。请用 classifier.py 或项目根目录 predict.py 测试新权重。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default=str(MODEL_PATH))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()
    finetune(args.resume, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
