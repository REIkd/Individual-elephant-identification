"""
在已有 checkpoint 上微调单个（或多个）象类，默认混少量其他象样本防遗忘。

用法（玛丽亚补训）:
  cd elephant_training
  set ELEPHANT_DATASET=..\\dataset
  python fix_maria_voc_tags.py
  python finetune_elephant.py --elephant 玛丽亚 --base-model ..\\best_elephant_model.pth --output ..\\best_elephant_model.pth
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from elephant_net import build_model, set_backbone_trainable
from paths import DATASET_DIR, MODEL_PATH, ensure_torch_home
from train_industrial import (
    ARCH,
    CropSample,
    IndustrialCropDataset,
    TrainConfig,
    collect_image_records,
    expand_crops,
    get_transforms,
    save_checkpoint,
    set_seed,
    train_epoch,
    validate,
)

ensure_torch_home()


def _build_loaders(
    records: list,
    class_names: list[str],
    target_names: set[str],
    *,
    val_split: float,
    seed: int,
    mix_ratio: float,
    batch_size: int,
    image_size: int,
    body_only: bool = False,
    mix_all: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, list[CropSample], list[CropSample], list[CropSample]]:
    labels = [r.label for r in records]
    train_rec, val_rec, _, _ = train_test_split(
        records, labels, test_size=val_split, random_state=seed, stratify=labels
    )

    target_train = [r for r in train_rec if r.elephant_name in target_names]
    other_train = [r for r in train_rec if r.elephant_name not in target_names]
    if not target_train:
        raise RuntimeError(f"训练集中未找到目标象: {sorted(target_names)}")

    target_crops = expand_crops(
        target_train, use_body=True, use_features=not body_only
    )
    if mix_all:
        other_crops = expand_crops(other_train, use_body=True, use_features=False)
    else:
        other_crops = expand_crops(
            other_train, use_body=True, use_features=not body_only
        )
        if mix_ratio > 0 and other_crops:
            n_mix = min(len(other_crops), max(1, int(len(target_crops) * mix_ratio)))
            rng = random.Random(seed)
            other_crops = rng.sample(other_crops, n_mix)

    train_crops = target_crops + other_crops
    target_val_crops = expand_crops(
        [r for r in val_rec if r.elephant_name in target_names],
        use_body=True,
        use_features=False,
    )
    full_val_crops = expand_crops(val_rec, use_body=True, use_features=False)

    train_tf, val_tf = get_transforms(image_size)
    cfg = TrainConfig()
    train_loader = DataLoader(
        IndustrialCropDataset(train_crops, train_tf, cfg.body_pad, cfg.feature_pad),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    target_val_loader = DataLoader(
        IndustrialCropDataset(target_val_crops, val_tf, cfg.body_pad, cfg.feature_pad),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    full_val_loader = DataLoader(
        IndustrialCropDataset(full_val_crops, val_tf, cfg.body_pad, cfg.feature_pad),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        train_loader,
        target_val_loader,
        full_val_loader,
        train_crops,
        target_val_crops,
        full_val_crops,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="单象类增量微调")
    parser.add_argument("--elephant", action="append", required=True, help="象名，可重复指定")
    parser.add_argument("--base-model", type=str, default="", help="起始权重")
    parser.add_argument("--output", type=str, default="", help="输出权重")
    parser.add_argument("--epochs-head", type=int, default=3, help="仅训分类头 epoch")
    parser.add_argument("--epochs-ft", type=int, default=6, help="低学习率全网络 epoch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mix-ratio", type=float, default=0.15, help="其他象训练 patch 占比上限")
    parser.add_argument(
        "--mix-all",
        action="store_true",
        help="回放全部其他象主体框 patch（防遗忘，推荐）",
    )
    parser.add_argument(
        "--select-val",
        choices=("full", "target"),
        default="full",
        help="保存 checkpoint 时依据的验证集（默认 full=全体 17 象）",
    )
    parser.add_argument("--body-only", action="store_true", help="仅用主体框 patch（更快，适合补训单象）")
    parser.add_argument("--head-only", action="store_true", help="跳过全网络微调，仅训分类头")
    parser.add_argument("--lr-head", type=float, default=2e-4, help="分类头学习率")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    target_names = {str(x).strip() for x in args.elephant if str(x).strip()}
    base_path = Path(args.base_model or MODEL_PATH)
    out_path = Path(args.output or MODEL_PATH)
    if not base_path.is_file():
        raise FileNotFoundError(base_path)

    ckpt = torch.load(str(base_path), map_location="cpu", weights_only=False)
    class_names: list[str] = list(ckpt["class_names"])
    arch = ckpt.get("arch", ARCH)
    image_size = int(ckpt.get("image_size", 384 if arch == "efficientnet_v2_s" else 224))
    for name in target_names:
        if name not in class_names:
            raise ValueError(f"权重中无类别 {name!r}，现有: {class_names}")

    records, _ = collect_image_records()
    cfg = TrainConfig(
        image_size=image_size,
        batch_size=args.batch_size,
        num_epochs_head=args.epochs_head,
        num_epochs_finetune=args.epochs_ft,
        lr_head=args.lr_head,
        lr_backbone=1e-5,
        model_save_path=str(out_path),
    )
    device = cfg.device

    (
        train_loader,
        target_val_loader,
        full_val_loader,
        train_crops,
        target_val_crops,
        full_val_crops,
    ) = _build_loaders(
        records,
        class_names,
        target_names,
        val_split=cfg.val_split,
        seed=args.seed,
        mix_ratio=args.mix_ratio,
        batch_size=cfg.batch_size,
        image_size=image_size,
        body_only=args.body_only,
        mix_all=args.mix_all,
    )

    print("=" * 60)
    print("单象类增量微调")
    print("=" * 60)
    print(f"目标象: {', '.join(sorted(target_names))}")
    print(f"base: {base_path.resolve()}")
    print(f"out:  {out_path.resolve()}")
    print(
        f"train patches: {len(train_crops)} | "
        f"target val(body): {len(target_val_crops)} | full val(body): {len(full_val_crops)}"
    )
    print(
        f"mix_all: {args.mix_all} | mix_ratio(other): {args.mix_ratio} | "
        f"body_only: {args.body_only} | select_val: {args.select_val}"
    )

    model = build_model(arch, len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_target_val = 0.0
    best_full_val = 0.0
    ckpt_val = float(ckpt.get("val_acc") or 0.0)

    _, baseline_full_val = validate(model, full_val_loader, criterion, device)
    _, baseline_target_val = validate(model, target_val_loader, criterion, device)
    print(
        f"baseline full val(body)={baseline_full_val:.2f}% | "
        f"target val(body)={baseline_target_val:.2f}%"
    )
    best_full_val = baseline_full_val
    best_target_val = baseline_target_val

    def maybe_save(phase: str) -> None:
        nonlocal best_target_val, best_full_val
        _, target_va = validate(model, target_val_loader, criterion, device)
        _, full_va = validate(model, full_val_loader, criterion, device)
        score = full_va if args.select_val == "full" else target_va
        best_score = best_full_val if args.select_val == "full" else best_target_val
        if score > best_score:
            if args.select_val == "full":
                best_full_val = full_va
            else:
                best_target_val = target_va
            best_target_val = max(best_target_val, target_va)
            best_full_val = max(best_full_val, full_va)
            save_checkpoint(
                model,
                class_names,
                full_va,
                cfg,
                arch,
                f"finetune_{'_'.join(sorted(target_names))}_{phase}",
            )
            print(
                f"  [√] 保存 {out_path} full_val={full_va:.2f}% "
                f"target_val={target_va:.2f}%"
            )
        else:
            print(
                f"  [-] 未保存 full_val={full_va:.2f}% target_val={target_va:.2f}% "
                f"(best {args.select_val}={best_score:.2f}%)"
            )

    set_backbone_trainable(model, arch, trainable=False)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    print("\n--- 阶段 1：仅分类头 ---")
    for epoch in range(cfg.num_epochs_head):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  epoch {epoch + 1}: train={ta:.2f}%")
        maybe_save("head")

    if not args.head_only and cfg.num_epochs_finetune > 0:
        set_backbone_trainable(model, arch, trainable=True)
        optimizer = optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and ("features" in n or n.startswith("conv"))
                    ],
                    "lr": cfg.lr_backbone,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and not ("features" in n or n.startswith("conv"))
                    ],
                    "lr": cfg.lr_head * 0.25,
                },
            ],
            weight_decay=cfg.weight_decay,
        )
        print("\n--- 阶段 2：低学习率全网络 ---")
        for epoch in range(cfg.num_epochs_finetune):
            tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"  epoch {epoch + 1}: train={ta:.2f}%")
            maybe_save("ft")

    if best_full_val <= baseline_full_val and best_target_val <= baseline_target_val:
        save_checkpoint(
            model,
            class_names,
            ckpt_val,
            cfg,
            arch,
            f"finetune_{'_'.join(sorted(target_names))}_last",
        )
        print(f"  [!] 目标象 val 未提升，保存末轮权重到 {out_path}")

    report = {
        "target_elephants": sorted(target_names),
        "base_model": str(base_path.resolve()),
        "output_model": str(out_path.resolve()),
        "train_patches": len(train_crops),
        "target_val_body_patches": len(target_val_crops),
        "full_val_body_patches": len(full_val_crops),
        "checkpoint_val_acc": ckpt_val,
        "baseline_full_val_acc": baseline_full_val,
        "baseline_target_val_acc": baseline_target_val,
        "best_full_val_acc": best_full_val,
        "best_target_val_acc": best_target_val,
        "select_val": args.select_val,
        "mix_all": args.mix_all,
    }
    report_path = out_path.parent / "reports" / "training" / f"finetune_{'_'.join(sorted(target_names))}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n完成。best full val(body)={best_full_val:.2f}% | "
        f"best target val(body)={best_target_val:.2f}%"
    )
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
