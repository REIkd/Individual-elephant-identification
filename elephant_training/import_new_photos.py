"""
把新拍照片（按象名分子文件夹）YOLO 裁象体后追加到 dataset/，供 finetune.py 微调。

目录示例:
  新拍照片/
    安妮/photo1.jpg
    玛丽亚/IMG_002.jpg

用法:
  python import_new_photos.py --src 新拍照片
  python finetune.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from yolo_crop import imread_bgr, normalize_image_path, yolo_elephant_roi_bgr

from paths import DATASET_DIR, MODEL_PATH

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def import_folder(
    src_dir: Path,
    dataset_dir: Path,
    *,
    copy_only: bool = False,
    yolo_weights: str = "yolov8n.pt",
    gpu: int = -1,
    pad_ratio: float = 0.12,
) -> None:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    total, ok, skip = 0, 0, 0

    subdirs = [d for d in sorted(src_dir.iterdir()) if d.is_dir()]
    if not subdirs:
        raise RuntimeError(f"{src_dir} 下没有象名子文件夹")

    for sub in subdirs:
        elephant = sub.name
        out_dir = dataset_dir / elephant
        out_dir.mkdir(parents=True, exist_ok=True)
        images = [f for f in sub.iterdir() if f.suffix in IMAGE_EXT]
        if not images:
            print(f"[跳过] {elephant}: 无图片")
            continue

        for img_path in tqdm(images, desc=elephant):
            total += 1
            stem = f"new_{img_path.stem}"
            out_path = out_dir / f"{stem}.jpg"

            if copy_only:
                shutil.copy2(img_path, out_dir / img_path.name)
                ok += 1
                continue

            frame = imread_bgr(img_path)
            if frame is None:
                print(f"  无法读取: {img_path}")
                skip += 1
                continue

            roi, detail = yolo_elephant_roi_bgr(
                frame,
                yolo_weights=yolo_weights,
                cuda_device=gpu,
                pad_ratio=pad_ratio,
            )
            if roi is None:
                print(f"  未检出大象: {img_path.name} ({detail})")
                skip += 1
                continue

            cv2.imencode(".jpg", roi)[1].tofile(str(out_path))
            ok += 1

    print("\n" + "=" * 50)
    print(f"完成: 成功 {ok} | 跳过 {skip} | 共 {total}")
    print(f"已写入: {dataset_dir.resolve()}")
    print("下一步: python finetune.py")


def main():
    p = argparse.ArgumentParser(description="新拍照片导入 dataset（YOLO 裁象体）")
    p.add_argument("--src", required=True, help="源目录，子文件夹名为象名")
    p.add_argument("--dataset", default=str(DATASET_DIR), help="目标 dataset 目录")
    p.add_argument(
        "--copy-only",
        action="store_true",
        help="不裁剪，仅复制（仅当照片本身已是象体特写时用）",
    )
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--gpu", type=int, default=-1)
    args = p.parse_args()
    import_folder(
        normalize_image_path(args.src),
        Path(args.dataset),
        copy_only=args.copy_only,
        yolo_weights=args.yolo_weights,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
