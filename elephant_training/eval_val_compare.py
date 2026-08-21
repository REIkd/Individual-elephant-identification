"""在固定 val 划分上对比 body-only 与 body+feature 权重（VOC 主体框 crop，与 train_industrial 一致）。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(1, str(_ROOT))

from elephant_net import build_model
from train_industrial import elephant_name_from_folder, pad_box, parse_voc_crops


def load_clf(model_path: Path) -> tuple[torch.nn.Module, list[str], transforms.Compose, dict]:
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_v2_s")
    image_size = int(ckpt.get("image_size", 384))
    names: list[str] = ckpt["class_names"]
    model = build_model(arch, len(names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, names, tf, ckpt


def collect_val_samples(dataset_dir: Path) -> list[tuple[Path, str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    samples: list[tuple[Path, str]] = []
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        true = elephant_name_from_folder(folder.name)
        for img in folder.iterdir():
            if img.suffix in exts:
                samples.append((img, true))
    paths = [str(p) for p, _ in samples]
    labels = [n for _, n in samples]
    _, val_paths, _, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return [(Path(p), n) for p, n in zip(val_paths, val_labels)]


def eval_model(
    model: torch.nn.Module,
    class_names: list[str],
    tf: transforms.Compose,
    val_samples: list[tuple[Path, str]],
) -> tuple[float, int, int, dict[str, dict[str, float | int]]]:
    correct = 0
    per: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "accuracy_pct": 0.0}
    )
    for img_path, true_name in tqdm(val_samples, leave=False):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        xml_path = img_path.with_suffix(".xml")
        body, _ = parse_voc_crops(xml_path, true_name) if xml_path.is_file() else (None, [])
        if body is not None:
            img = img.crop(pad_box(body, w, h, 0.12))
        x = tf(img).unsqueeze(0)
        with torch.inference_mode():
            pred = int(model(x).argmax(1).item())
        pred_name = class_names[pred]
        per[true_name]["total"] = int(per[true_name]["total"]) + 1
        if pred_name == true_name:
            correct += 1
            per[true_name]["correct"] = int(per[true_name]["correct"]) + 1
    total = len(val_samples)
    for name, st in per.items():
        t = int(st["total"])
        st["accuracy_pct"] = round(int(st["correct"]) / t * 100, 2) if t else 0.0
    acc = correct / total * 100 if total else 0.0
    return acc, total, correct, dict(per)


def main() -> None:
    dataset = _ROOT / "dataset"
    models = {
        "Config_B_body_only": _ROOT / "best_elephant_model_body_only_backup.pth",
        "Config_C_body_feature": _ROOT / "best_elephant_model.pth",
        "Config_C_maria_ft": _ROOT / "best_elephant_model_maria_ft.pth",
    }
    val_samples = collect_val_samples(dataset)
    print(f"Val: {len(val_samples)} images | classes in val: {len({n for _, n in val_samples})}")

    report: dict = {
        "val_n": len(val_samples),
        "protocol": "VOC body box (parse_voc_crops), split 80/20 seed=42",
        "models": {},
    }
    for key, path in models.items():
        if not path.is_file():
            print(f"SKIP missing: {path}")
            continue
        model, names, tf, ckpt = load_clf(path)
        acc, total, correct, per = eval_model(model, names, tf, val_samples)
        ckpt_val = ckpt.get("val_acc")
        ckpt_str = f"{ckpt_val:.2f}%" if ckpt_val is not None else "n/a"
        print(
            f"\n{key}\n"
            f"  path: {path.name}\n"
            f"  n_classes: {len(names)}\n"
            f"  checkpoint val_acc: {ckpt_str}\n"
            f"  reproduced val_acc: {acc:.2f}% ({correct}/{total})"
        )
        report["models"][key] = {
            "path": str(path),
            "n_classes": len(names),
            "class_names": names,
            "checkpoint_val_acc": float(ckpt_val) if ckpt_val is not None else None,
            "reproduced_val_acc": round(acc, 4),
            "correct": correct,
            "total": total,
            "per_class": per,
        }

    out = _ROOT / "reports" / "training" / "val_compare_body_vs_feature.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
