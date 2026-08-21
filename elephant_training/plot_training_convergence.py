"""
从 train.log 与已训练权重生成矢量图，用于检查 loss/acc 收敛与特征空间聚类。

输出（默认 reports/training/）：
  training_curves.svg          — 全程 loss / 准确率
  training_finetune_zoom.svg   — 微调阶段放大
  feature_tsne.svg             — 验证样本特征 t-SNE（看类间是否分开）
  feature_pca.svg              — 验证样本特征 PCA
  class_separation.svg         — 类内/类间余弦距离（越大越好 = 特征越收敛）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torchvision import transforms

from elephant_net import build_model

from paths import DATASET_DIR, REPORTS_DIR, ensure_torch_home

ensure_torch_home()

ROOT = Path(__file__).resolve().parent
EPOCH_LINE = re.compile(
    r"train loss=([\d.]+) acc=([\d.]+)% \| val loss=([\d.]+) acc=([\d.]+)%"
)


def _read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _setup_cjk_font() -> None:
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def parse_train_log(log_path: Path) -> dict:
    phases: list[str] = []
    tl, vl, ta, va = [], [], [], []
    if not log_path.is_file():
        return {"phases": phases, "train_loss": tl, "val_loss": vl, "train_acc": ta, "val_acc": va}

    mode = "head"
    for line in _read_text_auto(log_path).splitlines():
        if "Head Epoch" in line:
            mode = "head"
        elif "Finetune Epoch" in line or "Finetune " in line and "/40" in line:
            mode = "finetune"
        m = EPOCH_LINE.search(line)
        if m:
            tl.append(float(m.group(1)))
            ta.append(float(m.group(2)))
            vl.append(float(m.group(3)))
            va.append(float(m.group(4)))
            phases.append(mode)
    return {
        "phases": phases,
        "train_loss": tl,
        "val_loss": vl,
        "train_acc": ta,
        "val_acc": va,
    }


def _save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {path.resolve()}")


def plot_training_curves(hist: dict, out_dir: Path) -> None:
    tl = hist["train_loss"]
    if not tl:
        print("train.log 中未解析到 epoch 指标，跳过训练曲线。")
        return

    phases = hist["phases"]
    xs = list(range(1, len(tl) + 1))
    head_n = sum(1 for p in phases if p == "head")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, y_train, y_val, ylabel, title in (
        (axes[0], tl, hist["val_loss"], "Loss", "Loss 收敛"),
        (axes[1], hist["train_acc"], hist["val_acc"], "Accuracy (%)", "准确率收敛"),
    ):
        ax.plot(xs, y_train, "o-", markersize=3, label="train", color="#2563eb")
        ax.plot(xs, y_val, "s-", markersize=3, label="val", color="#dc2626")
        if 0 < head_n < len(xs):
            ax.axvline(head_n + 0.5, color="#94a3b8", linestyle="--", linewidth=1)
            ax.text(
                head_n + 0.6,
                ax.get_ylim()[1] * 0.95 if ylabel == "Loss" else ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "微调开始",
                fontsize=9,
                color="#64748b",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("大象个体识别训练曲线（EfficientNet-V2-S）", fontsize=13)
    fig.tight_layout()
    _save_svg(fig, out_dir / "training_curves.svg")

    if head_n < len(tl):
        xs2 = xs[head_n:]
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, y_train, y_val, ylabel, title in (
            (axes2[0], tl[head_n:], hist["val_loss"][head_n:], "Loss", "微调阶段 Loss"),
            (axes2[1], hist["train_acc"][head_n:], hist["val_acc"][head_n:], "Accuracy (%)", "微调阶段准确率"),
        ):
            ax.plot(xs2, y_train, "o-", markersize=4, label="train", color="#2563eb")
            ax.plot(xs2, y_val, "s-", markersize=4, label="val", color="#dc2626")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig2.suptitle("微调阶段放大（观察是否已平台期）", fontsize=13)
        fig2.tight_layout()
        _save_svg(fig2, out_dir / "training_finetune_zoom.svg")


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


def _pad_box(box, w, h, pad_ratio=0.12):
    xmin, ymin, xmax, ymax = box
    bw, bh = xmax - xmin, ymax - ymin
    pad_w, pad_h = int(bw * pad_ratio), int(bh * pad_ratio)
    return (
        max(0, xmin - pad_w),
        max(0, ymin - pad_h),
        min(w, xmax + pad_w),
        min(h, ymax + pad_h),
    )


def collect_val_samples(
    max_per_class: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[int], list[str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"未找到数据集目录: {DATASET_DIR}")

    class_names = [d.name for d in class_dirs]
    all_paths: list[Path] = []
    all_labels: list[int] = []
    for i, d in enumerate(class_dirs):
        imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
        for p in imgs:
            all_paths.append(p)
            all_labels.append(i)

    train_idx, val_idx = train_test_split(
        range(len(all_paths)),
        test_size=val_ratio,
        random_state=seed,
        stratify=all_labels,
    )
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    by_cls: dict[int, list[int]] = {}
    for j, lb in enumerate(val_labels):
        by_cls.setdefault(lb, []).append(j)
    picked: list[int] = []
    rng = random.Random(seed)
    for lb, idxs in by_cls.items():
        rng.shuffle(idxs)
        picked.extend(idxs[:max_per_class])

    paths = [val_paths[i] for i in picked]
    labels = [val_labels[i] for i in picked]
    return paths, labels, class_names


def load_model(model_path: Path):
    try:
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(model_path), map_location="cpu")

    if isinstance(ckpt, dict) and "class_names" in ckpt:
        class_names = ckpt["class_names"]
    else:
        with open(ROOT / "class_names.json", encoding="utf-8") as f:
            class_names = json.load(f)

    arch = ckpt.get("arch", "efficientnet_v2_s") if isinstance(ckpt, dict) else "efficientnet_v2_s"
    image_size = int(ckpt.get("image_size", 384)) if isinstance(ckpt, dict) else 384
    model = build_model(arch, len(class_names), pretrained=False)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, arch, image_size, class_names


@torch.no_grad()
def extract_features(model, arch: str, batch: torch.Tensor) -> torch.Tensor:
    if arch == "efficientnet_v2_s":
        x = model.features(batch)
        x = model.avgpool(x)
        return torch.flatten(x, 1)
    if arch == "resnet50":
        x = model.conv1(batch)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)
    raise ValueError(arch)


def embed_dataset(
    model_path: Path,
    max_per_class: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths, labels, class_names = collect_val_samples(max_per_class, 0.2, seed)
    model, arch, image_size, ckpt_names = load_model(model_path)
    if ckpt_names != class_names:
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        labels = [name_to_idx[class_names[lb]] for lb in labels]

    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    feats: list[np.ndarray] = []
    lbls: list[int] = []
    batch_imgs: list[torch.Tensor] = []
    batch_lbs: list[int] = []

    def _flush():
        if not batch_imgs:
            return
        t = torch.stack(batch_imgs)
        f = extract_features(model, arch, t).cpu().numpy()
        feats.append(f)
        lbls.extend(batch_lbs)
        batch_imgs.clear()
        batch_lbs.clear()

    for p, lb in zip(paths, labels):
        img = Image.open(p).convert("RGB")
        xml = p.with_suffix(".xml")
        if xml.is_file():
            boxes = _parse_voc_boxes(xml)
            if boxes:
                box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x0, y0, x1, y1 = _pad_box(box, img.width, img.height)
                img = img.crop((x0, y0, x1, y1))
        batch_imgs.append(transform(img))
        batch_lbs.append(lb)
        if len(batch_imgs) >= batch_size:
            _flush()
    _flush()

    return np.vstack(feats), np.array(lbls), class_names


def _class_colors(n: int) -> list:
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(n, 1))
    return [cmap(i) for i in range(n)]


def plot_embedding_2d(
    feats: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    reducer_name: str,
    out_path: Path,
    seed: int,
) -> None:
    if len(feats) < 3:
        print(f"样本过少，跳过 {reducer_name}。")
        return

    if reducer_name == "pca":
        xy = PCA(n_components=2, random_state=seed).fit_transform(feats)
        title = "特征 PCA（验证集样本）"
        xlabel, ylabel = "PC1", "PC2"
    else:
        perplexity = min(30, max(5, len(feats) // 4))
        xy = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(feats)
        title = "特征 t-SNE（验证集样本 · 类簇越分离表示特征越收敛）"
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"

    colors = _class_colors(len(class_names))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        mask = labels == i
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=28,
            alpha=0.75,
            c=[colors[i]],
            label=name,
            edgecolors="white",
            linewidths=0.3,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        markerscale=1.2,
        frameon=True,
    )
    fig.tight_layout()
    _save_svg(fig, out_path)


def plot_class_separation(
    feats: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    """类内平均距离 vs 类间平均距离（归一化特征余弦距离）。"""
    norms = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    intra: list[float] = []
    inter: list[float] = []
    names_ok: list[str] = []

    for i, name in enumerate(class_names):
        idx = np.where(labels == i)[0]
        if len(idx) < 2:
            continue
        sub = norms[idx]
        dists = 1.0 - sub @ sub.T
        tri = dists[np.triu_indices(len(idx), k=1)]
        intra.append(float(np.mean(tri)))

        other = np.where(labels != i)[0]
        if len(other) == 0:
            continue
        cross = 1.0 - sub @ norms[other].T
        inter.append(float(np.mean(cross)))
        names_ok.append(name)

    if not intra:
        print("样本不足，跳过类间分离图。")
        return

    x = np.arange(len(names_ok))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(names_ok) * 0.55), 5))
    ax.bar(x - w / 2, intra, w, label="类内距离（越小越好）", color="#3b82f6")
    ax.bar(x + w / 2, inter, w, label="类间距离（越大越好）", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(names_ok, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("1 - 余弦相似度")
    ax.set_title("各类特征分离度（类间应明显高于类内 → 特征已收敛）")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_svg(fig, out_path)

    ratio = np.mean(inter) / (np.mean(intra) + 1e-8)
    print(f"平均 类间/类内 距离比: {ratio:.2f}（>2 通常表示分得较开）")


def main():
    parser = argparse.ArgumentParser(description="生成训练收敛与特征聚类矢量图")
    parser.add_argument("--log", type=str, default="train.log")
    parser.add_argument("--model", type=str, default="best_elephant_model.pth")
    parser.add_argument("--out_dir", type=str, default="reports/training")
    parser.add_argument("--max_per_class", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_features", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _setup_cjk_font()
    hist = parse_train_log(Path(args.log))
    plot_training_curves(hist, out_dir)

    if args.skip_features:
        return

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"未找到权重 {model_path}，跳过特征图。")
        return

    print("正在提取验证集特征（用于 t-SNE / PCA）…")
    feats, labels, class_names = embed_dataset(
        model_path, args.max_per_class, args.batch_size, args.seed
    )
    print(f"特征矩阵: {feats.shape[0]} 样本 × {feats.shape[1]} 维")

    plot_embedding_2d(
        feats, labels, class_names, "tsne", out_dir / "feature_tsne.svg", args.seed
    )
    plot_embedding_2d(
        feats, labels, class_names, "pca", out_dir / "feature_pca.svg", args.seed
    )
    plot_class_separation(feats, labels, class_names, out_dir / "class_separation.svg")


if __name__ == "__main__":
    main()
