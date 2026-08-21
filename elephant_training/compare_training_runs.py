"""
对比两次训练的模型差异（例如：特征多点标注 VOC vs 仅大象主体框标注）。

用法示例：
  python compare_training_runs.py ^
    --model_feature runs/feature_best.pth ^
    --model_body runs/body_only_best.pth ^
    --test_dir dataset/test ^
    --out comparison_report.json

若不加 --test_dir，则只对比 checkpoint 内保存的验证准确率等元数据。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from classifier import ElephantClassifier

# 中国时区常用于报告落款（与用户环境一致）；显示用 UTC+8
_CN_TZ = timezone(timedelta(hours=8))


def _load_ckpt_meta(path: str) -> dict:
    import torch

    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        return {"path": str(ckpt_path), "note": "非 dict checkpoint，仅有 state_dict"}

    meta = {
        "path": str(ckpt_path.resolve()),
        "arch": ckpt.get("arch"),
        "val_acc": float(ckpt.get("val_acc", 0.0)) if ckpt.get("val_acc") is not None else None,
        "image_size": ckpt.get("image_size"),
        "class_names": ckpt.get("class_names"),
    }
    return meta


def _class_lists_match(names_a: list | None, names_b: list | None) -> bool:
    if not names_a or not names_b:
        return False
    return names_a == names_b


def _dual_eval_on_folder(
    clf_a: ElephantClassifier,
    clf_b: ElephantClassifier,
    test_path: Path,
    image_extensions: set[str],
) -> tuple[dict, dict, list[dict]]:
    """
    与 batch_test 一致：按类别子文件夹。
    每张图只对 A/B 各推理一次；同时统计准确率与预测分歧。
    """
    sta = {"total": 0, "correct": 0, "per_class": defaultdict(lambda: {"total": 0, "correct": 0})}
    stb = {"total": 0, "correct": 0, "per_class": defaultdict(lambda: {"total": 0, "correct": 0})}
    disagreements: list[dict] = []

    subdirs = sorted([d for d in test_path.iterdir() if d.is_dir()])
    if not subdirs:
        return sta, stb, disagreements

    for subdir in subdirs:
        true_label = subdir.name
        images = sorted([f for f in subdir.iterdir() if f.suffix in image_extensions])
        for img_path in images:
            pa, _, _ = clf_a.predict(str(img_path))
            pb, _, _ = clf_b.predict(str(img_path))

            sta["total"] += 1
            sta["per_class"][true_label]["total"] += 1
            if pa == true_label:
                sta["correct"] += 1
                sta["per_class"][true_label]["correct"] += 1

            stb["total"] += 1
            stb["per_class"][true_label]["total"] += 1
            if pb == true_label:
                stb["correct"] += 1
                stb["per_class"][true_label]["correct"] += 1

            if pa != pb:
                disagreements.append(
                    {
                        "image": str(img_path.resolve()),
                        "true_label": true_label,
                        "pred_feature": pa,
                        "pred_body": pb,
                    }
                )

    return sta, stb, disagreements


def _stats_to_accuracy(stats: dict) -> float:
    t = stats["total"]
    if t == 0:
        return 0.0
    return stats["correct"] / t * 100.0


def _find_user_checkpoints(project_root: Path) -> list[Path]:
    skip = ".torch_home"
    out: list[Path] = []
    for p in project_root.rglob("*.pth"):
        if skip in p.parts:
            continue
        out.append(p)
    return sorted(out)


def _meta_lines(meta: dict, title: str) -> list[str]:
    lines = [title, "-" * len(title)]
    lines.append(f"路径: {meta.get('path')}")
    lines.append(f"arch: {meta.get('arch')}")
    lines.append(f"image_size: {meta.get('image_size')}")
    va = meta.get("val_acc")
    if isinstance(va, (int, float)):
        va_s = f"{va:.4f}%"
    elif va is None:
        va_s = "（未记录或无此权重）"
    else:
        va_s = str(va)
    lines.append(f"checkpoint val_acc: {va_s}")
    cn = meta.get("class_names")
    if cn:
        lines.append(f"类别数: {len(cn)}")
        lines.append(f"类别列表: {', '.join(cn)}")
    return lines


def build_report_plaintext(report: dict, header_extra: list[str]) -> str:
    lines = [
        "=" * 56,
        "大象个体识别 · 两轮训练模型对比",
        "=" * 56,
        f"生成时间: {datetime.now(_CN_TZ).strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)",
        "",
        *header_extra,
        "",
    ]
    feat = report["checkpoint_meta"]["feature"]
    body = report["checkpoint_meta"]["body_only"]
    lines.extend(_meta_lines(feat, "【A】特征标注法（第一次训练权重）"))
    lines.append("")
    lines.extend(_meta_lines(body, "【B】仅大象主体标注（第二次训练权重）"))
    lines.append("")

    va_a = feat.get("val_acc")
    va_b = body.get("val_acc")
    if isinstance(va_a, (int, float)) and isinstance(va_b, (int, float)):
        lines.append(
            f"checkpoint val_acc 差值 (B − A): {va_b - va_a:+.4f}%"
        )
        lines.append("(若两轮验证集划分不一致，仅此作参考。)")
        lines.append("")

    te = report.get("test_evaluation")
    if te:
        lines.extend(
            [
                "=" * 56,
                f"统一测试集: {te['test_dir']}",
                "=" * 56,
                "",
                f"样本数: {te['overall']['n_images']}",
                f"A 总体准确率: {te['overall']['acc_feature_pct']:.4f}%",
                f"B 总体准确率: {te['overall']['acc_body_pct']:.4f}%",
                f"差值 (B − A): {te['overall']['delta_body_minus_feature_pct']:+.4f}%",
                "",
                "各类别:",
                f"{'类别':<14} {'A %':>9} {'B %':>9} {'B-A':>9}",
                "-" * 44,
            ]
        )
        for row in te["per_class"]:
            lines.append(
                f"{row['class']:<14} {row['acc_feature']:9.2f} {row['acc_body']:9.2f} "
                f"{row['delta']:+9.2f}"
            )
        lines.extend(
            [
                "",
                f"预测不一致样本总数: {te['disagreement_count']}",
            ]
        )
        sample = report.get("disagreements_sample") or []
        if sample:
            lines.extend(["", "分歧样例（前若干条）："])
            for i, item in enumerate(sample, 1):
                lines.append(
                    f"  [{i}] 真={item['true_label']} | A→{item['pred_feature']} | B→{item['pred_body']}"
                )
                lines.append(f"      {item['image']}")

    return "\n".join(lines)


def build_report_markdown(report: dict, header_extra: list[str]) -> str:
    intro_lines = [f"> {h}" for h in header_extra]
    md = [
        "# 大象个体识别 · 两轮训练模型对比",
        "",
        f"- 生成时间: {datetime.now(_CN_TZ).strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)",
        "",
        *intro_lines,
        "",
        "## 【A】特征标注法（第一次训练权重）",
        "",
    ]
    mf = report["checkpoint_meta"]["feature"]
    mb = report["checkpoint_meta"]["body_only"]
    cn_a = mf.get("class_names") or []
    cn_b = mb.get("class_names") or []
    va_a = mf.get("val_acc")
    va_b = mb.get("val_acc")

    def _fmt_va(v):
        return f"{v:.4f}%" if isinstance(v, (int, float)) else ("-" if v is None else str(v))

    md.extend(
        [
            "| 项 | 值 |",
            "| --- | --- |",
            f"| 路径 | `{mf.get('path')}` |",
            f"| arch | `{mf.get('arch')}` |",
            f"| image_size | `{mf.get('image_size')}` |",
            f"| checkpoint val_acc | {_fmt_va(va_a)} |",
            f"| 类别数 | {len(cn_a)} |",
            "",
            "## 【B】仅大象主体标注（第二次训练权重）",
            "",
            "| 项 | 值 |",
            "| --- | --- |",
            f"| 路径 | `{mb.get('path')}` |",
            f"| arch | `{mb.get('arch')}` |",
            f"| image_size | `{mb.get('image_size')}` |",
            f"| checkpoint val_acc | {_fmt_va(va_b)} |",
            f"| 类别数 | {len(cn_b)} |",
            "",
        ]
    )

    if isinstance(va_a, (int, float)) and isinstance(va_b, (int, float)):
        md.extend(
            [
                "### checkpoint val_acc 差值 (B − A)",
                "",
                f"**{va_b - va_a:+.4f}%**（若两轮验证划分不同，仅此作粗略参考）",
                "",
            ]
        )

    te = report.get("test_evaluation")
    if te:
        md.extend(
            [
                "## 统一测试集评估",
                "",
                f"- 测试目录: `{te['test_dir']}`",
                f"- 样本数: **{te['overall']['n_images']}**",
                f"- A 总体准确率: **{te['overall']['acc_feature_pct']:.4f}%**",
                f"- B 总体准确率: **{te['overall']['acc_body_pct']:.4f}%**",
                f"- 差值 (B − A): **{te['overall']['delta_body_minus_feature_pct']:+.4f}%**",
                "",
                "### 各类别准确率",
                "",
                "| 类别 | A % | B % | B − A |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in te["per_class"]:
            md.append(
                f"| {row['class']} | {row['acc_feature']:.2f} | {row['acc_body']:.2f} | "
                f"{row['delta']:+.2f} |"
            )
        md.extend(
            [
                "",
                f"### 预测不一致样本总数: **{te['disagreement_count']}**",
                "",
            ]
        )
        sample = report.get("disagreements_sample") or []
        if sample:
            md.append("### 分歧样例（前若干条）")
            md.append("")
            for i, item in enumerate(sample, 1):
                md.append(
                    f"{i}. 真实标签 `{item['true_label']}` → A `{item['pred_feature']}` / "
                    f"B `{item['pred_body']}` — `{item['image']}`"
                )
            md.append("")

    return "\n".join(md)


def _run_scan_project_and_write_reports(
    project_root: Path,
    report_txt: Path | None,
    report_md: Path | None,
) -> None:
    ckpts = _find_user_checkpoints(project_root)
    header = []

    if len(ckpts) == 0:
        header.append(
            "【扫描结果】在项目下未发现用户训练权重 (.pth，已排除 .torch_home)。"
            "请先完成训练并保存第二份 checkpoint，或使用 --model_feature / --model_body 指定路径。"
        )
        dummy = {"path": None, "arch": None, "val_acc": None, "image_size": None, "class_names": None}
        report = {
            "checkpoint_meta": {"feature": dummy, "body_only": dummy},
            "test_evaluation": None,
            "disagreements_sample": [],
        }
    elif len(ckpts) == 1:
        meta = _load_ckpt_meta(str(ckpts[0]))
        header.append(
            f"【扫描结果】发现 1 个权重文件: {ckpts[0].name}"
            "。无法进行「两轮训练」数值差异对比；下方沿用对比报告版式。"
            "【A】仅反映当前扫描到的 checkpoint；【B】为缺失的第二份，请另存另一轮训练的 .pth 后执行完整对比。"
            "若该文件实际是「主体标注」而非「特征标注」，在有两份权重时请在命令中用 --model_feature / --model_body 自行对应。"
        )
        dummy = {"path": "(无)", "arch": None, "val_acc": None, "image_size": None, "class_names": None}
        report = {
            "checkpoint_meta": {"feature": meta, "body_only": dummy},
            "test_evaluation": None,
            "disagreements_sample": [],
        }
    else:
        header.append(
            f"【扫描结果】发现 {len(ckpts)} 个 .pth 文件，但未指定哪一轮对应「特征标注」或「主体标注」。"
            "请手动运行：compare_training_runs.py --model_feature <路径A> --model_body <路径B>"
            f"\n已检测到的文件: " + "; ".join(str(p) for p in ckpts)
        )
        dummy = {"path": None, "arch": None, "val_acc": None, "image_size": None, "class_names": None}
        report = {
            "checkpoint_meta": {"feature": dummy, "body_only": dummy},
            "test_evaluation": None,
            "disagreements_sample": [],
        }

    txt = build_report_plaintext(report, header)
    md = build_report_markdown(report, header)

    def _maybe_write(fp: Path | None, content: str):
        if fp is None:
            return
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        print(f"已写入: {fp.resolve()}")

    defaults = []
    rp = project_root / "training_comparison_report.txt"
    rm = project_root / "training_comparison_report.md"
    if report_txt is None and report_md is None:
        _maybe_write(rp, txt)
        _maybe_write(rm, md)
        defaults = [rp, rm]
    else:
        _maybe_write(report_txt, txt)
        _maybe_write(report_md, md)
    print(txt)
    if defaults:
        print(f"\n（默认输出: {defaults[0].name}, {defaults[1].name}）")


def main():
    parser = argparse.ArgumentParser(description="对比两次训练模型的结果差异")
    parser.add_argument(
        "--scan_project",
        action="store_true",
        help="扫描项目内 .pth（排除 .torch_home），写入 txt/md；仅 1 个权重时会说明无法对比",
    )
    parser.add_argument("--project_root", type=str, default=".", help="与 --scan_project 配合")
    parser.add_argument("--model_feature", type=str, default="", help="第一次训练权重（特征标注法等）")
    parser.add_argument("--model_body", type=str, default="", help="第二次训练权重（仅主体标注）")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        help="统一测试根目录（子文件夹名为真实类别）；为空则跳过测试集评估",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="comparison_report.json",
        help="详细结果 JSON",
    )
    parser.add_argument(
        "--limit_disagreements",
        type=int,
        default=50,
        help="报告中最多记录的预测分歧样本数（两边预测不同）",
    )
    parser.add_argument(
        "--report_txt",
        type=str,
        default="",
        help="可读文本报告路径（为空则沿用 scan 默认值或仅用 JSON）",
    )
    parser.add_argument(
        "--report_md",
        type=str,
        default="",
        help="Markdown 报告路径",
    )
    args = parser.parse_args()

    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

    if args.scan_project:
        root = Path(args.project_root).resolve()
        rt = Path(args.report_txt) if args.report_txt.strip() else None
        rm = Path(args.report_md) if args.report_md.strip() else None
        _run_scan_project_and_write_reports(root, rt, rm)
        return

    if not args.model_feature.strip() or not args.model_body.strip():
        parser.error(
            "请指定 --model_feature 与 --model_body，或使用 --scan_project 扫描当前项目"
        )

    mf = Path(args.model_feature).resolve()
    mb = Path(args.model_body).resolve()
    if mf == mb:
        parser.error(
            "两个参数指向同一文件，无法进行有意义对比。"
            "请将另一轮训练权重保存为单独 .pth 后再运行。"
        )

    meta_feat = _load_ckpt_meta(args.model_feature)
    meta_body = _load_ckpt_meta(args.model_body)

    print("=" * 56)
    print("两次训练模型对比")
    print("=" * 56)
    print("\n【A】特征标注法（或第一次训练）")
    print(f"  路径: {meta_feat.get('path')}")
    print(f"  arch: {meta_feat.get('arch')} | image_size: {meta_feat.get('image_size')}")
    if meta_feat.get("val_acc") is not None:
        print(f"  checkpoint 内验证准确率 val_acc: {meta_feat['val_acc']:.2f}%")

    print("\n【B】仅大象主体标注（或第二次训练）")
    print(f"  路径: {meta_body.get('path')}")
    print(f"  arch: {meta_body.get('arch')} | image_size: {meta_body.get('image_size')}")
    if meta_body.get("val_acc") is not None:
        print(f"  checkpoint 内验证准确率 val_acc: {meta_body['val_acc']:.2f}%")

    va_a = meta_feat.get("val_acc")
    va_b = meta_body.get("val_acc")
    if va_a is not None and va_b is not None:
        delta = va_b - va_a
        sign = "+" if delta >= 0 else ""
        print(f"\ncheckpoint val_acc 差值 (B - A): {sign}{delta:.2f}%")
        print("说明：两轮若验证划分或数据分布不同，此差值仅能作粗略参考。")

    report: dict = {
        "checkpoint_meta": {"feature": meta_feat, "body_only": meta_body},
        "test_evaluation": None,
        "disagreements_sample": [],
    }

    test_root = Path(args.test_dir.strip()) if args.test_dir.strip() else None
    if test_root and test_root.is_dir():
        clf_a = ElephantClassifier(args.model_feature, "class_names.json")
        clf_b = ElephantClassifier(args.model_body, "class_names.json")
        names_a = clf_a.class_names
        names_b = clf_b.class_names
        if not _class_lists_match(names_a, names_b):
            print("\n警告: 两份权重内 class_names 不一致，统一测试集的数值对比可能无意义。")
        else:
            print("\n类别顺序一致，可进行同分布测试。")

        st_a, st_b, disagreements = _dual_eval_on_folder(clf_a, clf_b, test_root, image_ext)
        acc_a = _stats_to_accuracy(st_a)
        acc_b = _stats_to_accuracy(st_b)

        print("\n" + "=" * 56)
        print(f"统一测试集: {test_root.resolve()}")
        print("=" * 56)
        if st_a["total"] == 0:
            print("未在测试目录下找到按类别子文件夹的图片，请检查目录结构。")
        else:
            print(f"\n总体准确率  A(特征/第一次): {acc_a:.2f}% ({st_a['correct']}/{st_a['total']})")
            print(f"总体准确率  B(主体/第二次): {acc_b:.2f}% ({st_b['correct']}/{st_b['total']})")
            d = acc_b - acc_a
            print(f"差值 (B - A): {'+' if d >= 0 else ''}{d:.2f}%")

            all_classes = sorted(set(st_a["per_class"].keys()) | set(st_b["per_class"].keys()))
            print("\n各类别准确率对比:")
            print("-" * 56)
            print(f"{'类别':<16} {'A %':>8} {'B %':>8} {'B-A':>8}")
            per_class_rows = []
            for c in all_classes:
                ta = st_a["per_class"][c]["total"]
                tb = st_b["per_class"][c]["total"]
                ca = (
                    st_a["per_class"][c]["correct"] / ta * 100.0 if ta else 0.0
                )
                cb = (
                    st_b["per_class"][c]["correct"] / tb * 100.0 if tb else 0.0
                )
                row = {"class": c, "acc_feature": round(ca, 4), "acc_body": round(cb, 4)}
                row["delta"] = round(cb - ca, 4)
                per_class_rows.append(row)
                print(f"{c:<16} {ca:8.2f} {cb:8.2f} {cb - ca:+8.2f}")

            sample = disagreements[: max(0, args.limit_disagreements)]
            print(f"\n预测不一致样本数: {len(disagreements)} （展示前 {len(sample)} 条）")

            report["test_evaluation"] = {
                "test_dir": str(test_root.resolve()),
                "overall": {
                    "acc_feature_pct": round(acc_a, 4),
                    "acc_body_pct": round(acc_b, 4),
                    "delta_body_minus_feature_pct": round(acc_b - acc_a, 4),
                    "n_images": st_a["total"],
                },
                "per_class": per_class_rows,
                "disagreement_count": len(disagreements),
            }
            report["disagreements_sample"] = sample
    elif args.test_dir.strip():
        print(f"\n未找到测试目录: {args.test_dir}，已跳过测试评估。")

    header_extra = [
        f"权重 A: {args.model_feature}",
        f"权重 B: {args.model_body}",
    ]
    if args.test_dir.strip():
        header_extra.append(f"测试集: {args.test_dir}")

    txt_body = build_report_plaintext(report, header_extra)
    md_body = build_report_markdown(report, header_extra)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已写入: {out_path.resolve()}")

    rt = Path(args.report_txt.strip()) if args.report_txt.strip() else None
    rm = Path(args.report_md.strip()) if args.report_md.strip() else None
    if rt is None and rm is None:
        base = Path(args.out)
        stem = base.stem + "_readable"
        rt = base.with_name(stem + ".txt")
        rm = base.with_name(stem + ".md")

    if rt:
        rt.write_text(txt_body, encoding="utf-8")
        print(f"可读报告 (txt): {rt.resolve()}")
    if rm:
        rm.write_text(md_body, encoding="utf-8")
        print(f"可读报告 (md): {rm.resolve()}")


if __name__ == "__main__":
    main()
