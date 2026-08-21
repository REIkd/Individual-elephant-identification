"""
批量修正玛丽亚文件夹 VOC 标注：XML 内「玛利亚」→ canonical「玛丽亚」。

用法:
  python fix_maria_voc_tags.py --dry-run
  python fix_maria_voc_tags.py
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from paths import DATASET_DIR
from voc_name_aliases import normalize_tag_to_canonical


def elephant_name_from_folder(folder_name: str) -> str:
    name = folder_name.strip()
    if "（" in name:
        name = name.split("（", 1)[0].strip()
    if "(" in name:
        name = name.split("(", 1)[0].strip()
    return name

CANONICAL = "玛丽亚"
WRONG_BODY = "玛利亚"
WRONG_PREFIX = "玛利亚-"


def fix_xml_file(path: Path, *, dry_run: bool) -> int:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return 0
    root = tree.getroot()
    changed = 0
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or name_el.text is None:
            continue
        old = name_el.text.strip()
        new = normalize_tag_to_canonical(old, CANONICAL)
        if new != old:
            changed += 1
            if not dry_run:
                name_el.text = new
    if changed and not dry_run:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="修正玛丽亚 VOC 标注拼写")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="dataset 根目录（默认 ELEPHANT_DATASET 或 elephant_training/dataset）",
    )
    args = parser.parse_args()

    root = Path(args.dataset) if args.dataset.strip() else DATASET_DIR
    maria_dir = None
    for folder in root.iterdir():
        if folder.is_dir() and elephant_name_from_folder(folder.name) == CANONICAL:
            maria_dir = folder
            break
    if maria_dir is None:
        raise SystemExit(f"未找到 {CANONICAL} 文件夹: {root}")

    total_files = 0
    total_tags = 0
    for xml_path in sorted(maria_dir.glob("*.xml")):
        n = fix_xml_file(xml_path, dry_run=args.dry_run)
        if n:
            total_files += 1
            total_tags += n

    mode = "DRY-RUN" if args.dry_run else "DONE"
    print(f"[{mode}] folder={maria_dir.name}")
    print(f"  XML files changed: {total_files}")
    print(f"  name tags fixed: {total_tags}")
    print(f"  ({WRONG_BODY} / {WRONG_PREFIX}* -> {CANONICAL} / {CANONICAL}-*)")


if __name__ == "__main__":
    main()
