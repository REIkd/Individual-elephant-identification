"""VOC 标注象名别名（文件夹 canonical 名 ↔ XML 内常见拼写差异）。"""

from __future__ import annotations

# canonical 文件夹象名 -> XML 中可能出现的等价写法
ELEPHANT_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "玛丽亚": ("玛利亚",),
}


def name_tags_for_elephant(canonical: str) -> tuple[str, ...]:
    extras = ELEPHANT_NAME_ALIASES.get(canonical, ())
    return (canonical, *extras)


def feature_prefixes_for_elephant(canonical: str) -> tuple[str, ...]:
    return tuple(f"{tag}-" for tag in name_tags_for_elephant(canonical))


def is_body_tag(tag: str, canonical: str) -> bool:
    return tag in name_tags_for_elephant(canonical) or tag == "大象"


def is_feature_tag(tag: str, canonical: str) -> bool:
    return any(tag.startswith(prefix) for prefix in feature_prefixes_for_elephant(canonical))


def normalize_tag_to_canonical(tag: str, canonical: str) -> str:
    """将 XML 内别名标签规范为 canonical（用于批量修正标注）。"""
    tag = (tag or "").strip()
    if not tag:
        return tag
    for alias in name_tags_for_elephant(canonical):
        if alias == canonical:
            continue
        if tag == alias:
            return canonical
        prefix = f"{alias}-"
        if tag.startswith(prefix):
            return canonical + tag[len(alias) :]
    return tag
