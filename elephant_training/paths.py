"""训练包路径与环境（解压后在本目录运行即可）。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("ELEPHANT_DATASET", ROOT / "dataset"))
MODEL_PATH = Path(os.environ.get("ELEPHANT_MODEL", ROOT / "best_elephant_model.pth"))
CLASS_NAMES_PATH = Path(os.environ.get("ELEPHANT_CLASS_NAMES", ROOT / "class_names.json"))
REPORTS_DIR = Path(os.environ.get("ELEPHANT_REPORTS", ROOT / "reports"))
TORCH_HOME = ROOT / ".torch_home"


def ensure_torch_home() -> None:
    if "TORCH_HOME" not in os.environ:
        TORCH_HOME.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(TORCH_HOME)
