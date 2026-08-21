"""兼容入口：训练代码已整合至 elephant_training/ 目录。"""
import os
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TRAIN_DIR = _ROOT / "elephant_training"
_SCRIPT = _TRAIN_DIR / "train.py"

if __name__ == "__main__":
    if "ELEPHANT_DATASET" not in os.environ and (_ROOT / "dataset").is_dir():
        os.environ["ELEPHANT_DATASET"] = str(_ROOT / "dataset")
    if "ELEPHANT_MODEL" not in os.environ and (_ROOT / "best_elephant_model.pth").exists():
        os.environ.setdefault("ELEPHANT_MODEL", str(_ROOT / "best_elephant_model.pth"))
    sys.path.insert(0, str(_TRAIN_DIR))
    runpy.run_path(str(_SCRIPT), run_name="__main__")
