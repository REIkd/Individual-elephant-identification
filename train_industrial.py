"""兼容入口：训练代码已整合至 elephant_training/ 目录。"""
import runpy
import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent / "elephant_training"
_SCRIPT = _TRAIN_DIR / "train_industrial.py"

if __name__ == "__main__":
    sys.path.insert(0, str(_TRAIN_DIR))
    runpy.run_path(str(_SCRIPT), run_name="__main__")
