#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
echo ""
echo "环境就绪。请运行: source .venv/bin/activate && python train.py"
