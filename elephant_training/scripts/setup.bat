@echo off
cd /d "%~dp0.."
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -U pip
pip install -r requirements.txt
echo.
echo 环境就绪。请激活虚拟环境后运行: python train.py
pause
