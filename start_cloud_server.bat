@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Elephant cloud API - run on PC, Pi connects to this machine
REM Edit API key below (must match pi_cloud_config.sh on Pi)

set CLOUD_API_KEY=elephant-demo-2026
set CLOUD_PORT=8000

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found. Run install_cloud_deps.bat first.
    pause
    exit /b 1
)

if not exist "best_elephant_model.pth" (
    echo ERROR: best_elephant_model.pth not found in project folder.
    pause
    exit /b 1
)

REM Port 8000 already in use? Often a previous cloud_server is still running.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%CLOUD_PORT% " ^| findstr "LISTENING"') do (
    echo.
    echo WARNING: Port %CLOUD_PORT% is already in use by PID %%a
    echo   If you already started cloud_server, just use that window.
    echo   Local test: curl http://127.0.0.1:%CLOUD_PORT%/health
    echo   To free the port: taskkill /PID %%a /F
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo.
echo ========================================
echo  Elephant Cloud Server
echo  Port: %CLOUD_PORT%
echo  API Key: %CLOUD_API_KEY%
echo ========================================
echo  Pi pi_cloud_config.sh should use ONE of these IPv4:
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /c:"IPv4"') do (
  for /f "tokens=* delims= " %%j in ("%%i") do echo    http://%%j:%CLOUD_PORT%
)
echo  Local test: curl http://127.0.0.1:%CLOUD_PORT%/health
echo  Press Ctrl+C to stop
echo ========================================
echo.

python cloud_server.py --host 0.0.0.0 --port %CLOUD_PORT% --yolo-weights yolov8m.pt --yolo-imgsz 640 --infer-max-width 1280 --recog-interval 2

pause
