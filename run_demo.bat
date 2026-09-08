@echo off
setlocal
cd /d "%~dp0"

title J.A.R.V.I.S CAN IDS - Offline Demo

echo ==============================================================
echo  J.A.R.V.I.S CAN IDS - OFFLINE JUDGING DEMO
echo ==============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    echo.
    echo Install Python and run:
    echo     pip install -r requirements_demo.txt
    echo.
    pause
    exit /b 1
)

echo [1/2] Running offline preflight...
echo.

python dashboard\preflight.py
if errorlevel 1 (
    echo.
    echo [ERROR] Preflight failed. Demo was NOT started.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting J.A.R.V.I.S dashboard...
echo.
echo URL: http://127.0.0.1:5000
echo.
echo Keep this window open during judging.
echo Press Ctrl+C here to stop the demo.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command ^
    "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:5000'"

python dashboard\app.py

echo.
echo J.A.R.V.I.S dashboard stopped.
pause
endlocal
