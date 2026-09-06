@echo off
title CET-6 Practice
echo ============================================
echo   CET-6 Practice System - Starting
echo   Generation model: DeepSeek cloud
echo   TTS + pronunciation models preload to GPU
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] Checking server status...
curl -s --max-time 2 http://127.0.0.1:8123/ >nul 2>&1
if errorlevel 1 (
    echo Server not running, starting a new one...
    start "CET6-server" /min python server.py
) else (
    echo Server is already running, skip starting.
)

echo [2/2] Waiting for server ready...
set ready=0
for /l %%i in (1,1,20) do (
    curl -s --max-time 2 http://127.0.0.1:8123/ >nul 2>&1
    if not errorlevel 1 (
        set ready=1
        goto :open
    )
    timeout /t 1 /nobreak >nul
)
:open
if "%ready%"=="1" (
    echo Server ready, opening browser...
    echo Note: TTS + scoring models are preloading to GPU in background - about 1 min.
    start http://127.0.0.1:8123
) else (
    echo [FAILED] Server start timeout. Check python environment.
    pause
    exit /b 1
)
echo.
echo Server is running in a minimized window.
echo Stop: kill the python process in Task Manager.
echo.
pause