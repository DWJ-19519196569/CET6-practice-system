@echo off
title CET-6 Practice
echo ============================================
echo   CET-6 Practice System - Starting
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] Checking server status...
curl -sk --max-time 2 https://127.0.0.1:8123/ >nul 2>&1
if errorlevel 1 (
    echo Server not running, starting a new one...
    start "CET6-server" /min python server.py
) else (
    echo Server is already running, skip starting.
)

echo [2/2] Waiting for server ready...
for /l %%i in (1,1,20) do (
    curl -sk --max-time 2 https://127.0.0.1:8123/ >nul 2>&1
    if not errorlevel 1 goto :open
    timeout /t 1 /nobreak >nul
)
echo [FAILED] Server start timeout. Check python environment.
pause
exit /b 1

:open
echo Server ready.
echo Note: TTS + scoring models preloading to GPU in background - about 1 min.
echo.
echo ============================================
if exist "access_urls.txt" (
    type "access_urls.txt"
) else (
    echo     [local] https://127.0.0.1:8123
)
echo   First phone visit: tap Advanced / Proceed to trust the certificate.
echo ============================================
start https://127.0.0.1:8123
echo.
echo Server is running in a minimized window.
echo Stop: Task Manager - end the python process.
echo.
pause
