@echo off
rem 系统默认代码页(GBK)
title CET-6 综合练习系统
echo ============================================
echo   CET-6 综合练习系统 · 启动
echo   只启动 Web 服务，不启动本地模型
echo ============================================
echo.

rem 检查 Qwen 本地模型是否已运行 (仅提示，绝不自动启动)
curl -s --max-time 3 http://127.0.0.1:8081/v1/models >nul 2>&1
if errorlevel 1 (
    echo [提示] 未检测到 Qwen 模型 ^(127.0.0.1:8081^)
    echo         请先自行启动本地模型，例如：
    echo         D:\llama.cpp\switch-qwen.bat
    echo         或  powershell -File D:\llama.cpp\start-qwen-A.ps1
    echo         模型未启动时，生成功能会失败。
    echo.
    echo 按任意键继续启动服务...
    pause >nul
)

cd /d "%~dp0"

echo [1/2] 启动服务中...
start "CET6-server" /min python server.py

echo [2/2] 等待服务就绪...
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
    echo 服务已就绪，打开浏览器...
    start http://127.0.0.1:8123
) else (
    echo [失败] 服务启动超时，请检查 python 环境。
    pause
    exit /b 1
)
echo.
echo 服务在后台运行（最小化窗口）。
echo 停止服务：任务管理器结束 python 进程。
echo.
pause
