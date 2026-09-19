@echo off
chcp 65001 >nul
title Khởi chạy ToolRecap V2

if exist "%~dp0release\ToolRecapV2\ToolRecapV2.exe" (
    start "" "%~dp0release\ToolRecapV2\ToolRecapV2.exe"
    exit /b 0
)

if exist "%~dp0ToolRecapV2.exe" (
    start "" "%~dp0ToolRecapV2.exe"
    exit /b 0
)

echo Đang khởi chạy bằng môi trường Python...
python "%~dp0auto_main.py"
