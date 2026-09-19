@echo off
chcp 65001 >nul
title Đóng gói ToolRecap V2 Portable
echo Đang tiến hành đóng gói ToolRecap V2 thành bản portable...
python build_exe.py
if errorlevel 1 (
    echo.
    echo Đóng gói thất bại. Vui lòng kiểm tra thông báo lỗi ở trên.
    pause
    exit /b 1
)
echo.
echo Đóng gói hoàn tất thành công! Bản dựng nằm trong thư mục release\ToolRecapV2
pause
exit /b 0
