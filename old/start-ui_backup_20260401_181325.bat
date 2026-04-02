@echo off
setlocal
title Z-Image Local UI
cd /d "%~dp0"

py -3 "%~dp0zimage_ui.py"
if %errorlevel% equ 0 goto :end

python "%~dp0zimage_ui.py"
if %errorlevel% equ 0 goto :end

echo.
echo [ERROR] Could not start the UI.
echo Tried:  py -3 zimage_ui.py   then   python zimage_ui.py
echo Install Python 3 and ensure it is on PATH ^(or use the "py" launcher^).
echo.
pause
exit /b 1

:end
endlocal
exit /b 0
