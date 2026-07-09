@echo off
setlocal

set "MARVIS_INSTALL_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%MARVIS_INSTALL_DIR%bin\Start-MARVIS.ps1" %*

exit /b %ERRORLEVEL%
