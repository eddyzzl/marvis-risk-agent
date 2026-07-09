@echo off
setlocal

set "MARVIS_INSTALL_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%MARVIS_INSTALL_DIR%bin\Start-MARVIS.ps1" %*
set "MARVIS_EXIT_CODE=%ERRORLEVEL%"

if not "%MARVIS_EXIT_CODE%"=="0" (
  echo.
  echo MARVIS-Agent failed to start. Review the message above and logs under:
  echo %LOCALAPPDATA%\MARVIS-Agent\logs
  if /I not "%CI%"=="true" pause
)

exit /b %MARVIS_EXIT_CODE%
