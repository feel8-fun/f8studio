@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed
where pixi >nul 2>&1
if not errorlevel 1 goto launch
if not defined PIXI_HOME set "PIXI_HOME=%USERPROFILE%\.pixi"
set "PATH=%PIXI_HOME%\bin;%PATH%"
where pixi >nul 2>&1
if not errorlevel 1 goto launch
echo Installing Pixi from https://pixi.sh ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $env:PIXI_NO_PATH_UPDATE = '1'; Invoke-RestMethod -Uri 'https://pixi.sh/install.ps1' | Invoke-Expression"
if errorlevel 1 goto failed
where pixi >nul 2>&1
if errorlevel 1 goto missing_pixi

:launch
call install_env.bat
if errorlevel 1 goto failed
pixi run --locked -e studio-runtime studio_launch %*
if errorlevel 1 goto failed
exit /b 0

:failed
set "F8_EXIT_CODE=%ERRORLEVEL%"
echo F8Studio failed. See the error above.
pause
exit /b %F8_EXIT_CODE%

:missing_pixi
echo Pixi installer finished but pixi was not found in "%PIXI_HOME%\bin".
echo Install it manually from https://pixi.sh and run F8Studio again.
pause
exit /b 2
