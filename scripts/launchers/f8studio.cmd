@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
call install_env.bat
if errorlevel 1 exit /b %ERRORLEVEL%
call activate.bat
if errorlevel 1 exit /b %ERRORLEVEL%
set "F8_SERVICE_INDEX=%CD%\config\service-index.json"
set "F8_MODEL_ROOT=%CD%\resources\models"
env\python.exe -I -m f8studio_server --tray --open-browser %*
exit /b %ERRORLEVEL%
