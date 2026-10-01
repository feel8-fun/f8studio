@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist env\python.exe goto install
if not exist .runtime-location goto install
set /p F8_PREVIOUS_ROOT=<.runtime-location
if "%F8_PREVIOUS_ROOT%"=="%CD%" exit /b 0
:install
mkdir .runtime-install-lock 2>nul
if errorlevel 1 (
  echo Runtime setup is already running. If interrupted, remove .runtime-install-lock and retry.
  exit /b 2
)
if exist .runtime-location del .runtime-location
if exist env rmdir /s /q env
offline\pixi-unpack.exe offline\base-runtime.tar --output-directory "%CD%" --shell cmd
if errorlevel 1 goto failed
env\python.exe -I -c "import f8studio_server"
if errorlevel 1 goto failed
> .runtime-location echo %CD%
rmdir .runtime-install-lock
exit /b 0
:failed
set "F8_EXIT_CODE=%ERRORLEVEL%"
rmdir .runtime-install-lock
exit /b %F8_EXIT_CODE%
