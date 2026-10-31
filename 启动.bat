@echo off
rem Watermark Remover launcher
rem Auto-detects a Python that actually has the required packages,
rem so it works on other machines (64-bit or 32-bit Windows).
setlocal

set "PY="
call :check "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :check "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
call :check "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :check "%%~D\python.exe"
if not defined PY for /d %%D in ("C:\Python3*") do call :check "%%~D\python.exe"
if not defined PY call :check "python"

if not defined PY (
  echo.
  echo [ERROR] No Python found that has numpy / opencv / pillow installed.
  echo Install Python 3.10 or newer, then run:
  echo     pip install numpy opencv-python pillow onnxruntime tkinterdnd2
  echo.
  pause
  exit /b 1
)

rem prefer pythonw.exe (no console window)
set "LAUNCH=%PY%"
for %%F in ("%PY%") do (
  if not "%%~dpF"=="" if exist "%%~dpFpythonw.exe" set "LAUNCH=%%~dpFpythonw.exe"
)

start "" "%LAUNCH%" "%~dp0app.py" %*
exit /b 0

:check
if defined PY exit /b 0
if not exist "%~1" if not "%~1"=="python" exit /b 0
"%~1" -c "import cv2, numpy, PIL" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0
