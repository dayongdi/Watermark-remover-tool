@echo off
rem Debug launcher: keeps console open to show errors
set "PY=C:\Users\32806\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python.exe"
"%PY%" "%~dp0app.py" %*
pause
