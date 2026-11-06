@echo off
cd /d "%~dp0"
set "PYTHON=%USERPROFILE%\venvs\qudor-ai\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Python environment not found. Follow the setup instructions in README.md.
    pause
    exit /b 1
)
"%PYTHON%" app.py
if errorlevel 1 pause
