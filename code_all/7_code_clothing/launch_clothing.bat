@echo off
echo Starting RefinedLoraData - Clothing Analysis Module
echo =====================================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install -r requirements_clothing.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Launching Clothing Analysis UI...
echo Open your browser to the URL that appears below
echo Press Ctrl+C to stop the server
echo.

python launch_clothing.py

pause
