@echo off
REM PyTorch Blackwell GPU Support Installer for Windows
REM This script installs the optimal PyTorch version for NVIDIA RTX 5000 series GPUs

echo ===============================================================
echo NVIDIA RTX 5000 Series (Blackwell) PyTorch Installer
echo ===============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10-3.12 first
    pause
    exit /b 1
)

REM Check for virtual environment
echo Checking Python environment...
python -c "import sys; print('Virtual environment detected' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No virtual environment detected')"

REM Check for project venv
if exist "venv\Scripts\activate.bat" (
    echo.
    echo Found project virtual environment: venv\
    set /p activate_venv="Do you want to activate it? (y/N): "
    if /i "%activate_venv%"=="y" (
        echo Activating virtual environment...
        call venv\Scripts\activate.bat
        echo Virtual environment activated.
    )
) else if exist ".venv\Scripts\activate.bat" (
    echo.
    echo Found project virtual environment: .venv\
    set /p activate_venv="Do you want to activate it? (y/N): "
    if /i "%activate_venv%"=="y" (
        echo Activating virtual environment...
        call .venv\Scripts\activate.bat
        echo Virtual environment activated.
    )
)

echo.

REM Show options
echo Available options:
echo   1. Install/upgrade PyTorch for Blackwell support
echo   2. Check current system status (dry run)
echo   3. Force reinstall (even if already supported)
echo   4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Installing PyTorch with Blackwell support...
    python install_pytorch_blackwell.py
) else if "%choice%"=="2" (
    echo.
    echo Checking system status (dry run)...
    python install_pytorch_blackwell.py --dry-run
) else if "%choice%"=="3" (
    echo.
    echo Force reinstalling PyTorch...
    python install_pytorch_blackwell.py --force
) else if "%choice%"=="4" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

echo.
echo ===============================================================
echo Installation process completed!
echo ===============================================================
echo.
echo You may need to restart your application to use the new PyTorch version.
pause
