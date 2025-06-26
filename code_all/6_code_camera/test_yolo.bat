@echo off
echo Testing YOLO Camera Detection for Blackwell GPU
echo ===============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Running YOLO compatibility test...
echo.

REM Run the test with error handling
python test_yolo_camera.py
if errorlevel 1 (
    echo.
    echo Test completed with some issues. Check output above.
) else (
    echo.
    echo Test completed successfully!
)

echo.
echo Press any key to continue...
pause >nul
