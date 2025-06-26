@echo off
echo ========================================
echo Testing Camera Backend Integration
echo ========================================

echo.
echo Activating virtual environment...
call ..\..\.venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo Virtual environment not found, continuing without activation...
)

echo.
echo Current directory: %CD%
echo Python path: 
python -c "import sys; print('\n'.join(sys.path[:5]))"

echo.
echo Running comprehensive integration tests...
python test_integration_full.py

echo.
echo ========================================
echo Test completed! Check output above for results.
echo ========================================
pause
