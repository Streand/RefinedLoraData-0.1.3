@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
python code_all\1_code_main_app\main_app.py %*