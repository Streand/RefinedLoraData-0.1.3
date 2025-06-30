"""
Test the main UI path construction
"""
import os
from pathlib import Path

# Test path construction from main UI directory
current_file_dir = Path(__file__).parent  # 1_code_main_app directory  
project_root = current_file_dir.parent.parent  # Go up two levels to project root
base_dir = project_root / "data_storage" / "data_store_camera"

print(f"Testing path construction from main UI context:")
print(f"Current file directory: {current_file_dir}")
print(f"Project root: {project_root}")
print(f"Base directory: {base_dir}")
print(f"Base directory exists: {base_dir.exists()}")
print(f"Base directory (absolute): {base_dir.absolute()}")

# Test creating a test folder
test_dir = base_dir / "main_ui_test"
test_dir.mkdir(exist_ok=True)
print(f"Test directory created: {test_dir.exists()}")

# Clean up
if test_dir.exists():
    test_dir.rmdir()
    print("Test directory cleaned up")
