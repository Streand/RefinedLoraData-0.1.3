"""
Quick test to verify the corrected path logic
"""
import os
from pathlib import Path

# Test the new path logic
current_file_dir = Path(__file__).parent  # 6_code_camera directory
project_root = current_file_dir.parent.parent  # Go up two levels to project root
base_dir = project_root / "data_storage" / "data_store_camera"

print(f"Current file directory: {current_file_dir}")
print(f"Project root: {project_root}")
print(f"Base directory: {base_dir}")
print(f"Base directory exists: {base_dir.exists()}")
print(f"Base directory (absolute): {base_dir.absolute()}")

# Test creating a folder
test_dir = base_dir / "path_test"
test_dir.mkdir(exist_ok=True)
print(f"Test directory created: {test_dir.exists()}")
print(f"Test directory (absolute): {test_dir.absolute()}")

# Clean up
if test_dir.exists():
    test_dir.rmdir()
    print("Test directory cleaned up")
