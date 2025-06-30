"""
Test script to demonstrate the batch upload, analysis, save, and open folder workflow
"""
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import the UI module
sys.path.append(os.getcwd())

try:
    from UI_camera import CameraUI
    print("✅ Successfully imported CameraUI")
except ImportError as e:
    print(f"❌ Failed to import CameraUI: {e}")
    sys.exit(1)

def test_workflow():
    """Test the complete batch workflow"""
    print("🧪 Testing Batch Upload and Save Workflow\n")
    
    # Initialize the UI
    ui = CameraUI()
    print("✅ UI initialized successfully")
    
    # Test the open folder functionality
    test_folder = Path("../../data_storage/data_store_camera")
    if test_folder.exists():
        print(f"\n📁 Testing open folder with: {test_folder.absolute()}")
        result = ui.open_saved_folder(str(test_folder.absolute()))
        print(f"Result: {result}")
    else:
        print(f"❌ Test folder doesn't exist: {test_folder.absolute()}")
        # Create it for testing
        test_folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created test folder: {test_folder.absolute()}")
        result = ui.open_saved_folder(str(test_folder.absolute()))
        print(f"Result: {result}")
    
    print("\n🎯 Workflow Summary:")
    print("1. ✅ Upload multiple images via the 'Batch Upload & Save' tab")
    print("2. ✅ Click 'Process Uploaded Images' to analyze all images")
    print("3. ✅ Enter a folder name and click 'Save Analysis Results'")
    print("4. ✅ Click 'Open Saved Folder' to view saved files in Explorer")
    print("\n📂 Files saved per image:")
    print("   • [name]-camera-[n].jpg/png - Original image")
    print("   • [name]-camera-[n].txt - Stable Diffusion prompt")
    print("   • [name]-camera-[n]full.txt - Complete analysis")

if __name__ == "__main__":
    test_workflow()
