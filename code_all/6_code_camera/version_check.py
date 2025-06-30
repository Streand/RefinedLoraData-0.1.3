"""
Version check script to verify the UI is using the latest path fix
"""
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import the UI module
sys.path.append(os.getcwd())

print("🔍 Version Check - Path Fix Verification\n")

def check_ui_version():
    """Check which version of the UI code is being used"""
    
    # Read the UI_camera.py file to check the path construction
    ui_file_path = Path("UI_camera.py")
    
    if not ui_file_path.exists():
        print("❌ UI_camera.py not found!")
        return False
    
    with open(ui_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the new path construction method
    if 'current_file_dir = Path(__file__).parent' in content:
        print("✅ LATEST VERSION: Using absolute path construction")
        print("   - Path construction: current_file_dir = Path(__file__).parent")
        print("   - This version should work correctly")
        return True
    elif 'Path("../../data_storage/data_store_camera")' in content:
        print("⚠️  OLDER VERSION: Using relative path construction")
        print("   - Path construction: Path('../../data_storage/data_store_camera')")
        print("   - This version may have path issues")
        return False
    elif 'Path("data_storage/data_store_camera")' in content:
        print("❌ OLD VERSION: Using incorrect relative path")
        print("   - Path construction: Path('data_storage/data_store_camera')")
        print("   - This version will create folders in wrong location")
        return False
    else:
        print("❓ UNKNOWN VERSION: Could not detect path construction method")
        return False

def test_path_construction():
    """Test the actual path construction"""
    print("\n🧪 Testing path construction:")
    
    try:
        from UI_camera import CameraUI
        ui = CameraUI()
        
        # Create a test to see what path would be constructed
        # We'll inspect the save_batch_results method indirectly
        current_file_dir = Path(__file__).parent
        project_root = current_file_dir.parent.parent
        expected_base_dir = project_root / "data_storage" / "data_store_camera"
        
        print(f"Expected base directory: {expected_base_dir}")
        print(f"Expected base directory exists: {expected_base_dir.exists()}")
        
        # Test with mock data
        mock_results = []
        test_result = ui.save_batch_results(mock_results, "test_version_check")
        
        if "No processed results to save" in test_result[0]:
            print("✅ UI loaded successfully (expected 'no results' message)")
            return True
        else:
            print(f"❓ Unexpected result: {test_result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing UI: {e}")
        return False

def main():
    print("Running version check...\n")
    
    # Check the file version
    file_version_ok = check_ui_version()
    
    # Test the actual import and functionality
    import_test_ok = test_path_construction()
    
    print(f"\n📊 Results:")
    print(f"File version check: {'✅ PASS' if file_version_ok else '❌ FAIL'}")
    print(f"Import/function test: {'✅ PASS' if import_test_ok else '❌ FAIL'}")
    
    if file_version_ok and import_test_ok:
        print(f"\n🎉 All checks passed! You're running the latest version.")
        print(f"If you're still having issues, try:")
        print(f"1. Restart the Gradio UI completely")
        print(f"2. Clear any browser cache")
        print(f"3. Check that you're looking in the correct folder:")
        print(f"   {Path(__file__).parent.parent.parent / 'data_storage' / 'data_store_camera'}")
    else:
        print(f"\n⚠️  Issues detected. The UI may not be using the latest version.")

if __name__ == "__main__":
    main()
