"""
Test script to verify the main UI loads without errors
"""

import sys
import os

# Add the main app path
main_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '1_code_main_app'))
sys.path.append(main_app_path)

def test_ui_import():
    """Test that UI_main can be imported"""
    print("🧪 Testing UI_main.py import...")
    
    try:
        from UI_main import launch_ui, create_main_ui
        print("✅ UI_main.py imported successfully")
        
        # Test creating the UI (without launching)
        print("🧪 Testing UI creation...")
        ui = create_main_ui()
        print("✅ UI created successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_camera_backend():
    """Test that camera backend can be imported from main UI context"""
    print("\n🧪 Testing camera backend access from main UI...")
    
    try:
        # Simulate the path addition that happens in UI_main
        backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '6_code_camera'))
        if backend_path not in sys.path:
            sys.path.append(backend_path)
        
        from backend_camera import CameraAnalyzer
        print("✅ Camera backend accessible from main UI context")
        
        # Test creating analyzer
        analyzer = CameraAnalyzer()
        print("✅ Camera analyzer created successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Camera backend import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Camera backend error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Main Application Integration Tests")
    print("=" * 40)
    
    ui_ok = test_ui_import()
    camera_ok = test_camera_backend()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"Main UI Import: {'✅ PASS' if ui_ok else '❌ FAIL'}")
    print(f"Camera Backend: {'✅ PASS' if camera_ok else '❌ FAIL'}")
    
    if ui_ok and camera_ok:
        print("\n🎉 All integration tests passed!")
        print("✅ Main application is ready to run")
        print("✅ Camera analysis is fully integrated")
        print("\n💡 You can now run: python main_app.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
