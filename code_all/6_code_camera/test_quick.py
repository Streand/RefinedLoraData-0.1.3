"""
Quick test to verify the main UI works with both camera backends
"""

import os
import sys
from pathlib import Path

# Add the root project directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_ui_import():
    """Test that the main UI can be imported and created"""
    print("🧪 Testing Main UI Import...")
    
    try:
        # Import using direct module access (VS Code should now find this)
        from UI_main import create_main_ui
        print("✅ Main UI imported successfully")
        
        # Try to create the interface
        interface = create_main_ui()
        print("✅ Main UI interface created successfully")
        print(f"   Interface type: {type(interface)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Main UI test failed: {e}")
        # Fallback to old method
        try:
            main_app_path = os.path.join(os.path.dirname(__file__), '..', '1_code_main_app')
            sys.path.append(main_app_path)
            from UI_main import create_main_ui  # type: ignore
            interface = create_main_ui()
            print("✅ Main UI imported successfully (fallback method)")
            return True
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False

def test_backend_imports():
    """Test that both backends can be imported"""
    print("\n🔧 Testing Backend Imports...")
    
    opencv_available = False
    yolo_available = False
    
    # Test OpenCV backend
    try:
        from backend_camera import CameraAnalyzer
        opencv_analyzer = CameraAnalyzer()
        print("✅ OpenCV backend imported and initialized")
        opencv_available = True
    except Exception as e:
        print(f"❌ OpenCV backend failed: {e}")
    
    # Test YOLO backend
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        yolo_analyzer = YOLOCameraAnalyzer(model_size="nano")
        print(f"✅ YOLO backend imported and initialized (device: {yolo_analyzer.device})")
        yolo_available = True
    except Exception as e:
        print(f"❌ YOLO backend failed: {e}")
    
    return opencv_available, yolo_available

def test_method_compatibility():
    """Test that both backends have compatible methods"""
    print("\n🔗 Testing Method Compatibility...")
    
    try:
        from backend_camera import CameraAnalyzer
        opencv_analyzer = CameraAnalyzer()
        
        # Check OpenCV methods
        assert hasattr(opencv_analyzer, 'analyze_image'), "OpenCV missing analyze_image"
        assert hasattr(opencv_analyzer, 'get_stable_diffusion_prompt'), "OpenCV missing get_stable_diffusion_prompt"
        print("✅ OpenCV has required methods")
        
    except Exception as e:
        print(f"❌ OpenCV method check failed: {e}")
    
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        yolo_analyzer = YOLOCameraAnalyzer(model_size="nano")
        
        # Check YOLO methods
        assert hasattr(yolo_analyzer, 'analyze_image'), "YOLO missing analyze_image"
        assert hasattr(yolo_analyzer, 'get_stable_diffusion_prompt'), "YOLO missing get_stable_diffusion_prompt"
        assert hasattr(yolo_analyzer, 'generate_stable_diffusion_prompt'), "YOLO missing generate_stable_diffusion_prompt"
        print("✅ YOLO has required methods")
        
    except Exception as e:
        print(f"❌ YOLO method check failed: {e}")

def main():
    """Run quick integration tests"""
    print("🚀 Quick Integration Test for Camera UI")
    print("="*50)
    
    # Test UI import
    ui_works = test_ui_import()
    
    # Test backend imports
    opencv_ok, yolo_ok = test_backend_imports()
    
    # Test method compatibility
    test_method_compatibility()
    
    # Summary
    print("\n" + "="*50)
    print("📋 QUICK TEST SUMMARY")
    print("="*50)
    print(f"🖥️  Main UI:        {'✅ WORKS' if ui_works else '❌ FAILED'}")
    print(f"🔧 OpenCV Backend: {'✅ WORKS' if opencv_ok else '❌ FAILED'}")
    print(f"🚀 YOLO Backend:   {'✅ WORKS' if yolo_ok else '❌ FAILED'}")
    
    if ui_works and (opencv_ok or yolo_ok):
        print("\n🎉 Camera system is ready to use!")
        print("   Run the main UI to start analyzing images.")
        if opencv_ok and yolo_ok:
            print("   Both backends are available - choose based on your needs:")
            print("   • OpenCV: Fast analysis, basic pose detection")
            print("   • YOLO: Accurate analysis, detailed pose detection")
        elif opencv_ok:
            print("   OpenCV backend is available for fast analysis")
        elif yolo_ok:
            print("   YOLO backend is available for accurate analysis")
    else:
        print("\n⚠️  Some components are not working properly.")
        print("   Check the error messages above for troubleshooting.")

if __name__ == "__main__":
    main()
