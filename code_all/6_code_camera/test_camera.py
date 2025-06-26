"""
Test script for Camera Analysis functionality
"""

import os
import sys
import numpy as np
from PIL import Image

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def create_test_image():
    """Create a simple test image for analysis"""
    # Create a simple test image with a face-like shape
    img_array = np.ones((400, 300, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add a simple "face" - circle
    center_x, center_y = 150, 150
    radius = 50
    
    for y in range(400):
        for x in range(300):
            if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                img_array[y, x] = [255, 220, 200]  # Skin tone
    
    # Add "eyes"
    for eye_x in [center_x - 15, center_x + 15]:
        for y in range(center_y - 10, center_y - 5):
            for x in range(eye_x - 5, eye_x + 5):
                if 0 <= y < 400 and 0 <= x < 300:
                    img_array[y, x] = [0, 0, 0]  # Black eyes
    
    img = Image.fromarray(img_array)
    return img

def test_backend():
    """Test the backend camera analyzer"""
    print("Testing Camera Analysis Backend...")
    
    try:
        from backend_camera import CameraAnalyzer
        print("✅ Backend imported successfully")
        
        # Create analyzer
        analyzer = CameraAnalyzer()
        print("✅ Analyzer created successfully")
        
        # Create test image
        test_img = create_test_image()
        test_path = "test_camera_image.jpg"
        test_img.save(test_path)
        print("✅ Test image created")
        
        # Analyze the test image
        result = analyzer.analyze_image(test_path)
        print("✅ Analysis completed")
        print("Analysis Result:", result)
        
        # Test prompt generation
        prompt = analyzer.get_stable_diffusion_prompt(result)
        print("✅ Prompt generation successful")
        print("SD Prompt:", prompt)
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        print("✅ Cleanup completed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Missing dependencies. Install with: pip install opencv-python numpy")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ui():
    """Test the UI components"""
    print("\nTesting Camera Analysis UI...")
    
    try:
        from UI_camera import CameraUI, create_camera_tab
        print("✅ UI components imported successfully")
        
        # Create UI instance
        ui = CameraUI()
        print("✅ UI instance created")
        
        # Test reference guide creation
        guide = ui.create_reference_guide()
        print("✅ Reference guide created")
        print(f"Guide length: {len(guide)} characters")
        
        return True
        
    except ImportError as e:
        print(f"❌ UI Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ UI Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Camera Analysis Module Tests")
    print("=" * 40)
    
    backend_ok = test_backend()
    ui_ok = test_ui()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"Backend: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"UI: {'✅ PASS' if ui_ok else '❌ FAIL'}")
    
    if backend_ok and ui_ok:
        print("\n🎉 All tests passed! Camera analysis is ready to use.")
        print("\n💡 Usage tips:")
        print("- Upload clear images with visible subjects for best results")
        print("- Works with portraits, full body shots, and various angles")
        print("- Use generated prompts in Stable Diffusion for consistent framing")
    else:
        print("\n⚠️  Some tests failed. Check dependencies and installation.")
        print("Required: pip install opencv-python numpy pillow gradio")

if __name__ == "__main__":
    main()
