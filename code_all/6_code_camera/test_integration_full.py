"""
Comprehensive integration test for both OpenCV and YOLO camera backends
Tests both backends individually and through the main UI integration
"""

import os
import sys
import time
import logging
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image(filename: str = "test_person.jpg", size: tuple = (512, 768)) -> str:
    """Create a simple test image with a person-like figure"""
    img = Image.new('RGB', size, color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple person figure
    # Head (circle)
    head_center = (size[0]//2, size[1]//4)
    head_radius = 50
    draw.ellipse([
        head_center[0] - head_radius, head_center[1] - head_radius,
        head_center[0] + head_radius, head_center[1] + head_radius
    ], fill='peachpuff', outline='black')
    
    # Body (rectangle)
    body_top = head_center[1] + head_radius + 10
    body_width = 80
    body_height = 200
    draw.rectangle([
        size[0]//2 - body_width//2, body_top,
        size[0]//2 + body_width//2, body_top + body_height
    ], fill='blue', outline='black')
    
    # Arms (lines)
    arm_y = body_top + 50
    draw.line([size[0]//2 - body_width//2, arm_y, size[0]//2 - body_width//2 - 60, arm_y + 80], fill='black', width=8)
    draw.line([size[0]//2 + body_width//2, arm_y, size[0]//2 + body_width//2 + 60, arm_y + 80], fill='black', width=8)
    
    # Legs (lines)
    leg_y = body_top + body_height
    draw.line([size[0]//2 - 20, leg_y, size[0]//2 - 20, leg_y + 150], fill='black', width=8)
    draw.line([size[0]//2 + 20, leg_y, size[0]//2 + 20, leg_y + 150], fill='black', width=8)
    
    # Save the image
    img.save(filename)
    logger.info(f"Created test image: {filename}")
    return filename

def test_opencv_backend():
    """Test the OpenCV backend"""
    print("\n" + "="*60)
    print("🔧 TESTING OPENCV BACKEND")
    print("="*60)
    
    try:
        from backend_camera import CameraAnalyzer
        
        analyzer = CameraAnalyzer()
        print("✅ OpenCV backend imported successfully")
        
        # Create test image
        test_img = create_test_image("test_opencv.jpg")
        
        # Analyze image
        start_time = time.time()
        result = analyzer.analyze_image(test_img)
        analysis_time = time.time() - start_time
        
        print(f"⏱️  Analysis time: {analysis_time:.3f} seconds")
        
        if "error" in result:
            print(f"❌ Analysis error: {result['error']}")
            return False
        
        print("📊 Analysis Results:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        
        # Test prompt generation
        prompt = analyzer.get_stable_diffusion_prompt(result)
        print(f"🎨 Generated prompt: {prompt}")
        
        # Cleanup
        if os.path.exists(test_img):
            os.remove(test_img)
        
        print("✅ OpenCV backend test completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV backend not available: {e}")
        print("   Install with: pip install opencv-python numpy")
        return False
    except Exception as e:
        print(f"❌ OpenCV backend error: {e}")
        return False

def test_yolo_backend():
    """Test the YOLO backend"""
    print("\n" + "="*60)
    print("🚀 TESTING YOLO BACKEND")
    print("="*60)
    
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        
        # Test different model sizes
        model_sizes = ["nano"]  # Start with nano for faster testing
        
        for model_size in model_sizes:
            print(f"\n🤖 Testing YOLO {model_size} model...")
            
            try:
                analyzer = YOLOCameraAnalyzer(model_size=model_size)
                print(f"✅ YOLO {model_size} backend imported successfully")
                print(f"🖥️  Device: {analyzer.device}")
                print(f"🔧 Model: {analyzer.model_size}")
                
                if hasattr(analyzer, 'model') and analyzer.model is not None:
                    print("✅ YOLO model loaded successfully")
                else:
                    print("⚠️  YOLO model not fully initialized")
                    continue
                
                # Create test image
                test_img = create_test_image(f"test_yolo_{model_size}.jpg")
                
                # Analyze image
                start_time = time.time()
                result = analyzer.analyze_image(test_img)
                analysis_time = time.time() - start_time
                
                print(f"⏱️  Analysis time: {analysis_time:.3f} seconds")
                
                if "error" in result:
                    print(f"❌ Analysis error: {result['error']}")
                    continue
                
                print("📊 Analysis Results:")
                for key, value in result.items():
                    print(f"   {key}: {value}")
                
                # Test prompt generation
                prompt = analyzer.generate_stable_diffusion_prompt(result)
                print(f"🎨 Generated prompt: {prompt}")
                
                # Show performance info
                if 'inference_time' in result:
                    print(f"📈 Backend analysis time: {result['inference_time']:.3f}s")
                if 'confidence' in result:
                    print(f"📊 Pose confidence: {result['confidence']:.3f}")
                
                # Cleanup
                if os.path.exists(test_img):
                    os.remove(test_img)
                
                print(f"✅ YOLO {model_size} backend test completed successfully")
                return True
                
            except Exception as e:
                print(f"❌ YOLO {model_size} backend error: {e}")
                continue
        
        return False
        
    except ImportError as e:
        print(f"❌ YOLO backend not available: {e}")
        print("   Install with: pip install torch torchvision ultralytics")
        return False
    except Exception as e:
        print(f"❌ YOLO backend error: {e}")
        return False

def test_main_ui_integration():
    """Test the main UI integration"""
    print("\n" + "="*60)
    print("🖥️  TESTING MAIN UI INTEGRATION")
    print("="*60)
    
    try:
        # Add main app to path
        main_app_path = os.path.join(os.path.dirname(__file__), '..', '1_code_main_app')
        sys.path.append(main_app_path)
        
        from UI_main import create_main_ui
        
        print("✅ Main UI imported successfully")
        
        # Create the UI (this tests that all components can be created)
        interface = create_main_ui()
        print("✅ Main UI created successfully")
        
        # Test that the camera tab is included
        if hasattr(interface, 'blocks') or hasattr(interface, '_Blocks__blocks'):
            print("✅ UI blocks structure created")
        
        print("✅ Main UI integration test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Main UI import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Main UI integration error: {e}")
        return False

def test_backend_comparison():
    """Compare performance and results between backends"""
    print("\n" + "="*60)
    print("⚖️  BACKEND COMPARISON")
    print("="*60)
    
    # Create test image
    test_img = create_test_image("comparison_test.jpg", (640, 960))
    
    results = {}
    
    # Test OpenCV
    try:
        from backend_camera import CameraAnalyzer
        opencv_analyzer = CameraAnalyzer()
        
        start_time = time.time()
        opencv_result = opencv_analyzer.analyze_image(test_img)
        opencv_time = time.time() - start_time
        
        results['opencv'] = {
            'time': opencv_time,
            'result': opencv_result,
            'prompt': opencv_analyzer.get_stable_diffusion_prompt(opencv_result)
        }
        print(f"🔧 OpenCV: {opencv_time:.3f}s")
        
    except Exception as e:
        print(f"🔧 OpenCV failed: {e}")
        results['opencv'] = None
    
    # Test YOLO
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        yolo_analyzer = YOLOCameraAnalyzer(model_size="nano")
        
        start_time = time.time()
        yolo_result = yolo_analyzer.analyze_image(test_img)
        yolo_time = time.time() - start_time
        
        results['yolo'] = {
            'time': yolo_time,
            'result': yolo_result,
            'prompt': yolo_analyzer.generate_stable_diffusion_prompt(yolo_result)
        }
        print(f"🚀 YOLO: {yolo_time:.3f}s")
        
    except Exception as e:
        print(f"🚀 YOLO failed: {e}")
        results['yolo'] = None
    
    # Compare results
    if results['opencv'] and results['yolo']:
        print("\n📊 COMPARISON RESULTS:")
        print(f"Speed: OpenCV {results['opencv']['time']:.3f}s vs YOLO {results['yolo']['time']:.3f}s")
        
        if results['opencv']['time'] < results['yolo']['time']:
            speedup = results['yolo']['time'] / results['opencv']['time']
            print(f"🏃 OpenCV is {speedup:.1f}x faster")
        else:
            speedup = results['opencv']['time'] / results['yolo']['time']
            print(f"🏃 YOLO is {speedup:.1f}x faster")
        
        print("\nFraming Detection:")
        print(f"  OpenCV: {results['opencv']['result'].get('framing', 'unknown')}")
        print(f"  YOLO:   {results['yolo']['result'].get('framing', 'unknown')}")
        
        print("\nAngle Detection:")
        print(f"  OpenCV: {results['opencv']['result'].get('angle', 'unknown')}")
        print(f"  YOLO:   {results['yolo']['result'].get('angle', 'unknown')}")
    
    # Cleanup
    if os.path.exists(test_img):
        os.remove(test_img)
    
    return results

def check_environment():
    """Check the current environment setup"""
    print("🔍 ENVIRONMENT CHECK")
    print("="*60)
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("📦 Virtual environment: Active")
        print(f"   Path: {sys.prefix}")
    else:
        print("📦 Virtual environment: Not detected")
    
    # Check CUDA availability
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            major, minor = torch.cuda.get_device_capability(0)
            print(f"📊 Compute capability: {major}.{minor}")
    except ImportError:
        print("🔥 PyTorch: Not installed")
    
    # Check OpenCV
    try:
        import cv2
        print(f"👁️  OpenCV: {cv2.__version__}")
    except ImportError:
        print("👁️  OpenCV: Not installed")
    
    # Check Ultralytics
    try:
        import ultralytics
        print(f"🚀 Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("🚀 Ultralytics: Not installed")
    
    print()

def main():
    """Run all integration tests"""
    print("🧪 CAMERA BACKEND INTEGRATION TESTS")
    print("="*60)
    
    # Check environment
    check_environment()
    
    # Run tests
    opencv_success = test_opencv_backend()
    yolo_success = test_yolo_backend()
    ui_success = test_main_ui_integration()
    
    # Run comparison if both backends work
    if opencv_success and yolo_success:
        test_backend_comparison()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"🔧 OpenCV Backend: {'✅ PASS' if opencv_success else '❌ FAIL'}")
    print(f"🚀 YOLO Backend:   {'✅ PASS' if yolo_success else '❌ FAIL'}")
    print(f"🖥️  UI Integration: {'✅ PASS' if ui_success else '❌ FAIL'}")
    
    if opencv_success or yolo_success:
        print("\n🎉 At least one backend is working!")
        if not yolo_success:
            print("💡 To enable YOLO backend: pip install torch torchvision ultralytics")
    else:
        print("\n⚠️  No backends are working. Check dependencies.")
        print("🔧 For OpenCV: pip install opencv-python numpy")
        print("🚀 For YOLO: pip install torch torchvision ultralytics")
    
    print(f"\n🚀 Ready to use camera analysis with {['OpenCV' if opencv_success else '', 'YOLO' if yolo_success else ''][0] or 'no backends'}")

if __name__ == "__main__":
    main()
