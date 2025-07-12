"""
Test script for Clothing Analysis Module
Quick tests to verify the module is working correctly
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not available")
        return False
    
    try:
        import gradio as gr
        print(f"✓ Gradio {gr.__version__}")
    except ImportError:
        print("✗ Gradio not available")
        return False
    
    try:
        from transformers import AutoProcessor
        print("✓ Transformers available")
    except ImportError:
        print("✗ Transformers not available - install with: pip install transformers")
        return False
    
    try:
        from backend_clothing import ClothingAnalyzer, create_clothing_analyzer
        print("✓ Backend module available")
    except ImportError as e:
        print(f"✗ Backend module not available: {e}")
        return False
    
    try:
        from UI_clothing import create_clothing_tab
        print("✓ UI module available")
    except ImportError as e:
        print(f"✗ UI module not available: {e}")
        return False
    
    return True

def test_device_detection():
    """Test GPU/device detection"""
    print("\nTesting device detection...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("! CUDA not available - will use CPU")
        
        try:
            from backend_clothing import ClothingAnalyzer
            analyzer = ClothingAnalyzer()
            device_info = analyzer.get_device_info()
            print(f"✓ Device info: {device_info}")
        except Exception as e:
            print(f"! Could not get device info: {e}")
            
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation without loading heavy models"""
    print("\nTesting model creation...")
    
    try:
        from backend_clothing import ClothingAnalyzer
        
        # Test analyzer creation (without loading models)
        analyzer = ClothingAnalyzer()
        print("✓ Analyzer created successfully")
        
        # Test configuration
        if hasattr(analyzer, 'device'):
            print(f"✓ Device configured: {analyzer.device}")
        
        print("✓ Model creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_ui_creation():
    """Test UI creation"""
    print("\nTesting UI creation...")
    
    try:
        import gradio as gr
        from UI_clothing import create_clothing_tab
        
        # Create a test interface
        with gr.Blocks() as demo:
            tab = create_clothing_tab()
        
        print("✓ UI created successfully")
        return True
        
    except Exception as e:
        print(f"✗ UI creation failed: {e}")
        return False

def test_data_storage():
    """Test data storage folder"""
    print("\nTesting data storage...")
    
    data_folder = Path("../../data_storage/data_store_clothing")
    
    if data_folder.exists():
        print(f"✓ Data storage folder exists: {data_folder.absolute()}")
        
        # Test write permissions
        try:
            test_file = data_folder / "test_write.txt"
            test_file.write_text("test")
            test_file.unlink()
            print("✓ Write permissions OK")
        except Exception as e:
            print(f"! Write permission issue: {e}")
            
    else:
        print(f"! Data storage folder missing: {data_folder.absolute()}")
        try:
            data_folder.mkdir(parents=True, exist_ok=True)
            print("✓ Created data storage folder")
        except Exception as e:
            print(f"✗ Could not create data storage folder: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("CLOTHING ANALYSIS MODULE - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Detection", test_device_detection),
        ("Model Creation", test_model_creation),
        ("UI Creation", test_ui_creation),
        ("Data Storage", test_data_storage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The clothing analysis module is ready to use.")
        print("You can now run:")
        print("  - launch_clothing.bat (Windows)")
        print("  - python launch_clothing.py (Cross-platform)")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the errors above.")
        print("Make sure to install dependencies: pip install -r requirements_clothing.txt")

if __name__ == "__main__":
    main()
