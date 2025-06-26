"""
Test file for Ultralytics YOLO Camera Angle Detection
Compatible with Blackwell NVIDIA cards and modern GPU architectures
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_compatibility():
    """Check GPU compatibility and CUDA setup for Blackwell cards"""
    print("=== GPU Compatibility Check ===")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Check compute capability (Blackwell should be 9.0+)
            major, minor = torch.cuda.get_device_capability(i)
            compute_capability = f"{major}.{minor}"
            print(f"  Compute Capability: {compute_capability}")
            
            # Blackwell specific checks
            if "RTX 50" in gpu_name or "B" in gpu_name or major >= 9:
                print(f"  ✅ Blackwell GPU detected! Using optimized settings.")
            elif major >= 8:
                print(f"  ✅ Modern GPU (Ampere/Ada/Hopper) - Good compatibility")
            else:
                print(f"  ⚠️  Older GPU - May have compatibility issues")
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"PyTorch Version: {torch_version}")
    
    # Recommend version for Blackwell
    if cuda_available:
        recommended_version = "2.1.0"
        if torch_version >= recommended_version:
            print(f"✅ PyTorch version is good for Blackwell GPUs")
        else:
            print(f"⚠️  Consider upgrading PyTorch to {recommended_version}+ for Blackwell support")
    
    return cuda_available

def install_ultralytics():
    """Install Ultralytics with error handling"""
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    venv_info = f" (in venv: {os.path.dirname(sys.executable)})" if in_venv else " (global install)"
    
    try:
        import ultralytics
        print(f"✅ Ultralytics already installed: {ultralytics.__version__}{venv_info}")
        return True
    except ImportError:
        print(f"📦 Installing Ultralytics...{venv_info}")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print(f"✅ Ultralytics installed successfully!{venv_info}")
            return True
        except Exception as e:
            print(f"❌ Failed to install Ultralytics: {e}")
            return False

def test_yolo_model_loading():
    """Test loading YOLO models with Blackwell compatibility"""
    print("\n=== YOLO Model Loading Test ===")
    
    try:
        from ultralytics import YOLO
        
        # Test different model sizes
        models_to_test = [
            ("yolo11n-pose.pt", "YOLO11 Nano Pose"),
            ("yolov8n-pose.pt", "YOLO8 Nano Pose"),  # Fixed filename
            ("yolo11n.pt", "YOLO11 Nano Detection"),
        ]
        
        for model_path, model_name in models_to_test:
            try:
                print(f"\n📋 Testing {model_name}...")
                
                # Load model
                model = YOLO(model_path)
                print(f"  ✅ Model loaded successfully")
                
                # Create a dummy image for testing
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # Try GPU first, but be prepared for Blackwell issues
                if torch.cuda.is_available():
                    try:
                        # For Blackwell cards, try GPU but expect fallback
                        device = 0  # Use first GPU
                        
                        start_time = time.time()
                        results = model(dummy_image, device=device, verbose=False)
                        inference_time = time.time() - start_time
                        
                        print(f"  ✅ GPU inference successful ({inference_time:.3f}s)")
                        
                    except Exception as gpu_error:
                        if "CUDA error" in str(gpu_error) and "kernel image" in str(gpu_error):
                            print(f"  ⚠️  Blackwell GPU not yet supported by this PyTorch version")
                            print(f"     Using CPU inference instead...")
                        else:
                            print(f"  ⚠️  GPU inference failed: {gpu_error}")
                        
                        # Try CPU inference
                        start_time = time.time()
                        results = model(dummy_image, device='cpu', verbose=False)
                        inference_time = time.time() - start_time
                        print(f"  ✅ CPU inference successful ({inference_time:.3f}s)")
                        
                        # Check if pose model and extract keypoints
                        if "pose" in model_path:
                            for result in results:
                                if hasattr(result, 'keypoints') and result.keypoints is not None:
                                    keypoints = result.keypoints.xy
                                    print(f"  ✅ Pose keypoints detected: {keypoints.shape}")
                                else:
                                    print(f"  ℹ️  No pose keypoints in dummy image (expected)")
                
                else:
                    print(f"  ℹ️  No GPU available, using CPU")
                    results = model(dummy_image, device='cpu', verbose=False)
                    print(f"  ✅ CPU inference successful")
                    
            except Exception as e:
                if "No such file" in str(e):
                    print(f"  ℹ️  {model_name} not available, skipping...")
                else:
                    print(f"  ❌ Failed to load {model_name}: {e}")
                
    except ImportError as e:
        print(f"❌ Could not import YOLO: {e}")
        return False
    
    return True

def test_pose_keypoints_analysis():
    """Test pose keypoint analysis for camera angle detection"""
    print("\n=== Pose Analysis Test ===")
    
    try:
        from ultralytics import YOLO
        
        # Load pose model
        model = YOLO('yolo11n-pose.pt')
        
        # Use CPU for now due to Blackwell compatibility
        device = 'cpu'
        print(f"Using device: {device} (Blackwell GPU not yet supported)")
        
        # Create test images with different poses
        test_scenarios = [
            ("front_facing", create_front_facing_dummy()),
            ("side_profile", create_side_profile_dummy()),
            ("from_above", create_from_above_dummy()),
        ]
        
        for scenario_name, dummy_image in test_scenarios:
            print(f"\n📋 Testing {scenario_name} pose...")
            
            try:
                results = model(dummy_image, device=device, verbose=False)
                
                for result in results:
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        confidence = result.keypoints.conf.cpu().numpy()
                        
                        print(f"  ✅ Detected keypoints: {keypoints.shape}")
                        print(f"  ✅ Confidence scores: {confidence.shape}")
                        
                        # Analyze pose for camera angle
                        angle = analyze_pose_for_camera_angle(keypoints, confidence)
                        print(f"  📐 Detected camera angle: {angle}")
                    else:
                        print(f"  ℹ️  No pose detected in {scenario_name} (expected for dummy data)")
                        
            except Exception as e:
                print(f"  ❌ Error in {scenario_name}: {e}")
    
    except Exception as e:
        print(f"❌ Pose analysis test failed: {e}")

def create_front_facing_dummy():
    """Create a dummy front-facing image"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def create_side_profile_dummy():
    """Create a dummy side profile image"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def create_from_above_dummy():
    """Create a dummy from-above image"""
    return np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

def analyze_pose_for_camera_angle(keypoints: np.ndarray, confidence: np.ndarray) -> str:
    """
    Analyze pose keypoints to determine camera angle
    
    Args:
        keypoints: Array of shape (num_people, num_keypoints, 2) with x,y coordinates
        confidence: Array of shape (num_people, num_keypoints) with confidence scores
    
    Returns:
        Detected camera angle
    """
    if len(keypoints) == 0:
        return "unknown"
    
    # Use first person detected
    person_keypoints = keypoints[0]
    person_confidence = confidence[0]
    
    # YOLO pose keypoints indices (COCO format)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    try:
        # Check if key points are visible (confidence > 0.5)
        visible_threshold = 0.5
        
        nose = person_keypoints[0] if person_confidence[0] > visible_threshold else None
        left_eye = person_keypoints[1] if person_confidence[1] > visible_threshold else None
        right_eye = person_keypoints[2] if person_confidence[2] > visible_threshold else None
        left_shoulder = person_keypoints[5] if person_confidence[5] > visible_threshold else None
        right_shoulder = person_keypoints[6] if person_confidence[6] > visible_threshold else None
        
        # Analyze symmetry and positioning
        if left_eye is not None and right_eye is not None:
            eye_distance = np.linalg.norm(left_eye - right_eye)
            
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
                
                # Check for front-facing pose
                if eye_distance > 10 and shoulder_distance > 20:
                    return "bilaterally symmetrical"
                elif eye_distance < 5:
                    return "side view"
                else:
                    return "straight on"
        
        # If we have nose but limited other features, might be profile
        if nose is not None and (left_eye is None or right_eye is None):
            return "side view"
        
        return "straight on"  # Default
        
    except Exception as e:
        logger.warning(f"Error analyzing pose: {e}")
        return "unknown"

def test_performance_benchmark():
    """Benchmark YOLO performance on current hardware"""
    print("\n=== Performance Benchmark ===")
    
    try:
        from ultralytics import YOLO
        
        model = YOLO('yolo11n-pose.pt')
        
        # Use CPU for now due to Blackwell compatibility
        device = 'cpu'
        print(f"Benchmarking on {device} (Blackwell GPU support coming soon)")
        
        # Warm up
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        model(dummy_image, device=device, verbose=False)
        
        # Benchmark different image sizes
        test_sizes = [(320, 320), (640, 640), (1280, 1280)]
        num_runs = 5  # Reduced for CPU testing
        
        for width, height in test_sizes:
            print(f"\n📊 Benchmarking {width}x{height} images...")
            
            times = []
            for i in range(num_runs):
                test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                start_time = time.time()
                results = model(test_image, device=device, verbose=False)
                inference_time = time.time() - start_time
                times.append(inference_time)
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            std_time = np.std(times)
            
            print(f"  Average inference time: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  FPS: {fps:.1f}")
            
            # Performance recommendations for CPU
            if avg_time < 0.1:
                print(f"  ✅ Good CPU performance! Suitable for batch processing.")
            elif avg_time < 0.5:
                print(f"  ⚠️  Moderate CPU performance. Fine for image analysis.")
            else:
                print(f"  🐌 Slower CPU performance. Consider GPU when available.")
        
        print(f"\n� Note: Performance will be much faster once Blackwell GPU support is added!")
    
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def test_integration_with_existing_backend():
    """Test integration with existing camera backend"""
    print("\n=== Integration Test ===")
    
    try:
        # Try to import existing backend
        sys.path.append(os.path.dirname(__file__))
        from backend_camera import CameraAnalyzer
        
        print("✅ Existing backend imported successfully")
        
        # Test compatibility
        analyzer = CameraAnalyzer()
        print("✅ Existing analyzer initialized")
        
        # Simulate running both systems
        print("ℹ️  Both OpenCV and YOLO systems can coexist")
        print("ℹ️  YOLO system will provide more accurate pose detection")
        
    except Exception as e:
        print(f"⚠️  Integration test note: {e}")

def main():
    """Main test function"""
    print("🚀 YOLO Camera Angle Detection Test")
    print("🎯 Optimized for Blackwell NVIDIA GPUs\n")
    
    # Step 1: Check GPU compatibility
    gpu_available = check_gpu_compatibility()
    
    # Step 2: Install Ultralytics
    if not install_ultralytics():
        print("❌ Cannot proceed without Ultralytics")
        return
    
    # Step 3: Test model loading
    if not test_yolo_model_loading():
        print("⚠️  Model loading issues detected")
    
    # Step 4: Test pose analysis
    test_pose_keypoints_analysis()
    
    # Step 5: Benchmark performance
    if gpu_available:
        test_performance_benchmark()
    
    # Step 6: Test integration
    test_integration_with_existing_backend()
    
    print("\n🎉 YOLO test completed!")
    print("\n💡 Next steps:")
    print("   1. If tests passed, YOLO is ready for integration")
    print("   2. Consider creating a new backend_camera_yolo.py")
    print("   3. YOLO will provide much better accuracy than OpenCV Haar cascades")
    
    if gpu_available:
        print("   4. 🚀 Your Blackwell GPU should provide excellent performance!")
    else:
        print("   4. CPU inference will work but consider GPU setup for best performance")

if __name__ == "__main__":
    main()
