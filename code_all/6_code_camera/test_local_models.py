#!/usr/bin/env python3
"""
Test script to verify YOLO models are loaded from local camera folder
"""

import os
import sys

# Add camera folder to path
camera_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(camera_dir)

from backend_camera_yolo import YOLOCameraAnalyzer

def test_local_model_usage():
    """Test that models are loaded from the local camera folder"""
    
    print("=" * 60)
    print("YOLO Local Model Usage Test")
    print("=" * 60)
    
    # List available models in camera folder
    models_in_folder = []
    for filename in os.listdir(camera_dir):
        if filename.endswith('.pt'):
            models_in_folder.append(filename)
    
    print(f"Models available in camera folder: {models_in_folder}")
    
    # Test loading analyzer
    try:
        print("\nTesting YOLO model loading...")
        analyzer = YOLOCameraAnalyzer(model_size="nano")
        
        print(f"✅ Model loaded successfully!")
        print(f"Device: {analyzer.device}")
        print(f"Model size: {analyzer.model_size}")
        print(f"Initialized: {analyzer.is_initialized}")
        
        # Check if root folder has the model (should not be created by our loading)
        root_model = os.path.join("..", "..", "yolo11n-pose.pt")
        if os.path.exists(root_model):
            print(f"⚠️  Model also exists in root folder")
        else:
            print(f"✅ No model in root folder - using local only")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_local_model_usage()
