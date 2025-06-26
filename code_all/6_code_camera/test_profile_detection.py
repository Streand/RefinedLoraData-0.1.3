"""
Test specifically for profile/side view face detection issues
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

def create_realistic_profile_face():
    """Create a realistic side profile face for testing"""
    img = Image.new('RGB', (400, 600), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # Profile face outline
    face_points = [
        (150, 150),  # forehead
        (160, 120),  # top of head
        (180, 110),  # crown
        (220, 110),  # back of head
        (250, 130),  # back neck
        (240, 180),  # neck
        (230, 220),  # jaw back
        (210, 250),  # jaw line
        (180, 270),  # chin
        (150, 260),  # front jaw
        (130, 240),  # mouth area
        (125, 220),  # nose tip
        (120, 200),  # nose bridge
        (125, 180),  # eye area
        (140, 160),  # forehead front
    ]
    
    # Draw face outline
    draw.polygon(face_points, fill='peachpuff', outline='black', width=2)
    
    # Eye (only one visible in profile)
    draw.ellipse([140, 175, 155, 185], fill='white', outline='black')
    draw.ellipse([145, 178, 150, 183], fill='black')
    
    # Eyebrow
    draw.arc([135, 170, 160, 180], 0, 180, fill='brown', width=2)
    
    # Nose
    draw.ellipse([120, 195, 128, 205], fill='pink', outline='darkred')
    
    # Mouth
    draw.arc([125, 235, 145, 245], 0, 180, fill='red', width=2)
    
    # Ear
    draw.ellipse([235, 190, 255, 220], fill='peachpuff', outline='black', width=2)
    
    # Hair
    hair_points = [
        (160, 120), (180, 110), (220, 110), (250, 130),
        (260, 140), (270, 160), (265, 180), (240, 170),
        (200, 160), (170, 150), (150, 140)
    ]
    draw.polygon(hair_points, fill='brown')
    
    return img

def test_profile_detection():
    """Test profile face detection with realistic image"""
    print("🧪 Testing Profile Face Detection")
    print("="*50)
    
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        
        # Create analyzer
        analyzer = YOLOCameraAnalyzer(model_size="nano")
        print(f"✅ YOLO analyzer initialized (device: {analyzer.device})")
        
        # Create realistic profile face
        profile_img = create_realistic_profile_face()
        test_path = "test_profile_face.jpg"
        profile_img.save(test_path)
        print(f"✅ Created profile face image: {test_path}")
        
        # Analyze image
        print("\n🔍 Running YOLO analysis on profile face...")
        result = analyzer.analyze_image(test_path)
        
        print("\n📊 Profile Analysis Results:")
        for key, value in result.items():
            if key == 'pose_analysis' and isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Check if results match expected profile characteristics
        expected_angle = "side view"
        expected_framing = "close-up"
        
        actual_angle = result.get('camera_angle', 'unknown')
        actual_framing = result.get('framing', 'unknown')
        
        print(f"\n🎯 Expected vs Actual:")
        print(f"  Camera Angle: expected='{expected_angle}', actual='{actual_angle}'")
        print(f"  Framing: expected='{expected_framing}', actual='{actual_framing}'")
        
        # Generate SD prompt
        prompt = analyzer.get_stable_diffusion_prompt(result)
        print(f"\n🎨 Generated SD Prompt: {prompt}")
        
        # Evaluate accuracy
        angle_correct = actual_angle == expected_angle
        framing_correct = actual_framing == expected_framing
        
        print(f"\n📋 Accuracy Assessment:")
        print(f"  Camera Angle: {'✅ CORRECT' if angle_correct else '❌ INCORRECT'}")
        print(f"  Framing: {'✅ CORRECT' if framing_correct else '❌ INCORRECT'}")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        return angle_correct and framing_correct
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def debug_yolo_detection():
    """Debug what YOLO is actually detecting"""
    print("\n🔍 Debugging YOLO Detection Process")
    print("="*50)
    
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        from ultralytics import YOLO
        
        # Create analyzer
        analyzer = YOLOCameraAnalyzer(model_size="nano")
        
        # Create test image
        profile_img = create_realistic_profile_face()
        test_path = "debug_profile.jpg"
        profile_img.save(test_path)
        
        # Run raw YOLO detection
        print("🔍 Running raw YOLO detection...")
        results = analyzer.model(test_path, device=analyzer.device, verbose=True)
        
        result = results[0]
        print(f"\n📊 Raw YOLO Results:")
        print(f"  Has keypoints: {hasattr(result, 'keypoints')}")
        print(f"  Has boxes: {hasattr(result, 'boxes')}")
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            confidence = result.keypoints.conf.cpu().numpy()
            print(f"  Keypoints shape: {keypoints.shape}")
            print(f"  Confidence shape: {confidence.shape}")
            print(f"  People detected: {len(keypoints)}")
            
            if len(keypoints) > 0:
                print(f"  First person confidence scores:")
                for i, conf in enumerate(confidence[0]):
                    if conf > 0.3:  # Show visible keypoints
                        print(f"    Keypoint {i}: confidence={conf:.3f}")
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            print(f"  Boxes shape: {boxes.shape}")
            print(f"  Number of detections: {len(boxes)}")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")

def main():
    """Run profile detection tests"""
    print("🚀 Profile Face Detection Test")
    print("="*60)
    
    # Test profile detection
    profile_test = test_profile_detection()
    
    # Debug YOLO detection
    debug_yolo_detection()
    
    # Summary
    print("\n" + "="*60)
    print("📋 PROFILE DETECTION TEST SUMMARY")
    print("="*60)
    print(f"👤 Profile Detection: {'✅ PASSED' if profile_test else '❌ FAILED'}")
    
    if not profile_test:
        print("\n💡 The issue might be:")
        print("   1. YOLO model not detecting faces in synthetic images")
        print("   2. Keypoint detection threshold too high")
        print("   3. Need to test with real photos instead of drawings")
        print("\n🔧 Suggested fixes:")
        print("   1. Lower confidence thresholds")
        print("   2. Test with actual photographs")
        print("   3. Improve synthetic image realism")

if __name__ == "__main__":
    main()
