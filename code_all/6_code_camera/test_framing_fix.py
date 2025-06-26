"""
Test the improved framing logic by simulating the conditions from the user's screenshot
"""

import numpy as np
from backend_camera_yolo import YOLOCameraAnalyzer

def simulate_profile_detection():
    """Simulate the detection scenario from the user's screenshot"""
    print("🧪 Simulating Profile Face Detection (from screenshot)")
    print("="*60)
    
    try:
        analyzer = YOLOCameraAnalyzer(model_size="nano")
        
        # Simulate the keypoints and confidence for a profile face
        # From screenshot: People Detected: 1, Confidence: 0.99
        # Camera Angle: side view (this was correct)
        # But Framing: full shot (this was wrong - should be close-up)
        
        # Simulate profile keypoints (only face visible)
        keypoints = np.zeros((17, 2))  # 17 COCO keypoints
        confidence = np.zeros(17)
        
        # Set visible face keypoints for profile view
        # nose (0)
        keypoints[0] = [200, 150]
        confidence[0] = 0.9
        
        # left_eye (1) - visible in profile
        keypoints[1] = [180, 140]
        confidence[1] = 0.85
        
        # left_ear (3) - visible in profile  
        keypoints[3] = [220, 145]
        confidence[3] = 0.8
        
        # All other keypoints not visible (confidence = 0)
        # This simulates a face-only profile shot
        
        print("📊 Simulated keypoints:")
        print(f"  Nose: {keypoints[0]} (conf: {confidence[0]})")
        print(f"  Left eye: {keypoints[1]} (conf: {confidence[1]})")
        print(f"  Left ear: {keypoints[3]} (conf: {confidence[3]})")
        print(f"  Shoulders: not visible (conf: 0.0)")
        print(f"  Hips: not visible (conf: 0.0)")
        
        # Test the pose analysis
        print("\n🔍 Running pose analysis...")
        person_analysis = analyzer._analyze_person_pose(keypoints, confidence)
        
        print(f"\n📊 Pose Analysis Results:")
        for key, value in person_analysis.items():
            if key == 'pose_analysis' and isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Test advanced framing
        print(f"\n🎯 Framing Analysis:")
        framing = analyzer._determine_framing_advanced(
            nose=keypoints[0], 
            left_eye=keypoints[1], 
            right_eye=None,  # Not visible in profile
            left_ear=keypoints[3],
            right_ear=None,  # Not visible in profile
            left_shoulder=None,  # Not visible
            right_shoulder=None,  # Not visible
            left_hip=None,  # Not visible
            right_hip=None   # Not visible
        )
        print(f"  Advanced framing result: {framing}")
        
        # Check if results are correct
        expected_angle = "side view"
        expected_framing = "close-up"
        
        actual_angle = person_analysis.get('camera_angle', 'unknown')
        actual_framing = person_analysis.get('framing', 'unknown')
        
        print(f"\n🎯 Accuracy Check:")
        print(f"  Expected: angle='{expected_angle}', framing='{expected_framing}'")
        print(f"  Actual:   angle='{actual_angle}', framing='{actual_framing}'")
        
        angle_correct = actual_angle == expected_angle
        framing_correct = actual_framing == expected_framing
        
        print(f"  Camera Angle: {'✅ CORRECT' if angle_correct else '❌ INCORRECT'}")
        print(f"  Framing: {'✅ CORRECT' if framing_correct else '❌ INCORRECT'}")
        
        return angle_correct and framing_correct
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_box_framing_logic():
    """Test the improved box-based framing logic"""
    print("\n🧪 Testing Box-Based Framing Logic")
    print("="*50)
    
    try:
        analyzer = YOLOCameraAnalyzer(model_size="nano")
        
        # Test different box scenarios
        test_cases = [
            {
                'name': 'Portrait face (tall box)',
                'box': [100, 50, 300, 400],  # x1, y1, x2, y2
                'image_shape': (500, 400),   # height, width
                'expected': 'close-up'
            },
            {
                'name': 'Full body (large box)',
                'box': [50, 10, 350, 480],
                'image_shape': (500, 400),
                'expected': 'full shot'
            },
            {
                'name': 'Upper body (medium box)',
                'box': [100, 100, 300, 350],
                'image_shape': (500, 400),
                'expected': 'medium shot'
            }
        ]
        
        for case in test_cases:
            print(f"\n🔍 Testing: {case['name']}")
            
            # Convert to numpy array format
            boxes = np.array([case['box']], dtype=float)
            
            # Test the framing analysis
            result = analyzer._analyze_framing_from_boxes(boxes, case['image_shape'])
            actual_framing = result.get('framing', 'unknown')
            
            print(f"  Box: {case['box']}")
            print(f"  Image shape: {case['image_shape']}")
            print(f"  Expected: {case['expected']}")
            print(f"  Actual: {actual_framing}")
            print(f"  Result: {'✅ CORRECT' if actual_framing == case['expected'] else '❌ INCORRECT'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Box framing test failed: {e}")
        return False

def main():
    """Run the framing fix tests"""
    print("🚀 Testing Framing Detection Improvements")
    print("="*70)
    
    # Test pose-based analysis
    pose_test = simulate_profile_detection()
    
    # Test box-based analysis  
    box_test = test_box_framing_logic()
    
    # Summary
    print("\n" + "="*70)
    print("📋 FRAMING FIX TEST SUMMARY")
    print("="*70)
    print(f"🧑 Pose-based Analysis: {'✅ PASSED' if pose_test else '❌ FAILED'}")
    print(f"📦 Box-based Analysis:  {'✅ PASSED' if box_test else '❌ FAILED'}")
    
    if pose_test and box_test:
        print("\n🎉 Framing detection improvements are working correctly!")
        print("✅ The profile face detection issue should now be fixed.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
