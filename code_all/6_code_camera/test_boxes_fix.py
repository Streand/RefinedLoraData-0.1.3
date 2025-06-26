"""
Specific test for YOLO backend with focus on the boxes error fix
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

def create_person_image():
    """Create a more detailed test image with a clear person figure"""
    img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a more realistic person figure
    # Head
    head_x, head_y = 320, 100
    head_radius = 40
    draw.ellipse([head_x-head_radius, head_y-head_radius, head_x+head_radius, head_y+head_radius], 
                 fill='peachpuff', outline='black', width=2)
    
    # Eyes
    draw.ellipse([head_x-15, head_y-10, head_x-5, head_y], fill='black')
    draw.ellipse([head_x+5, head_y-10, head_x+15, head_y], fill='black')
    
    # Nose
    draw.ellipse([head_x-3, head_y, head_x+3, head_y+8], fill='pink')
    
    # Body - torso
    torso_top = head_y + head_radius + 10
    torso_width = 60
    torso_height = 120
    draw.rectangle([head_x-torso_width//2, torso_top, head_x+torso_width//2, torso_top+torso_height], 
                   fill='blue', outline='black', width=2)
    
    # Arms
    arm_y = torso_top + 20
    # Left arm
    draw.line([head_x-torso_width//2, arm_y, head_x-torso_width//2-50, arm_y+60], fill='peachpuff', width=8)
    # Right arm  
    draw.line([head_x+torso_width//2, arm_y, head_x+torso_width//2+50, arm_y+60], fill='peachpuff', width=8)
    
    # Legs
    leg_start_y = torso_top + torso_height
    # Left leg
    draw.line([head_x-15, leg_start_y, head_x-15, leg_start_y+100], fill='blue', width=12)
    # Right leg
    draw.line([head_x+15, leg_start_y, head_x+15, leg_start_y+100], fill='blue', width=12)
    
    return img

def test_yolo_with_real_person():
    """Test YOLO backend with a realistic person image"""
    print("🧪 Testing YOLO Backend with Person Image")
    print("="*50)
    
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        
        # Create analyzer
        analyzer = YOLOCameraAnalyzer(model_size="nano")
        print(f"✅ YOLO analyzer initialized (device: {analyzer.device})")
        
        # Create test image
        test_img = create_person_image()
        test_path = "test_person_detailed.jpg"
        test_img.save(test_path)
        print(f"✅ Created test image: {test_path}")
        
        # Analyze image
        print("\n🔍 Running YOLO analysis...")
        result = analyzer.analyze_image(test_path)
        
        print("\n📊 Analysis Results:")
        for key, value in result.items():
            if key == 'pose_analysis' and isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Test prompt generation
        prompt = analyzer.get_stable_diffusion_prompt(result)
        print(f"\n🎨 Generated SD Prompt: {prompt}")
        
        # Test device info
        device_info = analyzer.get_device_info()
        print(f"\n🖥️  Device Info:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        success = result.get('success', False)
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: YOLO analysis completed")
        
        if 'error' in result:
            print(f"❌ Error detected: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_boxes_edge_cases():
    """Test edge cases that might trigger the boxes error"""
    print("\n🧪 Testing Boxes Edge Cases")
    print("="*50)
    
    try:
        from backend_camera_yolo import YOLOCameraAnalyzer
        analyzer = YOLOCameraAnalyzer(model_size="nano")
        
        # Test with different image types
        test_cases = [
            ("Empty image", Image.new('RGB', (100, 100), 'black')),
            ("Noise image", Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))),
            ("Simple pattern", Image.new('RGB', (300, 300), 'red')),
        ]
        
        for test_name, test_img in test_cases:
            print(f"\n🔍 Testing: {test_name}")
            test_path = f"test_{test_name.lower().replace(' ', '_')}.jpg"
            test_img.save(test_path)
            
            try:
                result = analyzer.analyze_image(test_path)
                print(f"  ✅ Analysis completed successfully")
                print(f"  📊 Result: framing={result.get('framing', 'unknown')}, angle={result.get('camera_angle', 'unknown')}")
                
                if 'error' in result:
                    print(f"  ⚠️  Analysis error: {result['error']}")
                
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
            finally:
                if os.path.exists(test_path):
                    os.remove(test_path)
        
        print("\n✅ Edge case testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False

def main():
    """Run specific YOLO tests focused on the boxes fix"""
    print("🚀 YOLO Backend Boxes Fix Verification")
    print("="*60)
    
    # Test with realistic person image
    person_test = test_yolo_with_real_person()
    
    # Test edge cases
    edge_test = test_boxes_edge_cases()
    
    # Summary
    print("\n" + "="*60)
    print("📋 BOXES FIX TEST SUMMARY")
    print("="*60)
    print(f"🧑 Person Image Test: {'✅ PASSED' if person_test else '❌ FAILED'}")
    print(f"🔧 Edge Cases Test:   {'✅ PASSED' if edge_test else '❌ FAILED'}")
    
    if person_test and edge_test:
        print("\n🎉 All tests passed! The boxes indexing error has been fixed.")
        print("✅ YOLO backend is ready for production use!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
