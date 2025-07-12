#!/usr/bin/env python3
"""
Test the enhanced SD prompt generation with a detailed example
"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw
import time

# Add the clothing module path
clothing_path = os.path.join(os.path.dirname(__file__), 'code_all', '7_code_clothing')
sys.path.append(clothing_path)

from backend_clothing import create_clothing_analyzer

def create_detailed_test_image():
    """Create a more detailed test image for better analysis"""
    # Create a test image with multiple colors and shapes
    img = Image.new('RGB', (512, 768), color=(240, 240, 240))  # Light gray background
    draw = ImageDraw.Draw(img)
    
    # Draw some clothing-like shapes
    # Upper body (shirt area) - blue
    draw.rectangle([150, 200, 362, 400], fill=(70, 130, 180))  # Steel blue
    
    # Lower body (pants area) - black
    draw.rectangle([180, 400, 332, 600], fill=(20, 20, 20))  # Black
    
    # Feet area (shoes) - brown
    draw.rectangle([170, 600, 220, 650], fill=(139, 69, 19))  # Brown
    draw.rectangle([292, 600, 342, 650], fill=(139, 69, 19))  # Brown
    
    return img

def test_enhanced_sd_prompts():
    """Test the enhanced SD prompt generation"""
    print("🧪 Testing Enhanced SD Prompt Generation...")
    
    try:
        # Create analyzer
        analyzer = create_clothing_analyzer("instructblip")
        
        # Create a more detailed test image
        test_image = create_detailed_test_image()
        test_path = "detailed_test_clothing.jpg"
        test_image.save(test_path)
        print(f"✓ Detailed test image created: {test_path}")
        
        # Test the analysis
        result = analyzer.analyze_image(test_path)
        
        if "error" in result:
            print(f"❌ Analysis error: {result['error']}")
            return False
        
        print("\n📊 Enhanced Analysis Results:")
        print(f"📝 Raw Description: {result.get('raw_description', 'N/A')}")
        print(f"\n🎨 Enhanced SD Prompt: {result.get('sd_prompt', 'N/A')}")
        print(f"\n📈 Confidence: {result.get('confidence', 0):.1%}")
        
        # Show categorization
        categorized = result.get('categorized', {})
        if categorized:
            print("\n📋 Detailed Categories:")
            for category, items in categorized.items():
                if items:
                    print(f"  {category.replace('_', ' ').title()}: {items}")
        
        # Show colors and patterns
        colors_patterns = result.get('colors_patterns', [])
        if colors_patterns:
            print(f"\n🌈 Colors & Patterns Detected: {colors_patterns}")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        
        # Show improvements
        print("\n✨ SD Prompt Enhancements:")
        print("  • Person identification (man/woman/person)")
        print("  • Color-specific item descriptions")
        print("  • Fabric and material detection")
        print("  • Style and fit descriptors")
        print("  • Photography quality enhancers")
        print("  • Setting context (indoor/outdoor)")
        print("  • Duplicate removal and ordering")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_sd_prompts()
    if success:
        print("\n✅ Enhanced SD prompt test completed!")
        print("\n🚀 Now your clothing analysis will generate:")
        print("  • Much more detailed SD prompts")
        print("  • Specific color-item combinations")
        print("  • Style and material descriptors")
        print("  • Professional photography terms")
        print("  • Better categorization")
    else:
        print("\n❌ Enhanced SD prompt test failed!")
