#!/usr/bin/env python3
"""
Test script to check clothing analysis upload functionality
"""

import sys
import os
from pathlib import Path
from PIL import Image
import time

# Add the clothing module path
clothing_path = os.path.join(os.path.dirname(__file__), 'code_all', '7_code_clothing')
sys.path.append(clothing_path)

try:
    from backend_clothing import create_clothing_analyzer, ClothingAnalyzer
    print("✓ Successfully imported full clothing backend")
    backend_type = "full"
except ImportError as e:
    try:
        from backend_clothing_simple import create_clothing_analyzer, SimpleClothingAnalyzer as ClothingAnalyzer
        print("✓ Successfully imported simple clothing backend")
        backend_type = "simple"
    except ImportError as e2:
        try:
            from backend_clothing_ultra_simple import create_clothing_analyzer, UltraSimpleClothingAnalyzer as ClothingAnalyzer
            print("⚠️ Successfully imported ultra-simple clothing backend")
            backend_type = "ultra_simple"
        except ImportError as e3:
            print(f"❌ No clothing backend available: {e3}")
            sys.exit(1)

def test_clothing_analysis():
    """Test the clothing analysis functionality"""
    print(f"\n🧪 Testing {backend_type} clothing backend...")
    
    try:
        # Create analyzer
        print("Creating analyzer...")
        analyzer = create_clothing_analyzer("instructblip")
        print(f"✓ Analyzer created: {type(analyzer)}")
        
        # Create a test image
        print("Creating test image...")
        test_image = Image.new('RGB', (512, 512), color=(255, 200, 200))  # Light pink
        test_path = "test_clothing_image.jpg"
        test_image.save(test_path)
        print(f"✓ Test image saved: {test_path}")
        
        # Test the analysis
        print("Running analysis...")
        result = analyzer.analyze_image(test_path)
        print(f"✓ Analysis completed")
        print(f"Result keys: {list(result.keys())}")
        
        # Check for errors
        if "error" in result:
            print(f"❌ Analysis error: {result['error']}")
            return False
        
        # Print results
        print("\n📊 Analysis Results:")
        print(f"Raw description: {result.get('raw_description', 'N/A')}")
        print(f"SD prompt: {result.get('sd_prompt', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Model used: {result.get('model_used', 'N/A')}")
        
        categorized = result.get('categorized', {})
        if categorized:
            print("\nCategories:")
            for category, items in categorized.items():
                if items:
                    print(f"  {category}: {items}")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
            print(f"✓ Cleaned up test image")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clothing_analysis()
    if success:
        print("\n✅ Clothing analysis test passed!")
    else:
        print("\n❌ Clothing analysis test failed!")
