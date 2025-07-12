#!/usr/bin/env python3
"""Test import of backend_clothing module"""

import sys
import os
import traceback

print("Testing backend_clothing import...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    # Add current directory to path
    sys.path.insert(0, '.')
    
    # Test blackwell_support import
    print("\n1. Testing blackwell_support import...")
    import blackwell_support
    print("✅ blackwell_support imported successfully")
    
    # Test clothing backend import
    print("\n2. Testing backend_clothing import...")
    sys.path.insert(0, './code_all/7_code_clothing')
    import backend_clothing
    print("✅ backend_clothing imported successfully")
    
    # Test creating analyzer
    print("\n3. Testing analyzer creation...")
    analyzer = backend_clothing.create_clothing_analyzer()
    print("✅ Analyzer created successfully")
    
    # Test device info
    print("\n4. Testing device info...")
    device_info = analyzer.get_device_info()
    print(f"✅ Device info: {device_info}")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()
