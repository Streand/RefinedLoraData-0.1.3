"""
Launch the main UI to test camera integration
"""

import sys
import os

# Add the main app path
main_app_path = os.path.join(os.path.dirname(__file__), '..', '1_code_main_app')
sys.path.append(main_app_path)

from UI_main import create_main_ui

if __name__ == "__main__":
    print("🚀 Launching RefinedLoraData with Camera Analysis")
    print("📸 Camera tab includes both OpenCV and YOLO backends")
    print("🔧 Choose 'OpenCV (Fast)' for quick analysis")
    print("🚀 Choose 'YOLO (Accurate)' for detailed pose detection")
    print("\n🌐 Opening web interface...")
    
    interface = create_main_ui()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
