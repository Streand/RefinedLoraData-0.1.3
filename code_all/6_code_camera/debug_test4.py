"""
Debug the specific test4 folder creation issue
"""
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import the UI module
sys.path.append(os.getcwd())

from UI_camera import CameraUI

def debug_test4_creation():
    """Debug the specific test4 folder creation"""
    print("🔍 Debugging test4 folder creation\n")
    
    ui = CameraUI()
    folder_name = "test4"
    
    # Create mock processed results
    mock_results = [
        {
            'file_path': 'mock_image.jpg',
            'file_name': 'mock_image.jpg',
            'analysis': {
                'framing': 'medium shot',
                'camera_angle': 'straight on',
                'people_detected': 1,
                'confidence': 0.95,
                'inference_time': 0.1,
                'device': 'cpu'
            }
        }
    ]
    
    # Create a temporary mock image file for testing
    temp_image_path = Path("mock_image.jpg")
    with open(temp_image_path, 'w') as f:
        f.write("mock image content")
    
    # Update the mock results with the actual temp file path
    mock_results[0]['file_path'] = str(temp_image_path.absolute())
    
    print(f"Testing save_batch_results with folder_name: '{folder_name}'")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test the path construction manually first
    print("\n🧪 Testing path construction:")
    current_file_dir = Path(__file__).parent
    project_root = current_file_dir.parent.parent
    base_dir = project_root / "data_storage" / "data_store_camera"
    expected_output_dir = base_dir / folder_name
    
    print(f"Current file dir: {current_file_dir}")
    print(f"Project root: {project_root}")
    print(f"Base dir: {base_dir}")
    print(f"Expected output dir: {expected_output_dir}")
    print(f"Expected output dir (absolute): {expected_output_dir.absolute()}")
    print(f"Base dir exists: {base_dir.exists()}")
    
    try:
        result_msg, folder_path = ui.save_batch_results(mock_results, folder_name)
        print(f"\n📊 Save results:")
        print(f"Result message:\n{result_msg}")
        print(f"Returned folder path: {folder_path}")
        
        # Check if the folder was actually created
        if folder_path:
            print(f"\n🔍 Folder verification:")
            print(f"Folder exists: {os.path.exists(folder_path)}")
            
            # Check the expected location too
            print(f"Expected folder exists: {expected_output_dir.exists()}")
            
            if os.path.exists(folder_path):
                print(f"Folder contents: {list(Path(folder_path).iterdir())}")
            else:
                print("❌ Folder was not created!")
                
                # Check if it was created somewhere else
                print(f"\n🔍 Checking if folder exists in expected location:")
                if expected_output_dir.exists():
                    print(f"✅ Found in expected location: {expected_output_dir}")
                    print(f"Contents: {list(expected_output_dir.iterdir())}")
                else:
                    print(f"❌ Not found in expected location either")
        
    except Exception as e:
        print(f"❌ Error in save_batch_results: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp file
        if temp_image_path.exists():
            temp_image_path.unlink()
            print(f"\n🧹 Temp mock image cleaned up")

if __name__ == "__main__":
    debug_test4_creation()
