"""
Camera Tab UI for RefinedLoraData
Provides interface for analyzing camera angles and framing in images
"""

import gradio as gr
import os
import sys
from typing import Dict, Any, Optional, Tuple, List
import json
import shutil
from pathlib import Path
import time

# Add the current directory to path to import backend
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from backend_camera_yolo import YOLOCameraAnalyzer
    BACKEND_AVAILABLE = True
    CameraAnalyzer_class = YOLOCameraAnalyzer
except ImportError as e:
    print(f"Warning: Could not import YOLO camera backend: {e}")
    BACKEND_AVAILABLE = False
    CameraAnalyzer_class = None

class CameraUI:
    """UI component for camera analysis functionality"""
    
    def __init__(self):
        self.analyzer = CameraAnalyzer_class() if BACKEND_AVAILABLE and CameraAnalyzer_class is not None else None
        self.backend_available = BACKEND_AVAILABLE
        
        # Camera framing and angle reference data
        self.framing_descriptions = {
            "extreme close-up": "Very tight shot focusing on eyes/face details only",
            "close-up": "Head and shoulders visible, dialogue and portrait scenes",
            "medium shot": "Waist up, upper body shots for expressions and gestures", 
            "cowboy shot": "Mid-thigh up, action scenes and western-style shots",
            "full body shot": "Complete figure from head to toe, fashion and dance",
            "establishing shot": "Wide view showing subject in environment context",
            "unknown": "Framing type could not be determined"
        }
        
        self.angle_descriptions = {
            "straight on": "Camera at eye level, direct frontal view",
            "bilaterally symmetrical": "Centered, balanced composition with slight angle",
            "side view": "Profile view of the subject from the side",
            "unknown": "Camera angle could not be determined"
        }

    def analyze_single_image(self, image) -> Tuple[str, str, str, str]:
        """
        Analyze a single uploaded image
        
        Args:
            image: Gradio image input
            
        Returns:
            Tuple of (analysis_result, framing_info, angle_info, sd_prompt)
        """
        if not BACKEND_AVAILABLE:
            return (
                "YOLO backend not available - missing dependencies",
                "Install required packages: pip install torch torchvision ultralytics",
                "",
                ""
            )
            
        if image is None:
            return "No image uploaded", "", "", ""
            
        try:
            # Save temporary image for analysis
            temp_path = "temp_analysis_image.jpg"
            image.save(temp_path)
            
            # Analyze the image
            if self.analyzer is not None:
                result = self.analyzer.analyze_image(temp_path)
            else:
                return "Backend not available", "", "", ""
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if "error" in result:
                return f"Error: {result['error']}", "", "", ""
            
            # Format results
            analysis_text = self._format_analysis_result(result)
            framing_info = self._get_framing_info(result.get("framing", ""))
            angle_info = self._get_angle_info(result.get("camera_angle", ""))
            
            if self.analyzer is not None:
                sd_prompt = self.analyzer.get_stable_diffusion_prompt(result)
            else:
                sd_prompt = ""
            
            return analysis_text, framing_info, angle_info, sd_prompt
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}", "", "", ""

    def analyze_batch_images(self, folder_path: str) -> Tuple[str, str]:
        """
        Analyze multiple images in a folder
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Tuple of (results_text, csv_data)
        """
        if not BACKEND_AVAILABLE:
            return "YOLO backend not available - missing dependencies", ""
            
        if not folder_path or not os.path.exists(folder_path):
            return "Invalid folder path", ""
            
        try:
            if self.analyzer is not None:
                results = self.analyzer.analyze_batch(folder_path)
            else:
                return "Backend not available", ""
            
            if "error" in results:
                return f"Error: {results['error']}", ""
            
            # Format results
            results_text = self._format_batch_results(results)
            csv_data = self._generate_csv_data(results)
            
            return results_text, csv_data
            
        except Exception as e:
            return f"Error in batch analysis: {str(e)}", ""

    def process_batch_upload(self, files: List[Any]) -> Tuple[str, List[Dict]]:
        """
        Process uploaded batch of images for camera analysis
        
        Args:
            files: List of uploaded files from Gradio
            
        Returns:
            Tuple of (status_message, processed_results)
        """
        if not BACKEND_AVAILABLE:
            return "❌ YOLO backend not available", []
            
        if not files or len(files) == 0:
            return "📁 No images uploaded", []
        
        try:
            processed_results = []
            total_files = len(files)
            successful_count = 0
            failed_files = []
            
            for i, file in enumerate(files):
                # Update progress
                progress_msg = f"Processing {i+1} of {total_files} images..."
                
                try:
                    # Analyze the image
                    if self.analyzer is not None:
                        result = self.analyzer.analyze_image(file.name)
                    else:
                        failed_files.append(f"{os.path.basename(file.name)} (backend not available)")
                        continue
                    
                    if "error" in result and result.get("success", False) is False:
                        failed_files.append(f"{os.path.basename(file.name)} ({result['error']})")
                        continue
                    
                    # Store successful result with file info
                    processed_results.append({
                        'file_path': file.name,
                        'file_name': os.path.basename(file.name),
                        'analysis': result
                    })
                    successful_count += 1
                    
                except Exception as e:
                    failed_files.append(f"{os.path.basename(file.name)} (processing error: {str(e)})")
                    continue
            
            # Generate summary message
            status_msg = f"✅ Batch processing complete!\n"
            status_msg += f"📊 Successfully processed: {successful_count}/{total_files} images\n"
            
            if failed_files:
                status_msg += f"❌ Failed images:\n"
                for failed in failed_files[:5]:  # Show max 5 failed files
                    status_msg += f"   • {failed}\n"
                if len(failed_files) > 5:
                    status_msg += f"   • ... and {len(failed_files) - 5} more\n"
            
            status_msg += f"\n💾 Ready to save {successful_count} analyzed images"
            
            return status_msg, processed_results
            
        except Exception as e:
            return f"❌ Batch processing error: {str(e)}", []

    def save_batch_results(self, processed_results: List[Dict], folder_name: str) -> Tuple[str, str]:
        """
        Save batch analysis results to data storage
        
        Args:
            processed_results: List of processed image results
            folder_name: Custom folder name for saving
            
        Returns:
            Tuple of (status_message, output_folder_path)
        """
        if not processed_results:
            return "❌ No processed results to save", ""
            
        if not folder_name or folder_name.strip() == "":
            return "❌ Please enter a folder name", ""
        
        # Clean folder name (remove invalid characters)
        clean_folder_name = "".join(c for c in folder_name.strip() if c.isalnum() or c in (' ', '-', '_')).strip()
        if not clean_folder_name:
            return "❌ Invalid folder name. Use only letters, numbers, spaces, hyphens, and underscores.", ""
        
        try:
            # Create output directory - use absolute path to project root
            current_file_dir = Path(__file__).parent  # 6_code_camera directory
            project_root = current_file_dir.parent.parent  # Go up two levels to project root
            base_dir = project_root / "data_storage" / "data_store_camera"
            output_dir = base_dir / clean_folder_name
            
            # Check if directory already exists
            if output_dir.exists():
                timestamp = int(time.time())
                output_dir = base_dir / f"{clean_folder_name}_{timestamp}"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            failed_saves = []
            
            for i, result_data in enumerate(processed_results, 1):
                try:
                    file_path = result_data['file_path']
                    original_name = result_data['file_name']
                    analysis = result_data['analysis']
                    
                    # Get file extension
                    file_ext = Path(original_name).suffix.lower()
                    
                    # Generate new filenames
                    base_name = f"{clean_folder_name}-camera-{i}"
                    image_name = f"{base_name}{file_ext}"
                    txt_name = f"{base_name}.txt"
                    full_txt_name = f"{base_name}full.txt"
                    
                    # Copy image file
                    shutil.copy2(file_path, output_dir / image_name)
                    
                    # Generate SD prompt for txt file
                    if self.analyzer is not None:
                        sd_prompt = self.analyzer.get_stable_diffusion_prompt(analysis)
                    else:
                        sd_prompt = "Analysis not available"
                    
                    # Write simple txt file (SD prompt)
                    with open(output_dir / txt_name, 'w', encoding='utf-8') as f:
                        f.write(sd_prompt)
                    
                    # Write full analysis txt file
                    with open(output_dir / full_txt_name, 'w', encoding='utf-8') as f:
                        f.write("=== YOLO Camera Analysis Results ===\n\n")
                        
                        # Basic analysis
                        f.write(f"Framing: {analysis.get('framing', 'unknown')}\n")
                        f.write(f"Camera Angle: {analysis.get('camera_angle', 'unknown')}\n")
                        f.write(f"People Detected: {analysis.get('people_detected', 0)}\n")
                        f.write(f"Confidence: {analysis.get('confidence', 0.0):.3f}\n")
                        f.write(f"Analysis Time: {analysis.get('inference_time', 0.0):.3f}s\n")
                        f.write(f"Device: {analysis.get('device', 'unknown')}\n")
                        
                        # Pose analysis if available
                        if 'pose_analysis' in analysis and analysis['pose_analysis']:
                            pose = analysis['pose_analysis']
                            f.write(f"\n=== Pose Analysis ===\n")
                            f.write(f"Visible Keypoints: {pose.get('visible_keypoints', 0)}/17\n")
                            f.write(f"Symmetry Score: {pose.get('symmetry_score', 0.0):.3f}\n")
                            f.write(f"Face Visibility: {pose.get('face_visibility', 'unknown')}\n")
                            f.write(f"Body Orientation: {pose.get('body_orientation', 'unknown')}\n")
                        
                        # Generated SD prompt
                        f.write(f"\n=== Stable Diffusion Prompt ===\n")
                        f.write(sd_prompt)
                        
                        # Raw analysis data (as JSON)
                        f.write(f"\n\n=== Raw Analysis Data (JSON) ===\n")
                        f.write(json.dumps(analysis, indent=2))
                    
                    saved_count += 1
                    
                except Exception as e:
                    file_ref = result_data.get('file_name', f'image_{i}') if 'file_name' in result_data else f'image_{i}'
                    failed_saves.append(f"{file_ref} ({str(e)})")
                    continue
            
            # Generate success message
            success_msg = f"✅ Successfully saved {saved_count} images and analysis files!\n\n"
            success_msg += f"📁 Saved to: {output_dir.absolute()}\n\n"
            success_msg += f"📋 Files created per image:\n"
            success_msg += f"   • [name]-camera-[num].jpg/png - Original image\n"
            success_msg += f"   • [name]-camera-[num].txt - SD prompt\n"
            success_msg += f"   • [name]-camera-[num]full.txt - Complete analysis\n"
            
            if failed_saves:
                success_msg += f"\n❌ Failed to save:\n"
                for failed in failed_saves:
                    success_msg += f"   • {failed}\n"
            
            return success_msg, str(output_dir.absolute())
            
        except Exception as e:
            return f"❌ Save error: {str(e)}", ""

    def open_saved_folder(self, folder_path: str) -> str:
        """
        Open the saved folder in Windows Explorer
        
        Args:
            folder_path: Path to the folder to open
            
        Returns:
            Status message
        """
        if not folder_path or not os.path.exists(folder_path):
            return "❌ Folder path not found"
        
        try:
            import subprocess
            import platform
            
            # Convert to absolute path and normalize
            abs_path = os.path.abspath(folder_path)
            
            system = platform.system()
            if system == "Windows":
                # Use Windows explorer - try both methods for reliability
                subprocess.run(['explorer', '/select,', abs_path], check=False)
                subprocess.run(['explorer', abs_path], check=False)
                return f"✅ Opened folder in Windows Explorer: {abs_path}"
            elif system == "Darwin":  # macOS
                subprocess.run(['open', abs_path], check=True)
                return f"✅ Opened folder in Finder: {abs_path}"
            else:  # Linux
                subprocess.run(['xdg-open', abs_path], check=True)
                return f"✅ Opened folder: {abs_path}"
                
        except Exception as e:
            return f"❌ Could not open folder: {str(e)}"

    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format single image analysis result for YOLO backend"""
        lines = ["📊 **YOLO Image Analysis Results**\n"]
        
        if "framing" in result:
            lines.append(f"🎬 **Framing**: {result['framing']}")
            
        if "camera_angle" in result:
            lines.append(f"📐 **Camera Angle**: {result['camera_angle']}")
            
        if "people_detected" in result:
            lines.append(f"� **People Detected**: {result['people_detected']}")
            
        if "confidence" in result:
            lines.append(f"� **Confidence**: {result['confidence']:.2f}")
            
        if "inference_time" in result:
            lines.append(f"⏱️ **Analysis Time**: {result['inference_time']:.2f}s")
            
        if "device" in result:
            lines.append(f"🔧 **Device**: {result['device']}")
            
        # Add pose analysis if available
        if 'pose_analysis' in result and result['pose_analysis']:
            pose = result['pose_analysis']
            lines.append("\n🦴 **Pose Analysis**:")
            if 'visible_keypoints' in pose:
                lines.append(f"  - Visible Keypoints: {pose['visible_keypoints']}/17")
            if 'symmetry_score' in pose:
                lines.append(f"  - Symmetry Score: {pose['symmetry_score']:.2f}")
            if 'face_visibility' in pose:
                lines.append(f"  - Face Visibility: {pose['face_visibility']}")
            if 'body_orientation' in pose:
                lines.append(f"  - Body Orientation: {pose['body_orientation']}")
            
        return "\n".join(lines)

    def _format_batch_results(self, results: Dict[str, Dict]) -> str:
        """Format batch analysis results for YOLO backend"""
        lines = [f"📁 **Batch Analysis Results** ({len(results)} images)\n"]
        
        for filename, result in results.items():
            if "error" in result:
                lines.append(f"❌ **{filename}**: Error - {result['error']}")
            else:
                framing = result.get('framing', 'unknown')
                camera_angle = result.get('camera_angle', 'unknown')
                confidence = result.get('confidence', 0.0)
                lines.append(f"✅ **{filename}**: {framing}, {camera_angle} (conf: {confidence:.2f})")
        
        return "\n".join(lines)

    def _generate_csv_data(self, results: Dict[str, Dict]) -> str:
        """Generate CSV format data for batch results with YOLO backend"""
        lines = ["filename,framing,camera_angle,confidence,people_detected,device,inference_time"]
        
        for filename, result in results.items():
            if "error" not in result:
                row = [
                    filename,
                    result.get('framing', ''),
                    result.get('camera_angle', ''),
                    str(result.get('confidence', '')),
                    str(result.get('people_detected', '')),
                    result.get('device', ''),
                    str(result.get('inference_time', ''))
                ]
                lines.append(",".join(row))
        
        return "\n".join(lines)

    def _get_framing_info(self, framing: str) -> str:
        """Get detailed information about the detected framing"""
        if framing in self.framing_descriptions:
            return f"**{framing}**: {self.framing_descriptions[framing]}"
        return f"**{framing}**: Camera framing type"

    def _get_angle_info(self, angle: str) -> str:
        """Get detailed information about the detected angle"""
        if angle in self.angle_descriptions:
            return f"**{angle}**: {self.angle_descriptions[angle]}"
        return f"**{angle}**: Camera angle type"

    def create_reference_guide(self) -> str:
        """Create a reference guide for camera terms"""
        lines = ["# 📚 Camera Analysis Reference Guide\n"]
        
        lines.append("## 🎬 Camera Framing Types")
        for framing, desc in self.framing_descriptions.items():
            lines.append(f"- **{framing}**: {desc}")
        
        lines.append("\n## 📐 Camera Angle Types")
        for angle, desc in self.angle_descriptions.items():
            lines.append(f"- **{angle}**: {desc}")
            
        lines.append("\n## 🎯 Usage for Stable Diffusion")
        lines.append("These terms can be used in your SD prompts to control composition:")
        lines.append("- Use framing terms to control how much of the subject is visible")
        lines.append("- Use angle terms to control the camera's position relative to the subject")
        lines.append("- Combine with weights like `(medium shot:1.2)` for stronger effect")
        lines.append("- Works best with portrait aspect ratios for character shots")
        
        return "\n".join(lines)


def create_camera_tab() -> gr.Tab:
    """Create the Camera analysis tab interface"""
    
    ui = CameraUI()
    
    with gr.Tab("Camera") as camera_tab:
        gr.Markdown("# 📸 Camera Angle & Framing Analysis")
        gr.Markdown("Analyze images to determine camera framing and angles for Stable Diffusion prompts")
        
        with gr.Tabs():
            # Single Image Analysis Tab
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image for Analysis",
                            type="pil",
                            height=400
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyze Camera Angle",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        analysis_output = gr.Markdown(
                            label="Analysis Results",
                            value="Upload an image and click analyze to see results"
                        )
                        
                        framing_info = gr.Markdown(
                            label="Framing Details",
                            value=""
                        )
                        
                        angle_info = gr.Markdown(
                            label="Angle Details", 
                            value=""
                        )
                        
                        sd_prompt_output = gr.Textbox(
                            label="Stable Diffusion Prompt Components",
                            placeholder="Generated prompt components will appear here",
                            lines=2
                        )
                
                # Wire up the single image analysis
                analyze_btn.click(
                    fn=ui.analyze_single_image,
                    inputs=[image_input],
                    outputs=[analysis_output, framing_info, angle_info, sd_prompt_output]
                )
            
            # Batch Analysis Tab
            with gr.TabItem("Batch Analysis"):
                with gr.Row():
                    with gr.Column():
                        folder_input = gr.Textbox(
                            label="Image Folder Path",
                            placeholder="Enter path to folder containing images",
                            lines=1
                        )
                        batch_analyze_btn = gr.Button(
                            "📁 Analyze Batch",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        batch_results = gr.Markdown(
                            label="Batch Results",
                            value="Enter a folder path and click analyze"
                        )
                        
                        csv_output = gr.Textbox(
                            label="CSV Export Data",
                            placeholder="CSV data will appear here",
                            lines=10,
                            max_lines=20
                        )
                
                # Wire up batch analysis
                batch_analyze_btn.click(
                    fn=ui.analyze_batch_images,
                    inputs=[folder_input],
                    outputs=[batch_results, csv_output]
                )

            # Batch Upload & Save Tab
            with gr.TabItem("Batch Upload & Save"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Images")
                        
                        batch_upload = gr.File(
                            label="Select Multiple Images",
                            file_count="multiple",
                            file_types=["image"],
                            height=200
                        )
                        
                        upload_btn = gr.Button(
                            "🔍 Analyze Uploaded Images",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Processing status
                        processing_status = gr.Markdown(
                            label="Processing Status",
                            value="📁 Upload images above and click analyze"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💾 Save Results")
                        
                        folder_name_input = gr.Textbox(
                            label="Folder Name",
                            placeholder="e.g., person1, character_set_a, etc.",
                            lines=1,
                            info="Files will be saved as: [name]-camera-1.jpg, [name]-camera-1.txt, etc."
                        )
                        
                        save_btn = gr.Button(
                            "💾 Save Analysis Results",
                            variant="secondary",
                            size="lg",
                            interactive=False
                        )
                        
                        open_folder_btn = gr.Button(
                            "📁 Open Saved Folder",
                            variant="secondary",
                            size="lg",
                            interactive=False,
                            visible=False
                        )
                        
                        save_status = gr.Markdown(
                            label="Save Status",
                            value="🔄 Process images first, then save results"
                        )
                
                # Hidden state to store processed results and folder path
                processed_data = gr.State([])
                saved_folder_path = gr.State("")
                
                # Wire up batch upload processing
                def process_and_update_ui(files):
                    status, results = ui.process_batch_upload(files)
                    # Enable save button if we have results
                    save_enabled = len(results) > 0
                    return status, results, gr.update(interactive=save_enabled)
                
                upload_btn.click(
                    fn=process_and_update_ui,
                    inputs=[batch_upload],
                    outputs=[processing_status, processed_data, save_btn]
                )
                
                # Wire up save functionality
                def save_and_update_ui(results, folder_name):
                    save_msg, folder_path = ui.save_batch_results(results, folder_name)
                    # Show open folder button if save was successful
                    open_btn_visible = "✅ Successfully saved" in save_msg
                    open_btn_interactive = bool(folder_path)
                    return (save_msg, 
                            gr.update(interactive=False), 
                            folder_path,
                            gr.update(visible=open_btn_visible, interactive=open_btn_interactive))
                
                save_btn.click(
                    fn=save_and_update_ui,
                    inputs=[processed_data, folder_name_input],
                    outputs=[save_status, save_btn, saved_folder_path, open_folder_btn]
                )
                
                # Wire up open folder functionality
                def open_folder_and_update(folder_path):
                    open_msg = ui.open_saved_folder(folder_path)
                    return open_msg
                
                open_folder_btn.click(
                    fn=open_folder_and_update,
                    inputs=[saved_folder_path],
                    outputs=[save_status]
                )
            
            # Reference Guide Tab
            with gr.TabItem("Reference Guide"):
                reference_content = gr.Markdown(
                    value=ui.create_reference_guide()
                )
        
        # Status and tips
        with gr.Row():
            gr.Markdown("""
            ## 💡 Tips for Best Results
            - **Single Analysis**: Upload clear images with visible subjects
            - **Batch Analysis**: Ensure folder contains only image files (.jpg, .png, etc.)
            - **Batch Upload & Save**: Upload multiple images, analyze them, then save with custom naming
            - **Stable Diffusion**: Use the generated prompt components in your SD prompts
            - **Quality**: Higher resolution images generally provide more accurate analysis
            - **File Organization**: Saved files will be organized in data_storage/data_store_camera/[your_folder_name]/
            """)
    
    return camera_tab


# For testing the UI independently
if __name__ == "__main__":
    with gr.Blocks(title="Camera Analysis Test") as demo:
        camera_tab = create_camera_tab()
    
    demo.launch(share=False, debug=True)