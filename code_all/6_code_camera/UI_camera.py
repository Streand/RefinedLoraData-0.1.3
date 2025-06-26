"""
Camera Tab UI for RefinedLoraData
Provides interface for analyzing camera angles and framing in images
"""

import gradio as gr
import os
import sys
from typing import Dict, Any, Optional, Tuple, List
import json

# Add the current directory to path to import backend
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from backend_camera import CameraAnalyzer
    BACKEND_AVAILABLE = True
    CameraAnalyzer_class = CameraAnalyzer
except ImportError as e:
    print(f"Warning: Could not import camera backend: {e}")
    BACKEND_AVAILABLE = False
    CameraAnalyzer_class = None

class CameraUI:
    """UI component for camera analysis functionality"""
    
    def __init__(self):
        self.analyzer = CameraAnalyzer_class() if BACKEND_AVAILABLE and CameraAnalyzer_class is not None else None
        self.backend_available = BACKEND_AVAILABLE
        
        # Camera framing and angle reference data
        self.framing_descriptions = {
            "extreme close-up": "Very tight shot focusing on eyes/face details",
            "close-up": "Head and shoulders visible, intimate view",
            "medium shot": "Waist up, good for dialogue and expressions",
            "full body shot": "Complete figure from head to toe",
            "establishing shot": "Wide view showing environment and context"
        }
        
        self.angle_descriptions = {
            "straight on": "Camera at eye level, direct view",
            "bilaterally symmetrical": "Centered, balanced composition",
            "side view": "Profile view of the subject",
            "back view": "Camera positioned behind the subject",
            "from above": "High angle, camera above subject",
            "from below": "Low angle, camera below subject (hero shot)",
            "wide angle view": "Distorted perspective, wider field of view",
            "fisheye view": "Extreme wide angle with barrel distortion",
            "overhead shot": "Directly above the subject",
            "top down shot": "Bird's eye view perspective",
            "hero view": "Low angle making subject appear powerful",
            "selfie": "Close, personal angle typical of self-portraits"
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
                "Backend not available - missing dependencies (cv2, numpy)",
                "Install required packages: pip install opencv-python numpy",
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
            angle_info = self._get_angle_info(result.get("angle", ""))
            
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
            return "Backend not available - missing dependencies", ""
            
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

    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format single image analysis result"""
        lines = ["📊 **Image Analysis Results**\n"]
        
        if "framing" in result:
            lines.append(f"🎬 **Framing**: {result['framing']}")
            
        if "angle" in result:
            lines.append(f"📐 **Camera Angle**: {result['angle']}")
            
        if "aspect_ratio" in result:
            lines.append(f"📏 **Aspect Ratio**: {result['aspect_ratio']}")
            
        if "composition_notes" in result:
            lines.append(f"🎨 **Composition**: {result['composition_notes']}")
            
        if "image_dimensions" in result:
            lines.append(f"📐 **Dimensions**: {result['image_dimensions']}")
            
        return "\n".join(lines)

    def _format_batch_results(self, results: Dict[str, Dict]) -> str:
        """Format batch analysis results"""
        lines = [f"📁 **Batch Analysis Results** ({len(results)} images)\n"]
        
        for filename, result in results.items():
            if "error" in result:
                lines.append(f"❌ **{filename}**: Error - {result['error']}")
            else:
                framing = result.get('framing', 'unknown')
                angle = result.get('angle', 'unknown')
                lines.append(f"✅ **{filename}**: {framing}, {angle}")
        
        return "\n".join(lines)

    def _generate_csv_data(self, results: Dict[str, Dict]) -> str:
        """Generate CSV format data for batch results"""
        lines = ["filename,framing,angle,aspect_ratio,composition_notes,dimensions"]
        
        for filename, result in results.items():
            if "error" not in result:
                row = [
                    filename,
                    result.get('framing', ''),
                    result.get('angle', ''),
                    result.get('aspect_ratio', ''),
                    result.get('composition_notes', ''),
                    result.get('image_dimensions', '')
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
            - **Stable Diffusion**: Use the generated prompt components in your SD prompts
            - **Quality**: Higher resolution images generally provide more accurate analysis
            """)
    
    return camera_tab


# For testing the UI independently
if __name__ == "__main__":
    with gr.Blocks(title="Camera Analysis Test") as demo:
        camera_tab = create_camera_tab()
    
    demo.launch(share=False, debug=True)