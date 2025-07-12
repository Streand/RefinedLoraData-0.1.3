"""
Clothing Analysis UI for RefinedLoraData
Provides interface for analyzing clothing in images using InstructBLIP and BLIP-2
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

# Import priority: Full backend -> Simple backend -> Ultra-simple backend
create_clothing_analyzer = None
ClothingAnalyzer = None

try:
    from backend_clothing import ClothingAnalyzer, create_clothing_analyzer
    BACKEND_AVAILABLE = True
    BACKEND_TYPE = "full"
    print("✓ Using full clothing backend (InstructBLIP + BLIP-2)")
except ImportError as e:
    try:
        from backend_clothing_simple import create_clothing_analyzer, SimpleClothingAnalyzer as ClothingAnalyzer
        BACKEND_AVAILABLE = True
        BACKEND_TYPE = "simple"
        print("✓ Using simplified clothing backend (BLIP-2 only)")
    except ImportError as e2:
        try:
            from backend_clothing_ultra_simple import create_clothing_analyzer, UltraSimpleClothingAnalyzer as ClothingAnalyzer
            BACKEND_AVAILABLE = True
            BACKEND_TYPE = "ultra_simple"
            print("⚠️ Using ultra-simple clothing backend (basic descriptions)")
        except ImportError as e3:
            print(f"Warning: Could not import any clothing backend: {e3}")
            BACKEND_AVAILABLE = False
            BACKEND_TYPE = "none"
            ClothingAnalyzer = None

class ClothingUI:
    """UI component for clothing analysis functionality"""
    
    def __init__(self):
        self.analyzer_instructblip = None
        self.analyzer_blip2 = None
        self.backend_available = BACKEND_AVAILABLE
        self.current_model = "instructblip"
        
        # Log which backend is being used
        if BACKEND_AVAILABLE:
            print(f"Clothing backend type: {BACKEND_TYPE}")
        
        # Clothing categories for reference
        self.clothing_categories = {
            "Upper Body": ["shirt", "t-shirt", "blouse", "sweater", "hoodie", "jacket", "blazer", "coat"],
            "Lower Body": ["pants", "jeans", "trousers", "shorts", "skirt", "dress", "leggings"],
            "Footwear": ["shoes", "sneakers", "boots", "sandals", "heels", "flats", "loafers"],
            "Accessories": ["hat", "scarf", "belt", "bag", "jewelry", "watch", "glasses"],
            "Styles": ["casual", "formal", "business", "streetwear", "vintage", "athletic"]
        }
    
    def get_analyzer(self, model_name: str):
        """Get or create analyzer for specified model"""
        global create_clothing_analyzer
        
        if not BACKEND_AVAILABLE or create_clothing_analyzer is None:
            return None
            
        try:
            if model_name == "instructblip":
                if self.analyzer_instructblip is None:
                    self.analyzer_instructblip = create_clothing_analyzer("instructblip")
                return self.analyzer_instructblip
            elif model_name == "blip2":
                if self.analyzer_blip2 is None:
                    self.analyzer_blip2 = create_clothing_analyzer("blip2")
                return self.analyzer_blip2
            else:
                return None
        except Exception as e:
            print(f"Error creating analyzer: {e}")
            return None
    
    def analyze_single_image(self, image, model_choice: str) -> Tuple[str, str, str, str, str]:
        """
        Analyze a single uploaded image for clothing
        
        Args:
            image: Gradio image input
            model_choice: Selected model ("InstructBLIP" or "BLIP-2")
            
        Returns:
            Tuple of (analysis_result, categorized_info, sd_prompt, confidence_info, model_info)
        """
        if not BACKEND_AVAILABLE:
            return (
                "❌ Clothing analysis backend not available",
                "Please install required packages:\n```\npip install transformers torch pillow\n```",
                "",
                "",
                ""
            )
            
        if image is None:
            return "No image uploaded", "", "", "", ""
        
        try:
            # Map UI choice to model name
            model_name = "instructblip" if model_choice == "InstructBLIP" else "blip2"
            
            # Get analyzer
            analyzer = self.get_analyzer(model_name)
            if analyzer is None:
                return f"❌ Could not initialize {model_choice} model", "", "", "", ""
            
            # Save temporary image for analysis
            temp_path = f"temp_clothing_analysis_{int(time.time())}.jpg"
            image.save(temp_path)
            
            # Analyze the image
            result = analyzer.analyze_image(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if "error" in result:
                return f"❌ Error: {result['error']}", "", "", "", ""
            
            # Format results
            analysis_text = self._format_analysis_result(result)
            categorized_info = self._format_categorized_info(result.get("categorized", {}))
            sd_prompt = result.get("sd_prompt", "")
            confidence_info = f"Confidence: {result.get('confidence', 0):.2%}"
            model_info = f"Model: {result.get('model_used', 'unknown').upper()}"
            
            return analysis_text, categorized_info, sd_prompt, confidence_info, model_info
            
        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}", "", "", "", ""
    
    def analyze_batch_images(self, files: List, model_choice: str) -> Tuple[str, List]:
        """
        Analyze multiple images in batch
        
        Args:
            files: List of uploaded files
            model_choice: Selected model
            
        Returns:
            Tuple of (status_message, results_list)
        """
        if not BACKEND_AVAILABLE:
            return "❌ Clothing analysis backend not available", []
        
        if not files:
            return "No files uploaded", []
        
        try:
            # Map UI choice to model name
            model_name = "instructblip" if model_choice == "InstructBLIP" else "blip2"
            
            # Get analyzer
            analyzer = self.get_analyzer(model_name)
            if analyzer is None:
                return f"❌ Could not initialize {model_choice} model", []
            
            results = []
            successful_count = 0
            failed_files = []
            
            for file in files:
                try:
                    # Analyze each image
                    result = analyzer.analyze_image(file.name)
                    
                    if "error" in result:
                        failed_files.append(f"{os.path.basename(file.name)} (Error: {result['error']})")
                        continue
                    
                    # Store results for batch save
                    filename = os.path.basename(file.name)
                    results.append({
                        'filename': filename,
                        'original_path': file.name,
                        'analysis': result
                    })
                    
                    successful_count += 1
                    
                except Exception as e:
                    failed_files.append(f"{os.path.basename(file.name)} (Error: {str(e)})")
            
            # Create status message
            status_msg = f"✅ Analyzed {successful_count} images successfully"
            if failed_files:
                status_msg += f"\n❌ Failed: {len(failed_files)} files"
                status_msg += "\n" + "\n".join(failed_files[:5])  # Show first 5 failures
                if len(failed_files) > 5:
                    status_msg += f"\n... and {len(failed_files) - 5} more"
            
            if successful_count > 0:
                status_msg += f"\n💾 Ready to save {successful_count} analyzed images"
            
            return status_msg, results
            
        except Exception as e:
            return f"❌ Batch analysis error: {str(e)}", []
    
    def save_batch_results(self, results: List, folder_name: str) -> Tuple[str, str]:
        """
        Save batch analysis results to data_store_clothing
        
        Args:
            results: List of analysis results
            folder_name: Name for the save folder
            
        Returns:
            Tuple of (status_message, saved_folder_path)
        """
        if not results:
            return "No results to save", ""
        
        if not folder_name.strip():
            folder_name = f"clothing_batch_{int(time.time())}"
        
        try:
            # Create save directory
            base_dir = os.path.join("..", "..", "data_storage", "data_store_clothing")
            save_dir = os.path.join(base_dir, folder_name)
            
            # Ensure absolute path
            save_dir = os.path.abspath(save_dir)
            
            # Check if directory already exists
            if os.path.exists(save_dir):
                return f"❌ Folder '{folder_name}' already exists", ""
            
            os.makedirs(save_dir, exist_ok=True)
            
            saved_count = 0
            
            for i, result in enumerate(results, 1):
                try:
                    filename = result['filename']
                    analysis = result['analysis']
                    original_path = result['original_path']
                    
                    # Generate save names
                    base_name = os.path.splitext(filename)[0]
                    
                    # Copy image
                    image_save_path = os.path.join(save_dir, f"{base_name}_clothing.jpg")
                    shutil.copy2(original_path, image_save_path)
                    
                    # Save simple description (SD prompt)
                    txt_save_path = os.path.join(save_dir, f"{base_name}_clothing.txt")
                    with open(txt_save_path, 'w', encoding='utf-8') as f:
                        f.write(analysis.get('sd_prompt', ''))
                    
                    # Save full analysis data
                    full_save_path = os.path.join(save_dir, f"{base_name}_clothingfull.txt")
                    with open(full_save_path, 'w', encoding='utf-8') as f:
                        full_data = {
                            'filename': filename,
                            'model_used': analysis.get('model_used', 'unknown'),
                            'raw_description': analysis.get('raw_description', ''),
                            'categorized': analysis.get('categorized', {}),
                            'colors_patterns': analysis.get('colors_patterns', []),
                            'confidence': analysis.get('confidence', 0),
                            'sd_prompt': analysis.get('sd_prompt', ''),
                            'timestamp': analysis.get('timestamp', time.time())
                        }
                        json.dump(full_data, f, indent=2, ensure_ascii=False)
                    
                    saved_count += 1
                    
                except Exception as e:
                    print(f"Error saving result {i}: {e}")
                    continue
            
            status_msg = f"✅ Saved {saved_count} clothing analyses to '{folder_name}'"
            return status_msg, save_dir
            
        except Exception as e:
            return f"❌ Error saving results: {str(e)}", ""
    
    def open_saved_folder(self, folder_path: str) -> str:
        """Open the saved folder in file explorer"""
        if not folder_path or not os.path.exists(folder_path):
            return "❌ No valid folder path to open"
        
        try:
            # Open folder in Windows Explorer
            import subprocess
            subprocess.Popen(f'explorer "{folder_path}"')
            return f"✅ Opened folder: {os.path.basename(folder_path)}"
        except Exception as e:
            return f"❌ Could not open folder: {str(e)}"
    
    def get_gpu_status(self) -> str:
        """Get GPU status information for display in UI"""
        if not BACKEND_AVAILABLE:
            return "❌ **Backend Status:** Clothing analysis backend not available\n\n🔧 **Fix:** Install required packages:\n```\npip install transformers torch pillow\n```"
        
        try:
            # Get info from any available analyzer or create a temporary one
            analyzer = self.get_analyzer("blip2")  # Use BLIP-2 for faster status check
            if analyzer is None:
                return "⚠️ **Status:** Could not initialize models for status check"
            
            device_info = analyzer.get_device_info()
            
            status_lines = []
            
            # Basic device info
            device = device_info.get('device', 'unknown')
            if device == 'cuda':
                status_lines.append("✅ **GPU Acceleration:** Enabled")
            else:
                status_lines.append("⚠️ **GPU Acceleration:** Disabled (using CPU)")
            
            # Model info
            model_name = device_info.get('model_name', 'unknown')
            model_initialized = device_info.get('model_initialized', False)
            status_lines.append(f"🤖 **Model:** {model_name.upper()} ({'Ready' if model_initialized else 'Not initialized'})")
            
            # PyTorch info
            if device_info.get('torch_available', False):
                pytorch_version = device_info.get('pytorch_version', 'unknown')
                status_lines.append(f"🐍 **PyTorch:** {pytorch_version}")
            
            # GPU details if available
            if device_info.get('cuda_available', False):
                gpu_name = device_info.get('gpu_name', 'Unknown GPU')
                compute_cap = device_info.get('compute_capability', 'unknown')
                memory = device_info.get('gpu_memory', 0)
                
                status_lines.append(f"🎮 **GPU:** {gpu_name}")
                status_lines.append(f"📊 **Compute Capability:** {compute_cap}")
                status_lines.append(f"💾 **GPU Memory:** {memory:.1f} GB")
                
                # Blackwell support status
                is_blackwell = device_info.get('is_blackwell', False)
                is_rtx_5000 = device_info.get('is_rtx_5000_series', False)
                
                if is_blackwell:
                    status_lines.append("🚀 **Architecture:** Blackwell (RTX 5000 series)")
                    if is_rtx_5000:
                        sm_count = device_info.get('sm_count', 0)
                        status_lines.append(f"⚡ **Streaming Multiprocessors:** ~{sm_count} SMs")
                        if device_info.get('optimization_notes'):
                            status_lines.append(f"🎯 **Optimization:** {device_info['optimization_notes']}")
                    
                    blackwell_support = device_info.get('blackwell_support', False)
                    if blackwell_support:
                        status_lines.append("✅ **Blackwell Support:** Fully supported")
                    else:
                        status_lines.append("⚠️ **Blackwell Support:** Limited support")
                else:
                    status_lines.append("🔧 **Architecture:** Non-Blackwell (Fully supported)")
            
            return "\n".join(status_lines)
            
        except Exception as e:
            return f"❌ **Error getting GPU status:** {str(e)}"
    
    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format analysis result for display"""
        if not result or "error" in result:
            return "No analysis available"
        
        output = []
        output.append("## 👔 Clothing Analysis Results")
        output.append("")
        
        # Raw description
        raw_desc = result.get('raw_description', '')
        if raw_desc:
            output.append("**Detailed Description:**")
            output.append(raw_desc)
            output.append("")
        
        # Colors and patterns
        colors_patterns = result.get('colors_patterns', [])
        if colors_patterns:
            output.append("**Colors & Patterns:**")
            output.append(", ".join(colors_patterns))
            output.append("")
        
        return "\n".join(output)
    
    def _format_categorized_info(self, categorized: Dict[str, List[str]]) -> str:
        """Format categorized clothing information"""
        if not categorized:
            return "No categorization available"
        
        output = []
        output.append("## 📋 Clothing Categories")
        output.append("")
        
        for category, items in categorized.items():
            if items:
                output.append(f"**{category.replace('_', ' ').title()}:**")
                output.append(", ".join(items))
                output.append("")
        
        return "\n".join(output)

def create_clothing_tab() -> gr.Tab:
    """Create the Clothing analysis tab interface"""
    
    ui = ClothingUI()
    
    with gr.Tab("Clothing") as clothing_tab:
        gr.Markdown("# 👔 Clothing Analysis")
        gr.Markdown("Analyze clothing and fashion in images for Stable Diffusion and LoRA training")
        
        # GPU Status Section
        with gr.Accordion("🎮 GPU & System Status", open=False):
            gpu_status_md = gr.Markdown(
                value=ui.get_gpu_status(),
                label="System Status"
            )
            refresh_status_btn = gr.Button(
                "🔄 Refresh Status",
                variant="secondary",
                size="sm"
            )
            refresh_status_btn.click(
                fn=ui.get_gpu_status,
                outputs=gpu_status_md
            )
        
        # Model Selection
        with gr.Row():
            model_choice = gr.Radio(
                choices=["BLIP-2"],
                value="BLIP-2",
                label="🤖 Analysis Model",
                info="Currently using BLIP-2 only (InstructBLIP temporarily disabled due to compatibility issues)"
            )
        
        with gr.Tabs():
            # Single Image Analysis Tab
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image for Clothing Analysis",
                            type="pil",
                            height=400
                        )
                        analyze_btn = gr.Button(
                            "👔 Analyze Clothing",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        analysis_output = gr.Markdown(
                            label="Clothing Analysis",
                            value="Upload an image and click analyze to see clothing details"
                        )
                        
                        categorized_output = gr.Markdown(
                            label="Categorized Items",
                            value=""
                        )
                        
                        with gr.Row():
                            confidence_output = gr.Markdown(
                                label="Confidence",
                                value=""
                            )
                            model_info_output = gr.Markdown(
                                label="Model Used",
                                value=""
                            )
                        
                        sd_prompt_output = gr.Textbox(
                            label="🎨 Stable Diffusion Prompt",
                            placeholder="Generated SD prompt will appear here",
                            lines=3,
                            interactive=True
                        )
                
                # Wire up single image analysis
                analyze_btn.click(
                    fn=ui.analyze_single_image,
                    inputs=[image_input, model_choice],
                    outputs=[analysis_output, categorized_output, sd_prompt_output, confidence_output, model_info_output]
                )
            
            # Batch Analysis Tab
            with gr.TabItem("Batch Analysis"):
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(
                            label="📁 Upload Multiple Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        with gr.Row():
                            batch_analyze_btn = gr.Button(
                                "👔 Analyze Batch",
                                variant="primary",
                                size="lg"
                            )
                        
                        batch_status = gr.Markdown(
                            label="Batch Status",
                            value="Upload images and click 'Analyze Batch' to start"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 💾 Save Results")
                        
                        folder_name_input = gr.Textbox(
                            label="📂 Folder Name",
                            placeholder="Enter folder name for saving results",
                            value=""
                        )
                        
                        with gr.Row():
                            save_batch_btn = gr.Button(
                                "💾 Save Batch Results",
                                variant="secondary",
                                size="lg"
                            )
                            
                            open_folder_btn = gr.Button(
                                "📂 Open Saved Folder",
                                variant="secondary",
                                size="lg"
                            )
                        
                        save_status = gr.Markdown(
                            label="Save Status",
                            value=""
                        )
                
                # Hidden state to store batch results and folder path
                batch_results_state = gr.State([])
                saved_folder_path_state = gr.State("")
                
                # Wire up batch analysis
                batch_analyze_btn.click(
                    fn=ui.analyze_batch_images,
                    inputs=[batch_files, model_choice],
                    outputs=[batch_status, batch_results_state]
                )
                
                # Wire up batch save
                save_batch_btn.click(
                    fn=ui.save_batch_results,
                    inputs=[batch_results_state, folder_name_input],
                    outputs=[save_status, saved_folder_path_state]
                )
                
                # Wire up open folder
                open_folder_btn.click(
                    fn=ui.open_saved_folder,
                    inputs=[saved_folder_path_state],
                    outputs=[save_status]
                )
        
        # Clothing Categories Reference
        with gr.Accordion("📋 Clothing Categories Reference", open=False):
            ref_text = []
            for category, items in ui.clothing_categories.items():
                ref_text.append(f"**{category}:** {', '.join(items)}")
            
            gr.Markdown("\n\n".join(ref_text))
    
    return clothing_tab

# For testing the UI independently
if __name__ == "__main__":
    with gr.Blocks(title="Clothing Analysis Test") as demo:
        clothing_tab = create_clothing_tab()
    
    demo.launch(share=False, debug=True)
