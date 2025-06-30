import gradio as gr
from gradio import themes
import sys
import os

# Camera functionality is now embedded directly in this file to avoid import issues
CAMERA_UI_AVAILABLE = True

def create_main_ui():
    """Create the main Gradio interface with all tabs"""
    with gr.Blocks(title="RefinedLoraData-0.1.0", theme=themes.Soft()) as interface:

        # Main header
        gr.Markdown("# RefinedLoraData")
        
        # Create the horizontal tabs
        with gr.Tabs():
            # Main Tab
            with gr.TabItem("Main"):
                gr.Markdown("## Welcome to RefinedLoraData-0.1.0")
                gr.Markdown("This is the main landing page.")
            
            # Video Tab
            with gr.TabItem("Video"):
                gr.Markdown("## Video Processing")
                gr.Markdown("Video processing functionality will be added here.")
            
            # Face Tab
            with gr.TabItem("Face"):
                gr.Markdown("## Face Analysis")
                gr.Markdown("Face analysis functionality will be added here.")
            
            # Body Tab
            with gr.TabItem("Body"):
                gr.Markdown("## Body Analysis")
                gr.Markdown("Body analysis functionality will be added here.")
            
            # Pose Tab
            with gr.TabItem("Pose"):
                gr.Markdown("## Pose Analysis")
                gr.Markdown("Pose analysis functionality will be added here.")
            
            # Camera Tab - Use embedded implementation with both backends
            with gr.TabItem("Camera"):
                gr.Markdown("# 📸 Camera Angle & Framing Analysis")
                gr.Markdown("Analyze images to determine camera framing and angles for Stable Diffusion prompts")
                
                try:
                    # Create embedded camera UI
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
                                    backend_status = gr.Markdown(
                                        value="Backend will be checked when analysis starts",
                                        label="Backend Status"
                                    )
                                    
                                    analysis_output = gr.Markdown(
                                        label="Analysis Results",
                                        value="Upload an image and click analyze to see results"
                                    )
                                    
                                    sd_prompt_output = gr.Textbox(
                                        label="Stable Diffusion Prompt Components",
                                        placeholder="Generated prompt components will appear here",
                                        lines=2
                                    )
                            
                            # YOLO-only analysis function
                            def analyze_camera_angle(image):
                                if image is None:
                                    return "🔍 YOLO Backend: Ready", "No image uploaded", ""
                                
                                try:
                                    # Try to import and use the camera backend
                                    import sys
                                    import os
                                    backend_path = os.path.join(os.path.dirname(__file__), '..', '6_code_camera')
                                    sys.path.append(backend_path)
                                    
                                    # Use YOLO backend only
                                    try:
                                        from backend_camera_yolo import YOLOCameraAnalyzer
                                        analyzer = YOLOCameraAnalyzer()
                                        backend_info = f"🚀 **YOLO Backend Active**\n- Device: {analyzer.device}\n- Model: {analyzer.model_size}"
                                        if analyzer.device == 'cpu':
                                            backend_info += "\n- ⚠️ Using CPU (GPU not yet supported for Blackwell)"
                                    except ImportError as e:
                                        return f"❌ **YOLO Backend Error**: {str(e)}", "YOLO backend not available. Please install: pip install torch torchvision ultralytics", ""
                                    except Exception as e:
                                        return f"❌ **YOLO Backend Error**: {str(e)}", f"Error initializing YOLO backend: {str(e)}", ""
                                    
                                    # Save temporary image for analysis
                                    temp_path = "temp_analysis_image.jpg"
                                    image.save(temp_path)
                                    
                                    # Analyze the image
                                    result = analyzer.analyze_image(temp_path)
                                    
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                    
                                    if "error" in result:
                                        return backend_info, f"Error: {result['error']}", ""
                                    
                                    # Format YOLO analysis results
                                    analysis_text = f"""📊 **YOLO Image Analysis Results**

🎬 **Framing**: {result.get('framing', 'unknown')}
📐 **Camera Angle**: {result.get('camera_angle', 'unknown')}
� **People Detected**: {result.get('people_detected', 0)}
� **Confidence**: {result.get('confidence', 0.0):.2f}
⏱️ **Analysis Time**: {result.get('inference_time', 0.0):.2f}s
🔧 **Device**: {result.get('device', 'unknown')}"""

                                    # Add pose analysis if available
                                    if 'pose_analysis' in result:
                                        pose = result['pose_analysis']
                                        analysis_text += f"""

🦴 **Pose Analysis**:
  - Visible Keypoints: {pose.get('visible_keypoints', 0)}/17
  - Symmetry Score: {pose.get('symmetry_score', 0.0):.2f}
  - Face Visibility: {pose.get('face_visibility', 'unknown')}
  - Body Orientation: {pose.get('body_orientation', 'unknown')}"""
                                    
                                    sd_prompt = analyzer.get_stable_diffusion_prompt(result)
                                    
                                    return backend_info, analysis_text, sd_prompt
                                    
                                except ImportError as e:
                                    error_msg = "📊 YOLO analysis module not available."
                                    error_msg += "\nPlease install: pip install torch torchvision ultralytics"
                                    return f"❌ **Backend Error**: {str(e)}", error_msg, ""
                                except Exception as e:
                                    return f"❌ **Backend Error**: {str(e)}", f"📊 Error analyzing image: {str(e)}", ""
                            
                            analyze_btn.click(
                                fn=analyze_camera_angle,
                                inputs=[image_input],
                                outputs=[backend_status, analysis_output, sd_prompt_output]
                            )

                        # Batch Upload & Save Tab
                        with gr.TabItem("Batch Upload & Save"):
                            gr.Markdown("### 📤 Upload Multiple Images for Batch Analysis")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Upload Images")
                                    
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
                                        value="📁 Upload images above and click analyze"
                                    )
                                
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Save Results")
                                    
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
                                        value="🔄 Process images first, then save results"
                                    )
                            
                            # Hidden state to store processed results and folder path
                            processed_data = gr.State([])
                            saved_folder_path = gr.State("")
                            
                            # Batch processing functions (embedded)
                            def process_batch_upload(files):
                                if not files or len(files) == 0:
                                    return "📁 No images uploaded", [], gr.update(interactive=False)
                                
                                try:
                                    # Import camera backend
                                    import sys
                                    import os
                                    backend_path = os.path.join(os.path.dirname(__file__), '..', '6_code_camera')
                                    sys.path.append(backend_path)
                                    
                                    from backend_camera_yolo import YOLOCameraAnalyzer
                                    analyzer = YOLOCameraAnalyzer()
                                    
                                    processed_results = []
                                    total_files = len(files)
                                    successful_count = 0
                                    failed_files = []
                                    
                                    for i, file in enumerate(files):
                                        try:
                                            result = analyzer.analyze_image(file.name)
                                            
                                            if "error" in result and result.get("success", False) is False:
                                                failed_files.append(f"{os.path.basename(file.name)} ({result['error']})")
                                                continue
                                            
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
                                        for failed in failed_files[:3]:
                                            status_msg += f"   • {failed}\n"
                                        if len(failed_files) > 3:
                                            status_msg += f"   • ... and {len(failed_files) - 3} more\n"
                                    
                                    status_msg += f"\n💾 Ready to save {successful_count} analyzed images"
                                    
                                    save_enabled = len(processed_results) > 0
                                    return status_msg, processed_results, gr.update(interactive=save_enabled)
                                    
                                except Exception as e:
                                    return f"❌ Backend error: {str(e)}", [], gr.update(interactive=False)
                            
                            def save_batch_results(processed_results, folder_name):
                                if not processed_results:
                                    return "❌ No processed results to save", gr.update(interactive=False)
                                    
                                if not folder_name or folder_name.strip() == "":
                                    return "❌ Please enter a folder name", gr.update(interactive=True)
                                
                                try:
                                    import shutil
                                    from pathlib import Path
                                    import json
                                    import time
                                    
                                    # Import analyzer for SD prompt generation
                                    import sys
                                    import os
                                    backend_path = os.path.join(os.path.dirname(__file__), '..', '6_code_camera')
                                    sys.path.append(backend_path)
                                    from backend_camera_yolo import YOLOCameraAnalyzer
                                    analyzer = YOLOCameraAnalyzer()
                                    
                                    # Clean folder name
                                    clean_folder_name = "".join(c for c in folder_name.strip() if c.isalnum() or c in (' ', '-', '_')).strip()
                                    if not clean_folder_name:
                                        return "❌ Invalid folder name. Use only letters, numbers, spaces, hyphens, and underscores.", gr.update(interactive=True)
                                    
                                    # Create output directory - use absolute path to project root
                                    current_file_dir = Path(__file__).parent  # 1_code_main_app directory
                                    project_root = current_file_dir.parent.parent  # Go up two levels to project root
                                    base_dir = project_root / "data_storage" / "data_store_camera"
                                    output_dir = base_dir / clean_folder_name
                                    
                                    # Check if directory already exists
                                    if output_dir.exists():
                                        timestamp = int(time.time())
                                        output_dir = base_dir / f"{clean_folder_name}_{timestamp}"
                                    
                                    output_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    saved_count = 0
                                    
                                    for i, result_data in enumerate(processed_results, 1):
                                        file_path = result_data['file_path']
                                        analysis = result_data['analysis']
                                        
                                        # Get file extension
                                        file_ext = Path(result_data['file_name']).suffix.lower()
                                        
                                        # Generate new filenames
                                        base_name = f"{clean_folder_name}-camera-{i}"
                                        image_name = f"{base_name}{file_ext}"
                                        txt_name = f"{base_name}.txt"
                                        full_txt_name = f"{base_name}full.txt"
                                        
                                        # Copy image file
                                        shutil.copy2(file_path, output_dir / image_name)
                                        
                                        # Generate SD prompt
                                        sd_prompt = analyzer.get_stable_diffusion_prompt(analysis)
                                        
                                        # Write simple txt file (SD prompt)
                                        with open(output_dir / txt_name, 'w', encoding='utf-8') as f:
                                            f.write(sd_prompt)
                                        
                                        # Write full analysis txt file
                                        with open(output_dir / full_txt_name, 'w', encoding='utf-8') as f:
                                            f.write("=== YOLO Camera Analysis Results ===\n\n")
                                            f.write(f"Framing: {analysis.get('framing', 'unknown')}\n")
                                            f.write(f"Camera Angle: {analysis.get('camera_angle', 'unknown')}\n")
                                            f.write(f"People Detected: {analysis.get('people_detected', 0)}\n")
                                            f.write(f"Confidence: {analysis.get('confidence', 0.0):.3f}\n")
                                            f.write(f"Analysis Time: {analysis.get('inference_time', 0.0):.3f}s\n")
                                            f.write(f"Device: {analysis.get('device', 'unknown')}\n")
                                            
                                            if 'pose_analysis' in analysis and analysis['pose_analysis']:
                                                pose = analysis['pose_analysis']
                                                f.write(f"\n=== Pose Analysis ===\n")
                                                f.write(f"Visible Keypoints: {pose.get('visible_keypoints', 0)}/17\n")
                                                f.write(f"Symmetry Score: {pose.get('symmetry_score', 0.0):.3f}\n")
                                                f.write(f"Face Visibility: {pose.get('face_visibility', 'unknown')}\n")
                                                f.write(f"Body Orientation: {pose.get('body_orientation', 'unknown')}\n")
                                            
                                            f.write(f"\n=== Stable Diffusion Prompt ===\n")
                                            f.write(sd_prompt)
                                            
                                            f.write(f"\n\n=== Raw Analysis Data (JSON) ===\n")
                                            f.write(json.dumps(analysis, indent=2))
                                        
                                        saved_count += 1
                                    
                                    success_msg = f"✅ Successfully saved {saved_count} images and analysis files!\n\n"
                                    success_msg += f"📁 Saved to: {output_dir.absolute()}\n\n"
                                    success_msg += f"📋 Files created per image:\n"
                                    success_msg += f"   • [name]-camera-[num].jpg/png - Original image\n"
                                    success_msg += f"   • [name]-camera-[num].txt - SD prompt\n"
                                    success_msg += f"   • [name]-camera-[num]full.txt - Complete analysis"
                                    
                                    # Return folder path and update button states
                                    folder_path = str(output_dir.absolute())
                                    open_btn_visible = True
                                    open_btn_interactive = True
                                    
                                    return (success_msg, 
                                            gr.update(interactive=False), 
                                            folder_path,
                                            gr.update(visible=open_btn_visible, interactive=open_btn_interactive))
                                    
                                except Exception as e:
                                    return (f"❌ Save error: {str(e)}", 
                                            gr.update(interactive=True),
                                            "",
                                            gr.update(visible=False, interactive=False))
                            
                            # Wire up batch upload processing
                            upload_btn.click(
                                fn=process_batch_upload,
                                inputs=[batch_upload],
                                outputs=[processing_status, processed_data, save_btn]
                            )
                            
                            # Wire up save functionality
                            save_btn.click(
                                fn=save_batch_results,
                                inputs=[processed_data, folder_name_input],
                                outputs=[save_status, save_btn, saved_folder_path, open_folder_btn]
                            )
                            
                            # Wire up open folder functionality
                            def open_saved_folder(folder_path):
                                """Open the saved folder in system file explorer"""
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
                            
                            open_folder_btn.click(
                                fn=open_saved_folder,
                                inputs=[saved_folder_path],
                                outputs=[save_status]
                            )
                        
                        # Reference Guide Tab
                        with gr.TabItem("Reference Guide"):
                            gr.Markdown("""
                            # 📚 Camera Analysis Reference Guide

                            ## 🔧 Analysis Backends

                            ### OpenCV Backend (Fast)
                            - **Speed**: Very fast analysis (~0.1-0.5 seconds)
                            - **Accuracy**: Good for basic pose detection
                            - **Requirements**: opencv-python, numpy
                            - **Best for**: Batch processing, quick analysis
                            - **GPU Support**: CPU only

                            ### YOLO Backend (Accurate)
                            - **Speed**: Slower but more thorough (~1-3 seconds)
                            - **Accuracy**: Advanced pose detection with confidence scores
                            - **Requirements**: torch, torchvision, ultralytics
                            - **Best for**: Detailed analysis, complex poses
                            - **GPU Support**: CUDA (Blackwell support pending in PyTorch)
                            - **Models**: nano (fastest) to extra_large (most accurate)

                            ## 🎬 Camera Framing Types
                            - **extreme close-up**: Very tight shot focusing on eyes/face details
                            - **close-up**: Head and shoulders visible, intimate view
                            - **medium shot**: Waist up, good for dialogue and expressions
                            - **full body shot**: Complete figure from head to toe
                            - **establishing shot**: Wide view showing environment and context

                            ## 📐 Camera Angle Types
                            - **straight on**: Camera at eye level, direct view
                            - **bilaterally symmetrical**: Centered, balanced composition
                            - **side view**: Profile view of the subject
                            - **back view**: Camera positioned behind the subject
                            - **from above**: High angle, camera above subject
                            - **from below**: Low angle, camera below subject (hero shot)
                            - **wide angle view**: Distorted perspective, wider field of view
                            - **fisheye view**: Extreme wide angle with barrel distortion
                            - **overhead shot**: Directly above the subject
                            - **top down shot**: Bird's eye view perspective
                            - **hero view**: Low angle making subject appear powerful
                            - **selfie**: Close, personal angle typical of self-portraits

                            ## 🎯 Usage for Stable Diffusion
                            These terms can be used in your SD prompts to control composition:
                            - Use framing terms to control how much of the subject is visible
                            - Use angle terms to control the camera's position relative to the subject
                            - Combine with weights like `(medium shot:1.2)` for stronger effect
                            - Works best with portrait aspect ratios for character shots
                            
                            ## 💡 Tips for Best Results
                            - Upload clear images with visible subjects
                            - Higher resolution images provide more accurate analysis
                            - For YOLO: Use GPU when available for faster processing
                            - For OpenCV: Best for simple poses and quick analysis
                            - Generated prompt components can be directly used in SD
                            
                            ## 🚀 Performance Notes
                            - **Blackwell GPU**: Currently using CPU fallback, GPU support coming in future PyTorch releases
                            - **YOLO Models**: nano (fastest) → small → medium → large → extra_large (most accurate)
                            - **Batch Processing**: Use OpenCV backend for analyzing many images quickly
                            """)
                        
                        # Performance Comparison Tab
                        with gr.TabItem("Performance Comparison"):
                            gr.Markdown("""
                            # ⚡ Backend Performance Comparison

                            ## 🔧 OpenCV Backend
                            | Feature | Performance |
                            |---------|------------|
                            | **Speed** | ⭐⭐⭐⭐⭐ Very Fast (0.1-0.5s) |
                            | **Accuracy** | ⭐⭐⭐ Good for basic detection |
                            | **Face Detection** | ⭐⭐⭐⭐ Reliable Haar cascades |
                            | **Pose Analysis** | ⭐⭐ Basic orientation detection |
                            | **Memory Usage** | ⭐⭐⭐⭐⭐ Very Low |
                            | **Dependencies** | ⭐⭐⭐⭐⭐ Minimal (opencv + numpy) |

                            ## 🚀 YOLO Backend
                            | Feature | Performance |
                            |---------|------------|
                            | **Speed** | ⭐⭐⭐ Moderate (1-3s CPU, 0.1-0.5s GPU) |
                            | **Accuracy** | ⭐⭐⭐⭐⭐ Excellent pose detection |
                            | **Face Detection** | ⭐⭐⭐⭐⭐ Advanced keypoint detection |
                            | **Pose Analysis** | ⭐⭐⭐⭐⭐17+ keypoints with confidence |
                            | **Memory Usage** | ⭐⭐⭐ Higher (model loading) |
                            | **Dependencies** | ⭐⭐ Requires PyTorch + Ultralytics |

                            ## 🎯 When to Use Each Backend

                            ### Choose OpenCV When:
                            - Processing many images quickly
                            - Limited system resources
                            - Simple pose requirements
                            - Fast prototyping

                            ### Choose YOLO When:
                            - Need detailed pose analysis
                            - Working with complex poses
                            - Accuracy is more important than speed
                            - Have sufficient system resources

                            ## 🔮 Future GPU Support
                            - **Current**: Both backends use CPU
                            - **OpenCV**: Will remain CPU-based
                            - **YOLO**: Will auto-upgrade to GPU when PyTorch adds Blackwell support
                            - **Expected**: Significant speedup (5-10x) when GPU support is available
                            """)
                except Exception as e:
                    gr.Markdown(f"## Camera Analysis")
                    gr.Markdown(f"Error loading camera analysis: {e}")
                    gr.Markdown("Install dependencies: pip install opencv-python numpy")
                    gr.Markdown("For YOLO backend: pip install torch torchvision ultralytics")
            
            # Clothing Tab
            with gr.TabItem("Clothing"):
                gr.Markdown("## Clothing Analysis")
                gr.Markdown("Clothing analysis functionality will be added here.")
            
            # NSFW Tab
            with gr.TabItem("NSFW"):
                gr.Markdown("## NSFW Analysis")
                gr.Markdown("NSFW analysis functionality will be added here.")
            
            # Finalize Tab
            with gr.TabItem("Finalize"):
                gr.Markdown("## Finalize")
                gr.Markdown("Finalization functionality will be added here.")
    
    return interface

def launch_ui():
    """Launch the Gradio interface"""
    interface = create_main_ui()
    return interface