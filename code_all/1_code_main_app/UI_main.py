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
                                    # Backend selection
                                    backend_choice = gr.Radio(
                                        choices=["OpenCV (Fast)", "YOLO (Accurate)"],
                                        value="OpenCV (Fast)",
                                        label="🔧 Analysis Backend",
                                        info="Choose between speed (OpenCV) or accuracy (YOLO)"
                                    )
                                    
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
                            
                            # Enhanced analysis function with both backends
                            def analyze_camera_angle(image, backend_choice):
                                if image is None:
                                    return "🔍 Backend Check: Ready", "No image uploaded", ""
                                
                                # Determine which backend to use
                                use_yolo = "YOLO" in backend_choice
                                
                                try:
                                    # Try to import and use the camera backend
                                    import sys
                                    import os
                                    backend_path = os.path.join(os.path.dirname(__file__), '..', '6_code_camera')
                                    sys.path.append(backend_path)
                                    
                                    if use_yolo:
                                        # Try YOLO backend first
                                        try:
                                            from backend_camera_yolo import YOLOCameraAnalyzer
                                            analyzer = YOLOCameraAnalyzer()
                                            backend_info = f"🚀 **YOLO Backend Active**\n- Device: {analyzer.device}\n- Model: {analyzer.model_size}"
                                            if analyzer.device == 'cpu':
                                                backend_info += "\n- ⚠️ Using CPU (GPU not yet supported for Blackwell)"
                                        except ImportError as e:
                                            # Fall back to OpenCV if YOLO not available
                                            from backend_camera import CameraAnalyzer
                                            analyzer = CameraAnalyzer()
                                            backend_info = f"⚠️ **Fallback to OpenCV**: YOLO not available ({str(e)})"
                                        except Exception as e:
                                            from backend_camera import CameraAnalyzer
                                            analyzer = CameraAnalyzer()
                                            backend_info = f"⚠️ **Fallback to OpenCV**: YOLO error ({str(e)})"
                                    else:
                                        # Use OpenCV backend
                                        from backend_camera import CameraAnalyzer
                                        analyzer = CameraAnalyzer()
                                        backend_info = "🔧 **OpenCV Backend Active**\n- Fast face detection\n- Basic pose analysis"
                                    
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
                                    
                                    # Format results with backend-specific enhancements
                                    if hasattr(analyzer, 'device'):
                                        # YOLO backend results
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
                                    else:
                                        # OpenCV backend results  
                                        analysis_text = f"""📊 **OpenCV Image Analysis Results**

🎬 **Framing**: {result.get('framing', 'unknown')}
📐 **Camera Angle**: {result.get('angle', 'unknown')}
📏 **Aspect Ratio**: {result.get('aspect_ratio', 'unknown')}
🎨 **Composition**: {result.get('composition_notes', 'unknown')}
📐 **Dimensions**: {result.get('image_dimensions', 'unknown')}"""
                                    
                                    sd_prompt = analyzer.get_stable_diffusion_prompt(result)
                                    
                                    return backend_info, analysis_text, sd_prompt
                                    
                                except ImportError as e:
                                    error_msg = "📊 Camera analysis module not available."
                                    if use_yolo:
                                        error_msg += "\nFor YOLO: pip install torch torchvision ultralytics"
                                    else:
                                        error_msg += "\nFor OpenCV: pip install opencv-python numpy"
                                    return f"❌ **Backend Error**: {str(e)}", error_msg, ""
                                except Exception as e:
                                    return f"❌ **Backend Error**: {str(e)}", f"📊 Error analyzing image: {str(e)}", ""
                            
                            analyze_btn.click(
                                fn=analyze_camera_angle,
                                inputs=[image_input, backend_choice],
                                outputs=[backend_status, analysis_output, sd_prompt_output]
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