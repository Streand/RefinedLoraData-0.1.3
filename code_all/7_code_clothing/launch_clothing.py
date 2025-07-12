"""
Launch Script for Clothing Analysis Module
Simplified launcher for the RefinedLoraData clothing analysis system
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    """Launch the clothing analysis UI"""
    print("Starting RefinedLoraData - Clothing Analysis Module")
    print("=" * 60)
    
    try:
        import gradio as gr
        from UI_clothing import create_clothing_tab
        
        # Launch the interface
        print("Launching Gradio interface...")
        print("Open your browser to the URL shown below")
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Create and launch the Gradio interface
        with gr.Blocks(title="RefinedLoraData - Clothing Analysis") as demo:
            gr.Markdown("# 👗 RefinedLoraData - Clothing Analysis")
            gr.Markdown("Advanced clothing description extraction for Stable Diffusion and LoRA training")
            
            clothing_tab = create_clothing_tab()
        
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default port
            share=False,            # Don't create public link
            debug=False,            # Disable debug mode
            inbrowser=True          # Open browser automatically
        )
        
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("\nPlease install the required dependencies:")
        print("pip install -r requirements_clothing.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
