#!/usr/bin/env python3
"""
Main application entry point for RefinedLoraData-0.1.0
This file starts the main UI application.
"""

from UI_main import launch_ui

def main():
    """Main application entry point"""
    print("Starting RefinedLoraData-0.1.0...")
    
    # Launch the Gradio UI
    ui = launch_ui()
    ui.launch(share=False, debug=True, inbrowser=True)

if __name__ == "__main__":
    main()