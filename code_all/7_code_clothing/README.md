# 👗 Clothing Analysis Module - RefinedLoraData

Advanced clothing description extraction for Stable Diffusion and LoRA training using state-of-the-art vision-language models.

## 🌟 Features

- **Dual Model Support**: InstructBLIP (detailed analysis) and BLIP-2 (fast processing)
- **Batch Processing**: Analyze multiple images at once
- **Single Image Analysis**: Quick analysis for individual images
- **Editable Results**: Modify and refine generated descriptions
- **SD Prompt Formatting**: Optimized output for Stable Diffusion
- **Confidence Scoring**: Quality assessment for each analysis
- **GPU Acceleration**: NVIDIA Blackwell (RTX 5000 series) optimized
- **Windows Compatible**: Designed specifically for Windows environments

## 📁 File Structure

```
7_code_clothing/
├── backend_clothing.py          # Core analysis engine
├── UI_clothing.py               # Gradio web interface
├── launch_clothing.py           # Python launcher
├── launch_clothing.bat          # Windows batch launcher
├── requirements_clothing.txt    # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Method 1: Windows Batch File (Recommended)
1. Double-click `launch_clothing.bat`
2. The script will automatically check dependencies and launch the UI
3. Open your browser to the displayed URL

### Method 2: Python Direct
1. Install dependencies: `pip install -r requirements_clothing.txt`
2. Run: `python launch_clothing.py`
3. Open your browser to http://localhost:7860

## 📋 Requirements

### System Requirements
- Windows 10/11
- Python 3.8+
- NVIDIA GPU (recommended for performance)
- 8GB+ RAM (16GB+ recommended for large models)

### Python Dependencies
All dependencies are listed in `requirements_clothing.txt`:
- torch (with CUDA support)
- transformers
- gradio
- pillow
- numpy
- accelerate

## 🎯 Usage Guide

### Single Image Analysis
1. Select your preferred model (InstructBLIP or BLIP-2)
2. Upload an image using the file uploader
3. Click "Analyze Single Image"
4. Review and edit the generated description
5. Save results if needed

### Batch Analysis
1. Select images using the batch file uploader
2. Choose your model
3. Click "Analyze Batch"
4. Review results for each image
5. Save all results to the data storage folder

### Model Selection
- **InstructBLIP**: More detailed, accurate descriptions (slower)
- **BLIP-2**: Faster processing, good general descriptions

## 📊 Output Format

### Files Generated
- `imagename_clothing.jpg`: Processed image copy
- `imagename_clothing.txt`: SD-optimized prompt
- `imagename_clothingfull.txt`: Complete analysis with categories

### SD Prompt Example
```
beautiful woman wearing elegant black evening dress, silk fabric, off-shoulder design, floor-length gown, high heels, minimal jewelry, sophisticated style, formal attire
```

### Full Analysis Example
```json
{
  "clothing_description": "...",
  "categories": {
    "Upper Body": ["evening dress"],
    "Lower Body": ["long dress"],
    "Footwear": ["high heels"],
    "Accessories": ["jewelry"]
  },
  "confidence": 0.92,
  "model_used": "instructblip"
}
```

## 🔧 Configuration

### GPU Support
The module automatically detects and uses:
- NVIDIA Blackwell architecture (RTX 5000 series)
- CUDA-compatible GPUs
- CPU fallback if no GPU available

### Memory Management
- Models are loaded on-demand to save memory
- Automatic cleanup after processing
- Configurable batch sizes for large datasets

## 🚨 Troubleshooting

### Common Issues

**"Module not found" errors:**
- Run: `pip install -r requirements_clothing.txt`
- Ensure Python is in your PATH

**GPU not detected:**
- Install CUDA toolkit
- Update NVIDIA drivers
- Check PyTorch CUDA installation

**Out of memory errors:**
- Reduce batch size
- Close other applications
- Use BLIP-2 instead of InstructBLIP

**Slow processing:**
- Ensure GPU is being used (check status display)
- Update GPU drivers
- Close unnecessary applications

## 📁 Data Storage

Results are automatically saved to:
```
../../data_storage/data_store_clothing/
```

This folder structure matches the project's organization for easy integration.

## 🔗 Integration

### With Main App
The module can be integrated into the main RefinedLoraData application by importing:
```python
from code_all.7_code_clothing.UI_clothing import create_clothing_tab
```

### Standalone Usage
The module works completely independently and can be used as a standalone clothing analysis tool.

## 🛠 Development

### Adding New Models
1. Extend the `ClothingAnalyzer` class in `backend_clothing.py`
2. Add model loading logic
3. Update the UI dropdown in `UI_clothing.py`

### Customizing Categories
Modify the `clothing_categories` dictionary in `ClothingAnalyzer` class to add new clothing types.

### Performance Optimization
- Adjust batch sizes in the backend
- Modify model precision settings
- Implement model quantization for faster inference

## 📝 License

This module is part of the RefinedLoraData project. Please refer to the main project license.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- New features include appropriate error handling
- Documentation is updated for changes
- Testing is performed on Windows with NVIDIA GPUs

## 📞 Support

For issues specific to the clothing analysis module, please check:
1. This README
2. Error messages in the console
3. GPU compatibility requirements
4. Python environment setup

---

*Part of the RefinedLoraData pipeline helper project*
