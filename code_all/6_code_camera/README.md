# 📸 Camera Angle & Framing Analysis

This module analyzes images to determine camera framing and angles for use in Stable Diffusion prompts and LoRA training data curation.

## 🎯 Purpose

The Camera Analysis module helps you:
- **Detect camera framing** (extreme close-up, close-up, medium shot, full body, establishing shot)
- **Identify camera angles** (straight on, side view, from above, from below, hero view, etc.)
- **Generate Stable Diffusion prompt components** for consistent image generation
- **Analyze composition elements** (symmetry, rule of thirds, aspect ratios)

## 📋 Features

### Single Image Analysis
- Upload an image and get instant camera analysis
- Detailed breakdown of framing and angle
- Stable Diffusion prompt components
- Composition notes

### Batch Processing
- Analyze entire folders of images
- CSV export for data organization
- Perfect for LoRA training dataset preparation

### Reference Guide
- Complete list of supported camera terms
- Detailed descriptions for each framing type
- Usage tips for Stable Diffusion prompts

## 🛠️ Technical Details

### Supported Framing Types
- **extreme close-up**: Very tight shot focusing on eyes/face details
- **close-up**: Head and shoulders visible, intimate view
- **medium shot**: Waist up, good for dialogue and expressions
- **full body shot**: Complete figure from head to toe
- **establishing shot**: Wide view showing environment and context

### Supported Camera Angles
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

## 🔧 Dependencies

The module requires:
- `opencv-python` - Computer vision processing
- `numpy` - Numerical operations
- `PIL/Pillow` - Image handling
- `gradio` - User interface

All dependencies are included in the main `requirements.txt`.

## 🚀 Usage

### In Main Application
The Camera tab is automatically integrated into the main RefinedLoraData interface.

### Standalone Testing
```bash
python test_camera.py
```

### Programmatic Usage
```python
from backend_camera import CameraAnalyzer

# Create analyzer
analyzer = CameraAnalyzer()

# Analyze single image
result = analyzer.analyze_image("path/to/image.jpg")
print(result)

# Generate SD prompt components
prompt = analyzer.get_stable_diffusion_prompt(result)
print(f"Prompt components: {prompt}")

# Batch analysis
results = analyzer.analyze_batch("path/to/image/folder")
```

## 📊 Analysis Algorithm

The analysis uses computer vision techniques:

1. **Face Detection**: Uses OpenCV Haar cascades to detect faces and profiles
2. **Framing Analysis**: Calculates face/subject size relative to image dimensions
3. **Angle Detection**: Analyzes face position and orientation
4. **Perspective Analysis**: Uses line detection to identify vanishing points
5. **Composition Analysis**: Checks for symmetry, rule of thirds, and balance

## 🎨 Stable Diffusion Integration

### Using Analysis Results
The generated prompt components can be directly used in SD prompts:

```
a beautiful woman, (medium shot:1.2), (bilaterally symmetrical:1.1), portrait
```

### Best Practices
- Use framing terms to control how much of the subject is visible
- Use angle terms to control camera position relative to subject
- Combine with weights like `(term:1.2)` for stronger effect
- Works best with portrait aspect ratios for character shots

## 🔍 Accuracy Notes

- **Best Results**: Clear images with visible human subjects
- **Face Detection**: Works well with frontal and profile views
- **Lighting**: Better results with good contrast and lighting
- **Resolution**: Higher resolution images provide more accurate analysis

## 🛠️ Troubleshooting

### Common Issues

**"Backend not available"**
- Install OpenCV: `pip install opencv-python`
- Check Python environment

**Inaccurate face detection**
- Try different lighting or angles
- Ensure faces are clearly visible
- Check image resolution

**No analysis results**
- Verify image format (JPG, PNG supported)
- Check file permissions
- Ensure image is not corrupted

## 🔮 Future Enhancements

Potential improvements:
- Deep learning-based pose estimation
- More granular angle detection
- Support for multiple subjects
- Integration with other analysis modules
- Custom model training for specific use cases

## 📚 References

Based on camera terminology from:
- [StudioBinder Camera Shot Guide](https://www.studiobinder.com/blog/ultimate-guide-to-camera-shots/)
- [SDXL Camera Framing Guide](https://weirdwonderfulai.art/resources/sdxl-guide-to-camera-framing-and-angle/)
- Professional photography and cinematography standards
