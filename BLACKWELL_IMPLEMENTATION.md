# NVIDIA RTX 5000 Series (Blackwell) Support Implementation

This project now includes comprehensive support for NVIDIA RTX 5000 series GPUs with Blackwell architecture, following the A1111 (AUTOMATIC1111) approach.

## ✅ Implementation Status

### Core Features Implemented
- ✅ **Automatic GPU Detection**: Detects Blackwell GPUs using compute capability 12.0
- ✅ **PyTorch Optimization**: Recommends PyTorch 2.7.0+cu128 for 120 SM support
- ✅ **Smart Device Selection**: Automatically falls back to CPU if GPU not supported
- ✅ **UI Integration**: GPU status display in camera analysis interface
- ✅ **Installation Tools**: Automated PyTorch installer/upgrader script

### Files Added/Modified
- `blackwell_support.py` - Core Blackwell detection and PyTorch management
- `install_pytorch_blackwell.py` - Automated installer script
- `install_pytorch_blackwell.bat` - Windows batch installer
- `requirements.txt` - Updated with modern PyTorch 2.7.0+cu128
- `code_all/6_code_camera/backend_camera_yolo.py` - Integrated Blackwell support
- `code_all/6_code_camera/UI_camera.py` - Added GPU status display

## 🚀 Quick Start

### Check Your System
```bash
# Check GPU support status
python blackwell_support.py
```

### Install/Upgrade PyTorch for Blackwell
```bash
# Windows (recommended)
install_pytorch_blackwell.bat

# Or manually
python install_pytorch_blackwell.py

# Dry run (see what would be installed)
python install_pytorch_blackwell.py --dry-run
```

### Manual Installation
```bash
pip install torch==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

## 🐍 Virtual Environment Support

### Automatic Detection
The installer automatically detects if you're running in a virtual environment:

- ✅ **Virtual Environment Detected**: Installs to the active venv
- ⚠️ **Global Python Detected**: Warns and offers to activate project venv
- 🔍 **Project Venv Found**: Suggests activation if not currently active

### Recommended Setup
```bash
# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate

# Install PyTorch for Blackwell
install_pytorch_blackwell.bat  # or .py
```

### Environment Detection
The installer checks for:
- Active virtual environments (venv, conda, etc.)
- Project-specific venv directories (`venv/`, `.venv/`, `env/`)
- Python executable paths and environment isolation

## 🎮 GPU Support Status

### RTX 5000 Series (Blackwell)
- **RTX 5090**: ✅ Fully supported with 120 SMs
- **RTX 5080**: ✅ Fully supported with 120 SMs  
- **RTX 5070**: ✅ Supported (SM count varies)

### Requirements
- **Compute Capability**: 12.0 (consumer Blackwell)
- **PyTorch Version**: 2.7.0 or later
- **CUDA Version**: 12.8 or compatible
- **Python**: 3.10, 3.11, or 3.12
- **Platform**: Windows (primary), Linux (standard PyTorch)

## 📊 Features

### Automatic Detection
The system automatically detects:
- GPU compute capability using `nvidia-smi`
- Blackwell architecture (compute capability ≥ 12.0)
- RTX 5000 series identification
- Streaming Multiprocessor (SM) count estimation
- PyTorch compatibility status

### Smart Fallbacks
- Uses GPU acceleration when fully supported
- Falls back to CPU if GPU support is incomplete
- Provides clear status messages and upgrade recommendations
- Environment variable support for custom configurations

### UI Integration
- Real-time GPU status in camera analysis interface
- Blackwell detection and support status
- Performance optimization information
- PyTorch upgrade recommendations

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Custom PyTorch index URL
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

# Custom PyTorch install command
set TORCH_COMMAND=pip install torch==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

### Verification Commands
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
```

Expected output for RTX 5000 series:
```
PyTorch: 2.7.0+cu128
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 5080
Compute Capability: (12, 0)
```

## 📋 Troubleshooting

### Common Issues

1. **"sm_120 is not compatible" Warning**
   - Current PyTorch doesn't support Blackwell architecture
   - Solution: Run `install_pytorch_blackwell.bat` to upgrade

2. **GPU Detected but Using CPU**
   - PyTorch version too old for Blackwell support
   - Solution: Upgrade to PyTorch 2.7.0+cu128

3. **Installation Fails**
   - Check Python version (must be 3.10-3.12)
   - Verify NVIDIA drivers are up to date
   - Ensure internet connectivity for downloading wheels

### Debug Commands
```bash
# Check GPU status
nvidia-smi

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Test PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Full system check
python blackwell_support.py
```

## 📈 Performance Benefits

### 120 SM Utilization
- **Full SM Access**: PyTorch 2.7.0 utilizes all 120 streaming multiprocessors
- **Memory Efficiency**: CUDA 12.8 provides optimized memory management
- **Kernel Optimization**: Pre-compiled kernels optimized for Blackwell
- **Tensor Cores**: Enhanced Tensor Core support for mixed precision

### Optimization Tips
- Use larger batch sizes to fully utilize 120 SMs
- Enable mixed precision (FP16/BF16) for better performance
- Consider model parallelism for very large models
- Monitor GPU memory usage to avoid bottlenecks

## 🔄 Migration from Early Access Wheels

This implementation replaces the older NVIDIA Early Access wheels with official PyTorch 2.7.0+cu128:

### Old Approach (Deprecated)
```bash
# Early Access wheels (no longer recommended)
https://huggingface.co/w-e-w/torch-2.6.0-cu128.nv/...
```

### New Approach (Current)
```bash
# Official PyTorch wheels with Blackwell support
pip install torch==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

## 📚 Technical Details

### Architecture Support
- **Consumer Blackwell**: Compute capability 12.0 (RTX 5000 series)
- **Data-center Blackwell**: Compute capability 10.0 (H100, H200, etc.)
- **Backward Compatibility**: All previous generations still supported

### Integration Points
- **YOLO Backend**: Device selection with Blackwell awareness
- **UI Components**: Status display and user feedback
- **Installation**: Automated detection and upgrade tools
- **Fallback Logic**: Graceful degradation to CPU when needed

---

**Last Updated**: June 30, 2025  
**PyTorch Version**: 2.7.0+cu128  
**Target Hardware**: NVIDIA RTX 5000 Series (Blackwell, 120 SM)  
**Implementation**: Based on A1111 approach
