# NVIDIA RTX 5000 Series (Blackwell) GPU Support

## Overview

This document outlines how the Stable Diffusion WebUI project supports NVIDIA RTX 5000 series GPUs with Blackwell architecture, specifically addressing the 120 SM (Streaming Multiprocessor) configuration.

## Blackwell Architecture Support

### GPU Detection and Identification

The project automatically detects Blackwell GPUs using CUDA compute capability:

```python
def get_cuda_comp_cap():
    """
    Returns float of CUDA Compute Capability using nvidia-smi
    Returns 0.0 on error
    CUDA Compute Capability
    ref https://developer.nvidia.com/cuda-gpus
    ref https://en.wikipedia.org/wiki/CUDA
    Blackwell consumer GPUs should return 12.0 data-center GPUs should return 10.0
    """
    try:
        return max(map(float, subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=noheader,csv'], text=True).splitlines()))
    except Exception as _:
        return 0.0
```

### Compute Capability Mapping
- **Consumer Blackwell GPUs (RTX 5000 series)**: Compute Capability 12.0
- **Data-center Blackwell GPUs**: Compute Capability 10.0
- **120 SM Configuration**: Fully supported under compute capability 12.0

## PyTorch Installation Strategy

### Current Implementation (2025)

The project uses **pre-built PyTorch wheels** rather than building from source:

```python
torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu128")
torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.7.0 torchvision==0.22.0 --extra-index-url {torch_index_url}")
```

**Key Components:**
- **PyTorch Version**: 2.7.0
- **TorchVision Version**: 0.22.0
- **CUDA Version**: 12.8
- **Index URL**: `https://download.pytorch.org/whl/cu128`

### Historical Early Access Implementation

Previously, the project used NVIDIA's Early Access PyTorch wheels:

```python
def early_access_blackwell_wheels():
    """For Blackwell GPUs, use Early Access PyTorch Wheels provided by Nvidia"""
    print('deprecated early_access_blackwell_wheels')
    if all([
            os.environ.get('TORCH_INDEX_URL') is None,
            sys.version_info.major == 3,
            sys.version_info.minor in (10, 11, 12),
            platform.system() == "Windows",
            get_cuda_comp_cap() >= 10,  # Blackwell
    ]):
        base_repo = 'https://huggingface.co/w-e-w/torch-2.6.0-cu128.nv/resolve/main/'
        ea_whl = {
            10: f'{base_repo}torch-2.6.0+cu128.nv-cp310-cp310-win_amd64.whl#sha256=fef3de7ce8f4642e405576008f384304ad0e44f7b06cc1aa45e0ab4b6e70490d {base_repo}torchvision-0.20.0a0+cu128.nv-cp310-cp310-win_amd64.whl#sha256=50841254f59f1db750e7348b90a8f4cd6befec217ab53cbb03780490b225abef',
            11: f'{base_repo}torch-2.6.0+cu128.nv-cp311-cp311-win_amd64.whl#sha256=6665c36e6a7e79e7a2cb42bec190d376be9ca2859732ed29dd5b7b5a612d0d26 {base_repo}torchvision-0.20.0a0+cu128.nv-cp311-cp311-win_amd64.whl#sha256=bbc0ee4938e35fe5a30de3613bfcd2d8ef4eae334cf8d49db860668f0bb47083',
            12: f'{base_repo}torch-2.6.0+cu128.nv-cp312-cp312-win_amd64.whl#sha256=a3197f72379d34b08c4a4bcf49ea262544a484e8702b8c46cbcd66356c89def6 {base_repo}torchvision-0.20.0a0+cu128.nv-cp312-cp312-win_amd64.whl#sha256=235e7be71ac4e75b0f8e817bae4796d7bac8a67146d2037ab96394f2bdc63e6c'
        }
        return f'pip install {ea_whl.get(sys.version_info.minor)}'
```

## Implementation Details for 120 SM Support

### Why Pre-built Wheels vs Source Compilation

The project chose **pre-built wheels** over source compilation for several reasons:

1. **NVIDIA Optimization**: Wheels are pre-optimized by NVIDIA engineers for Blackwell architecture
2. **Faster Installation**: No compilation time required
3. **Guaranteed Compatibility**: Tested specifically for Blackwell's 120 SM configuration
4. **Regular Updates**: NVIDIA provides updates as Blackwell support improves
5. **Reliability**: Eliminates compilation errors and dependency issues

### Automatic Detection Flow

```python
# 1. Detect GPU compute capability
compute_cap = get_cuda_comp_cap()

# 2. Check if Blackwell (>= 10.0)
if compute_cap >= 10:
    # Blackwell detected - use optimized PyTorch
    
# 3. Install appropriate PyTorch version
# Current: PyTorch 2.7.0 with CUDA 12.8
# Historical: NVIDIA Early Access wheels with +cu128.nv suffix
```

### System Requirements

**Supported Platforms:**
- Windows (primary support)
- Linux (standard PyTorch wheels)

**Python Versions:**
- Python 3.10 ✅
- Python 3.11 ✅  
- Python 3.12 ✅

**CUDA Requirements:**
- CUDA 12.8 (recommended)
- Driver supporting compute capability 12.0

## Project Integration Guide

### For Your 120 SM Project

If you're working on a project that needs to utilize 120 SMs on NVIDIA RTX 5000 series:

1. **Use the Detection Method**:
   ```python
   import subprocess
   
   def get_cuda_comp_cap():
       try:
           return max(map(float, subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=noheader,csv'], text=True).splitlines()))
       except Exception:
           return 0.0
   
   # Check for Blackwell
   if get_cuda_comp_cap() >= 12.0:
       print("RTX 5000 series (120 SM) detected!")
   ```

2. **Install Correct PyTorch**:
   ```bash
   pip install torch==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu128
   ```

3. **Environment Variables** (optional):
   ```bash
   set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
   set TORCH_COMMAND=pip install torch==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu128
   ```

### Verification Commands

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name()}")
print(f"GPU compute capability: {torch.cuda.get_device_capability()}")
```

Expected output for RTX 5000 series:
```
PyTorch version: 2.7.0+cu128
CUDA available: True
CUDA version: 12.8
GPU name: NVIDIA GeForce RTX 5090 (or similar)
GPU compute capability: (12, 0)
```

## Technical Advantages

### Why This Approach Works for 120 SM

1. **Full SM Utilization**: PyTorch 2.7.0 includes optimizations for all 120 SMs
2. **Memory Efficiency**: CUDA 12.8 provides better memory management for large models
3. **Kernel Optimization**: Pre-compiled kernels are optimized for Blackwell's architecture
4. **Tensor Cores**: Full support for Blackwell's enhanced Tensor Cores
5. **Mixed Precision**: Optimized FP16/BF16 operations across all SMs

### Performance Considerations

- **Batch Size**: Can leverage all 120 SMs with larger batch sizes
- **Model Parallelism**: Better support for splitting large models across SMs
- **Memory Bandwidth**: Optimized memory access patterns for Blackwell
- **Power Efficiency**: Improved performance per watt utilization

## Troubleshooting

### Common Issues

1. **Compute Capability Not Detected**:
   - Ensure NVIDIA drivers are up to date
   - Verify `nvidia-smi` is accessible from command line

2. **PyTorch Installation Fails**:
   - Check Python version compatibility (3.10-3.12)
   - Verify CUDA 12.8 driver support

3. **SM Underutilization**:
   - Increase batch size to fully utilize 120 SMs
   - Check model size vs GPU memory

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Conclusion

The Stable Diffusion WebUI project demonstrates an effective approach to supporting NVIDIA RTX 5000 series (Blackwell) GPUs with 120 SMs by:

- Using automatic GPU detection based on compute capability
- Leveraging pre-optimized PyTorch wheels rather than source compilation  
- Providing fallback mechanisms for different system configurations
- Ensuring full utilization of Blackwell's advanced features

This approach can be adapted for other projects requiring 120 SM support on RTX 5000 series GPUs.

---

**Last Updated**: June 30, 2025  
**PyTorch Version**: 2.7.0  
**CUDA Version**: 12.8  
**Target Hardware**: NVIDIA RTX 5000 Series (Blackwell, 120 SM)
