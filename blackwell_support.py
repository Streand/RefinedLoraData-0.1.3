"""
NVIDIA RTX 5000 Series (Blackwell) GPU Support Module

Based on A1111's implementation for handling NVIDIA RTX 5000 series GPUs with 120 SM configuration.
Provides automatic GPU detection, PyTorch installation, and optimization for Blackwell architecture.
"""

import subprocess
import sys
import platform
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_cuda_compute_capability() -> float:
    """
    Returns float of CUDA Compute Capability using nvidia-smi
    Returns 0.0 on error
    
    CUDA Compute Capability Reference:
    - Consumer Blackwell GPUs (RTX 5000 series): 12.0
    - Data-center Blackwell GPUs: 10.0
    - Ada Lovelace (RTX 4000 series): 8.9
    - Ampere (RTX 3000 series): 8.6
    
    Returns:
        float: Compute capability (e.g., 12.0 for RTX 5000 series)
    """
    try:
        output = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=compute_cap', 
            '--format=noheader,csv'
        ], text=True)
        
        capabilities = [float(line.strip()) for line in output.splitlines() if line.strip()]
        return max(capabilities) if capabilities else 0.0
        
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.debug(f"Could not get CUDA compute capability: {e}")
        return 0.0

def get_gpu_info() -> dict:
    """
    Get comprehensive GPU information for Blackwell detection
    
    Returns:
        dict: GPU information including name, compute capability, memory, etc.
    """
    info = {
        'compute_capability': 0.0,
        'name': 'Unknown',
        'memory_gb': 0.0,
        'is_blackwell': False,
        'is_rtx_5000_series': False,
        'sm_count': 0
    }
    
    try:
        # Get compute capability
        info['compute_capability'] = get_cuda_compute_capability()
        
        # Get GPU name and memory
        gpu_query = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=name,memory.total',
            '--format=csv,noheader,nounits'
        ], text=True)
        
        for line in gpu_query.splitlines():
            if line.strip():
                parts = line.strip().split(', ')
                if len(parts) >= 2:
                    info['name'] = parts[0]
                    info['memory_gb'] = float(parts[1]) / 1024  # Convert MB to GB
                break
        
        # Detect Blackwell architecture
        info['is_blackwell'] = info['compute_capability'] >= 10.0
        info['is_rtx_5000_series'] = (
            info['compute_capability'] >= 12.0 and 
            ('RTX 50' in info['name'] or 'RTX 5' in info['name'])
        )
        
        # Estimate SM count for RTX 5000 series (120 SMs)
        if info['is_rtx_5000_series']:
            info['sm_count'] = 120  # RTX 5090/5080 have 120 SMs
        elif info['is_blackwell']:
            info['sm_count'] = 80   # Conservative estimate for other Blackwell
            
        logger.info(f"GPU detected: {info['name']} (compute {info['compute_capability']:.1f})")
        if info['is_blackwell']:
            logger.info(f"Blackwell GPU detected with ~{info['sm_count']} SMs")
            
    except Exception as e:
        logger.debug(f"Error getting GPU info: {e}")
    
    return info

def get_pytorch_install_command() -> str:
    """
    Get the appropriate PyTorch installation command based on A1111's strategy
    
    Returns:
        str: pip install command for PyTorch optimized for detected hardware
    """
    # Check environment variables first (A1111 approach)
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu128")
    torch_command = os.environ.get('TORCH_COMMAND')
    
    if torch_command:
        logger.info("Using custom TORCH_COMMAND from environment")
        return torch_command
    
    # Detect GPU and determine best PyTorch version
    gpu_info = get_gpu_info()
    
    if gpu_info['is_blackwell']:
        # Use A1111's current approach: PyTorch 2.7.0 with CUDA 12.8
        logger.info("Blackwell GPU detected - using optimized PyTorch 2.7.0")
        return f"pip install torch==2.7.0 torchvision==0.22.0 --extra-index-url {torch_index_url}"
    else:
        # Standard PyTorch for non-Blackwell GPUs
        logger.info("Non-Blackwell GPU detected - using standard PyTorch")
        return f"pip install torch torchvision --extra-index-url {torch_index_url}"

def verify_pytorch_blackwell_support() -> Tuple[bool, str]:
    """
    Verify that the current PyTorch installation supports Blackwell GPUs
    
    Returns:
        Tuple[bool, str]: (is_supported, status_message)
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "CUDA not available in PyTorch"
        
        gpu_info = get_gpu_info()
        
        if not gpu_info['is_blackwell']:
            return True, f"Non-Blackwell GPU detected: {gpu_info['name']}"
        
        # Test Blackwell GPU support
        try:
            # Try to create a tensor on GPU
            test_tensor = torch.randn(1, device='cuda')
            
            # Try a simple operation to verify kernel support
            result = test_tensor * 2
            
            return True, f"✅ Blackwell GPU fully supported: {gpu_info['name']} (PyTorch {torch.__version__})"
            
        except Exception as e:
            error_msg = str(e).lower()
            if "kernel image" in error_msg or "sm_120" in error_msg or "no kernel" in error_msg:
                return False, f"⚠️ Blackwell GPU detected but PyTorch lacks sm_120 support: {gpu_info['name']}"
            else:
                return False, f"❌ GPU error: {str(e)}"
                
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Error checking PyTorch support: {str(e)}"

def get_optimal_device() -> str:
    """
    Get the optimal device for computation based on GPU support
    
    Returns:
        str: 'cuda' if GPU is supported, 'cpu' otherwise
    """
    supported, message = verify_pytorch_blackwell_support()
    
    if supported and "CUDA not available" not in message:
        logger.info(f"Using GPU: {message}")
        return 'cuda'
    else:
        logger.warning(f"Using CPU: {message}")
        return 'cpu'

def print_system_info():
    """Print comprehensive system information for debugging"""
    print("=" * 60)
    print("NVIDIA RTX 5000 Series (Blackwell) GPU Support Status")
    print("=" * 60)
    
    # System info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # GPU info
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']:.1f}")
    print(f"Memory: {gpu_info['memory_gb']:.1f} GB")
    print(f"Is Blackwell: {gpu_info['is_blackwell']}")
    print(f"Is RTX 5000 Series: {gpu_info['is_rtx_5000_series']}")
    print(f"Estimated SMs: {gpu_info['sm_count']}")
    
    # PyTorch info
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Device Name: {torch.cuda.get_device_name()}")
            print(f"Device Capability: {torch.cuda.get_device_capability()}")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Support status
    supported, message = verify_pytorch_blackwell_support()
    print(f"GPU Support: {message}")
    
    # Recommended installation
    install_cmd = get_pytorch_install_command()
    print(f"Recommended PyTorch: {install_cmd}")
    
    print("=" * 60)

def setup_environment_variables():
    """Setup environment variables for optimal PyTorch installation"""
    if not os.environ.get('TORCH_INDEX_URL'):
        os.environ['TORCH_INDEX_URL'] = "https://download.pytorch.org/whl/cu128"
        logger.info("Set TORCH_INDEX_URL for CUDA 12.8 wheels")
    
    gpu_info = get_gpu_info()
    if gpu_info['is_blackwell'] and not os.environ.get('TORCH_COMMAND'):
        torch_cmd = get_pytorch_install_command()
        os.environ['TORCH_COMMAND'] = torch_cmd
        logger.info("Set TORCH_COMMAND for Blackwell GPU optimization")

# CLI interface for testing
if __name__ == "__main__":
    print_system_info()
