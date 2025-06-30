#!/usr/bin/env python3
"""
PyTorch Installer for NVIDIA RTX 5000 Series (Blackwell) Support

This script detects your GPU and installs the appropriate PyTorch version
following the A1111 approach for optimal Blackwell GPU support.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

# Add project root to path for blackwell_support import
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from blackwell_support import (
        get_gpu_info, 
        get_pytorch_install_command, 
        verify_pytorch_blackwell_support,
        print_system_info,
        setup_environment_variables
    )
    BLACKWELL_SUPPORT_AVAILABLE = True
except ImportError as e:
    BLACKWELL_SUPPORT_AVAILABLE = False
    print(f"Error: Could not import blackwell_support module: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

def check_current_pytorch():
    """Check if PyTorch is currently installed and get version info"""
    try:
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            try:
                # Handle different PyTorch versions - use try/except for version attribute
                try:
                    import torch.version
                    cuda_version = torch.version.cuda
                except (AttributeError, ImportError):
                    cuda_version = 'unknown'
                print(f"CUDA available: {cuda_version}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
            except Exception as e:
                print(f"CUDA available but error getting details: {e}")
        else:
            print("CUDA not available in current PyTorch installation")
        
        return True
    except ImportError:
        print("PyTorch is not currently installed")
        return False

def install_pytorch(command: str, dry_run: bool = False):
    """Install PyTorch using the provided command"""
    print(f"\nInstallation command: {command}")
    
    if dry_run:
        print("(Dry run - not actually installing)")
        return True
    
    print("\nStarting PyTorch installation...")
    try:
        # Run the main installation command
        result = subprocess.run(
            command.split(),
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ PyTorch installation completed successfully!")
        
        # Check for torchaudio conflicts and fix them
        print("\n🔍 Checking for torchaudio compatibility...")
        try:
            # Check if torchaudio is installed and potentially incompatible
            check_result = subprocess.run(
                [sys.executable, "-c", "import torchaudio; print(torchaudio.__version__)"],
                capture_output=True,
                text=True
            )
            
            if check_result.returncode == 0:
                torchaudio_version = check_result.stdout.strip()
                print(f"Found torchaudio version: {torchaudio_version}")
                
                # If we installed PyTorch 2.7.0, ensure torchaudio matches
                if "torch==2.7.0" in command and not torchaudio_version.startswith("2.7.0"):
                    print("⚠️ Torchaudio version mismatch detected, upgrading...")
                    
                    # Uninstall old torchaudio
                    subprocess.run([sys.executable, "-m", "pip", "uninstall", "torchaudio", "-y"], check=True)
                    
                    # Install compatible torchaudio
                    torchaudio_cmd = command.replace("torch==2.7.0 torchvision==0.22.0", "torchaudio==2.7.0")
                    subprocess.run(torchaudio_cmd.split(), check=True)
                    print("✅ Torchaudio upgraded to compatible version!")
                else:
                    print("✅ Torchaudio version is compatible")
            else:
                print("ℹ️ Torchaudio not installed (optional)")
                
        except Exception as e:
            print(f"⚠️ Could not check torchaudio compatibility: {e}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_virtual_environment():
    """Check if we're running in a virtual environment and provide info"""
    venv_info = {
        'in_venv': False,
        'venv_path': None,
        'venv_name': None,
        'python_path': sys.executable
    }
    
    # Check for virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_info['in_venv'] = True
        venv_info['venv_path'] = sys.prefix
        venv_info['venv_name'] = os.path.basename(sys.prefix)
    
    # Check for conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        venv_info['in_venv'] = True
        venv_info['venv_name'] = conda_env
        venv_info['venv_path'] = os.environ.get('CONDA_PREFIX', sys.prefix)
    
    return venv_info

def detect_project_venv():
    """Try to detect if there's a virtual environment in the project directory"""
    project_root = Path(__file__).parent
    
    # Common venv directory names
    venv_names = ['venv', 'env', '.venv', '.env', 'virtualenv']
    
    for venv_name in venv_names:
        venv_path = project_root / venv_name
        if venv_path.exists():
            # Check if it looks like a Python venv
            if (venv_path / 'Scripts' / 'python.exe').exists():  # Windows
                return venv_path / 'Scripts' / 'python.exe'
            elif (venv_path / 'bin' / 'python').exists():  # Unix/Linux
                return venv_path / 'bin' / 'python'
    
    return None

def main():
    """Main installer function"""
    print("=" * 60)
    print("NVIDIA RTX 5000 Series (Blackwell) PyTorch Installer")
    print("=" * 60)
    
    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    
    if dry_run:
        print("Running in DRY RUN mode - no actual installation will occur")
    
    # Check virtual environment status
    venv_info = check_virtual_environment()
    project_venv = detect_project_venv()
    
    print(f"\n🐍 Python Environment Information:")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python version: {sys.version.split()[0]}")
    
    if venv_info['in_venv']:
        print(f"   ✅ Virtual environment: {venv_info['venv_name']}")
        print(f"   📁 Environment path: {venv_info['venv_path']}")
    else:
        print(f"   ⚠️ No virtual environment detected - using global Python")
        if project_venv:
            print(f"   💡 Found project venv at: {project_venv}")
            print(f"   🔧 Consider activating it first:")
            if platform.system() == "Windows":
                print(f"      {project_venv.parent}\\activate.bat")
            else:
                print(f"      source {project_venv.parent}/activate")
            
            response = input(f"\n❓ Do you want to continue with global Python? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Please activate your virtual environment and run the installer again.")
                return
    
    # Show current system info
    print("\n📋 System Information:")
    print_system_info()
    
    # Check current PyTorch installation
    print("\n🔍 Checking current PyTorch installation...")
    pytorch_installed = check_current_pytorch()
    
    # Get GPU information
    gpu_info = get_gpu_info()
    
    # Check if current installation supports Blackwell (if PyTorch is installed)
    if pytorch_installed:
        supported, message = verify_pytorch_blackwell_support()
        print(f"\n🎯 Current Support Status: {message}")
        
        if supported and gpu_info['is_blackwell'] and not force:
            print("\n✅ Your current PyTorch installation already supports your Blackwell GPU!")
            print("Use --force flag to reinstall anyway.")
            return
    
    # Get recommended installation command
    install_command = get_pytorch_install_command()
    
    print(f"\n💡 Recommended installation:")
    print(f"   {install_command}")
    
    if gpu_info['is_blackwell']:
        print(f"\n🚀 Blackwell GPU detected: {gpu_info['name']}")
        if gpu_info['is_rtx_5000_series']:
            print(f"   Optimized for {gpu_info['sm_count']} SM configuration")
    else:
        print(f"\n🔧 Non-Blackwell GPU detected: {gpu_info['name']}")
        print("   Using standard PyTorch installation")
    
    # Confirm installation
    if not dry_run:
        response = input("\n❓ Proceed with installation? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Installation cancelled.")
            return
    
    # Setup environment variables
    setup_environment_variables()
    
    # Perform installation
    success = install_pytorch(install_command, dry_run)
    
    if success and not dry_run:
        print("\n🔄 Verifying installation...")
        
        # Verify the new installation
        try:
            # Remove any cached imports
            if 'torch' in sys.modules:
                del sys.modules['torch']
            
            import torch
            print(f"✅ New PyTorch version: {torch.__version__}")
            
            if torch.cuda.is_available():
                try:
                    import torch.version
                    cuda_version = torch.version.cuda
                except (AttributeError, ImportError):
                    cuda_version = 'unknown'
                print(f"✅ CUDA version: {cuda_version}")
                
                # Test Blackwell support if applicable
                if gpu_info['is_blackwell']:
                    try:
                        test_tensor = torch.randn(1, device='cuda')
                        result = test_tensor * 2
                        print("✅ Blackwell GPU support verified!")
                    except Exception as e:
                        print(f"⚠️ Blackwell GPU test failed: {e}")
            else:
                print("⚠️ CUDA not available in new installation")
                
        except ImportError as e:
            print(f"❌ Could not verify new installation: {e}")
    
    print("\n" + "=" * 60)
    print("Installation process completed!")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and "--help" in sys.argv:
        print("Usage: python install_pytorch_blackwell.py [OPTIONS]")
        print("\nOptions:")
        print("  --dry-run    Show what would be installed without actually installing")
        print("  --force      Force reinstallation even if current version supports GPU")
        print("  --help       Show this help message")
        print("\nThis script automatically detects your GPU and installs the appropriate")
        print("PyTorch version for optimal performance on NVIDIA RTX 5000 series GPUs.")
        sys.exit(0)
    
    main()
