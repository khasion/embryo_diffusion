#!/usr/bin/env python
import torch
import sys

print("=" * 60)
print("CUDA DIAGNOSTIC")
print("=" * 60)
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1e9:.2f} GB")
    print("✓ CUDA is working!")
else:
    print("✗ CUDA is NOT available")
    print("\nTroubleshooting:")
    print("1. Check if nvidia-smi works")
    print("2. Reinstall PyTorch with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("3. Or with conda: conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia")

print("=" * 60)
