import torch
import os

print('PyTorch CUDA Info:')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  PyTorch version: {torch.__version__}')
print(f'  Device count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'  Current device: {torch.cuda.current_device()}')
    print(f'  Device name: {torch.cuda.get_device_name(0)}')

print('\nEnvironment:')
print(f'  CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")}')
print(f'  HOME: {os.environ.get("HOME", "Not set")}')
print(f'  USER: {os.environ.get("USER", "Not set")}')