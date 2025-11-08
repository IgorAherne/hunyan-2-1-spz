from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Build Configuration

# Base C++ compiler flags for Windows
cxx_flags = ['/std:c++17', '/bigobj']

# Base nvcc flags, including performance and compatibility settings
nvcc_flags = [
    '--use_fast_math',
    '-O3',
    '--expt-relaxed-constexpr',
    '--allow-unsupported-compiler',
]

# Define stable architectures. We include sm_90 (Hopper) as it's well-supported
# by CUDA 12.8 and provides the best forward-compatibility via PTX 'compute_90'.
arch_flags = [
    '-gencode=arch=compute_61,code=sm_61',      # Pascal
    '-gencode=arch=compute_75,code=sm_75',      # Turing
    '-gencode=arch=compute_86,code=sm_86',      # Ampere
    '-gencode=arch=compute_89,code=sm_89',      # Ada Lovelace
    '-gencode=arch=compute_90,code=sm_90',      # Hopper
    '-gencode=arch=compute_90,code=compute_90'  # PTX for future GPUs
]
nvcc_flags.extend(arch_flags)

# Set the TORCH_CUDA_ARCH_LIST environment variable as a hint for PyTorch
# This mirrors the architectures we are building for.
os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1;7.5;8.6;8.9;9.0'

# CUDA Extension Definition

custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}
)

# Setup Call

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": BuildExtension},
)