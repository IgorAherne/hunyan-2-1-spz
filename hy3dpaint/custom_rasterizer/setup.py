# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from packaging.version import Version

# Defines the base CUDA architectures to build for.
nvcc_args = [
    '-gencode=arch=compute_61,code=sm_61',      # Pascal (GTX 10xx)
    '-gencode=arch=compute_75,code=sm_75',      # Turing (RTX 20xx)
    '-gencode=arch=compute_86,code=sm_86',      # Ampere (RTX 30xx)
    '-gencode=arch=compute_89,code=sm_89',      # Ada Lovelace (RTX 40xx)
    '-gencode=arch=compute_89,code=compute_89'  # PTX for forward-compatibility
]

# Conditionally add Blackwell support if CUDA version is 12.8 or greater.
# This is based on findings from open-source projects handling early support.
if torch.version.cuda and Version(torch.version.cuda) >= Version("12.8"):
    nvcc_args.extend([
        '-gencode=arch=compute_120,code=sm_120',    # Native binary for Blackwell
        '-gencode=arch=compute_120,code=compute_120' # PTX for future GPUs
    ])

# build custom rasterizer
custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args={'nvcc': nvcc_args}
)

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
