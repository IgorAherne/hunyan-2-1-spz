import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# C++ compiler flags translated for MSVC (Windows).
# We'll use C++17 as it's a modern standard fully supported by PyTorch and MSVC.
# Release builds in MSVC are highly optimized by default, similar to -O3.
extra_cxx_flags = ['/std:c++17']

# On Windows, MSVC can require this for complex C++ templates.
if os.name == 'nt':
    extra_cxx_flags.append('/bigobj')

# Define the C++ extension module.
# PyTorch's build system automatically finds pybind11 headers.
mesh_inpaint_module = CppExtension(
    'mesh_inpaint_processor',
    sources=['mesh_inpaint_processor.cpp'],
    extra_compile_args=extra_cxx_flags
)

setup(
    name="differentiable_renderer_mesh_painter",
    version="0.1",
    author="Hunyuan-3D Team",
    description="Mesh painting/inpaint processor for the Differentiable Renderer",
    packages=find_packages(),
    ext_modules=[mesh_inpaint_module],
    cmdclass={
        'build_ext': BuildExtension
    }
)