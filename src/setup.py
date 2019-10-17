from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import sysconfig


### Run this witht he command:
### python setup.py build_ext --inplace
flags = []

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra"]
extra_compile_args += ["--expt-relaxed-constexpr"]

nvcc_extra_args = [
    "--expt-relaxed-constexpr",
    "-O2",
    "--gpu-architecture=sm_61",
    "-lineinfo",
]

setup(
    name="bfpactivation_cpu",
    ext_modules=[
        CppExtension(
            "bfpactivation_cpu",
            sources=["bfpactivation_cpu.cpp"],
            extra_compile_args=flags,
        )
    ],
    extra_compile_args=extra_compile_args,
    cmdclass={"build_ext": BuildExtension},
)

setup(
    name="bfpactivation_cuda",
    ext_modules=[
        CUDAExtension(
            "bfpactivation_cuda",
            sources=["bfpactivation_cuda.cpp", "bfpactivation_cuda_kernel.cu"],
            extra_compile_args={"cxx": flags, "nvcc": flags + nvcc_extra_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
