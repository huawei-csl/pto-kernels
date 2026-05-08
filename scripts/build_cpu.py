import os
import sys

from torch.utils.cpp_extension import load as load_cpp

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME", "")
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)

# Get the directory of the current script
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
build_path = os.path.join(root_path, "build/cpu_sim")
os.makedirs(build_path, exist_ok=True)

sources = [
    os.path.join(root_path, "csrc/host/pybind11.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_abs.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_batch_matrix_square.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_csr_gather.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_scan_ul1.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_simple_matmul.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_swiglu.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_tri_inv_col_sweep.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_tri_inv_ns.cpp"),
    # os.path.join(root_path, "csrc/kernel/kernel_tri_inv_rec_unroll.cpp"),
    os.path.join(root_path, "csrc/kernel/kernel_tri_inv_trick.cpp"),
]

module = load_cpp(
    name="pto_kernels_cpu",
    sources=sources,
    extra_cflags=[
        "-std=c++23",
        "-O2",
        "-fPIC",
        "-D__CPU_SIM",
        "-DSOC_VERSION=Ascend910B4",
        # This is not requred, but reduces the number of warnings
        "-Wno-narrowing",
    ],
    build_directory=build_path,
    extra_include_paths=[f"{PTO_LIB_PATH}/include"],
    verbose=True,
    is_python_module=True,
)

print("Built pto_kernels for CPU simulation.")

sys.modules["pto_kernels"] = module


# Sanity tests, import declared functions
print("Test if the module contains custom ops...")

from pto_kernels import (  # noqa
    pto_abs,
    pto_batch_matrix_square,
    pto_csr_gather,
    pto_scan_ul1,
    pto_simple_matmul,
    pto_swiglu,
    pto_tri_inv,
    pto_tri_inv_ns,
    # pto_tri_inv_rec_unroll,
    pto_tri_inv_trick,
)

print("Imported custom ops successfully.")
