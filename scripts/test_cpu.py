import os
import sys

# importing Torch is required for the importlib to subsequently import custom ops
import torch  # no-qa
import importlib

# Get the directory of the current script
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
so_path = os.path.join(root_path, "build/cpu_sim/pto_kernels_cpu.so")

# Import the module
spec = importlib.util.spec_from_file_location("pto_kernels_cpu", so_path)
if spec is None:
    raise AssertionError(f"Failed to create spec for pto_kernels_cpu at {so_path}")
module = importlib.util.module_from_spec(spec)
if not isinstance(spec.loader, importlib.abc.Loader):
    raise AssertionError("spec.loader is not a valid importlib Loader")
spec.loader.exec_module(module)

# Make sure the module uses the name "pto_kernels" so it can be imported with that name, overriding any real NPU pto_kernels module
sys.modules["pto_kernels"] = module

# Set the device to CPU
os.environ["NPU_DEVICE"] = "cpu"

# Run the actual tests
import pytest

test_dir = os.path.join(root_path, "tests")
pytest.main(
    [
        os.path.join(test_dir, "test_abs.py"),
        os.path.join(test_dir, "test_batch_matrix_square.py"),
        os.path.join(test_dir, "test_csr_gather.py"),
        os.path.join(test_dir, "test_scan_ul1.py"),
        os.path.join(test_dir, "test_simple_matmul.py"),
        os.path.join(test_dir, "test_swiglu.py"),
        os.path.join(test_dir, "test_tri_inv_ns.py"),
        os.path.join(test_dir, "test_tri_inv_trick.py"),
        # Disabled for now
        # os.path.join(test_dir, "test_tri_inv_col_sweep.py"),
        # os.path.join(test_dir, "test_tri_inv_rec_unroll.py"),
        # os.path.join(test_dir, "test_tri_inv_rec_unroll_variable_sequence_lengths.py"),
    ]
)
