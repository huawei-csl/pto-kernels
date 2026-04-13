from pathlib import Path

import pytest

from standard.jit_util_hadamard import jit_compile


@pytest.fixture(scope="session")
def hadamard_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "standard" / "fast_hadamard.cpp"
    return jit_compile(str(src), verbose=True, device=npu_device)
