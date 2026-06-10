"""Shared helpers for PTO kernel demo runners under reference/."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent import futures
from pathlib import Path

import torch
import torch_npu  # noqa: F401

REFERENCE_DIR = Path(__file__).resolve().parent

# Match torch.testing.assert_close defaults (see SKILL.md Correctness Rules).
RTOL_BY_DTYPE: dict[torch.dtype, float] = {
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
    torch.float32: 1.3e-6,
}
DEFAULT_ATOL = 1e-5
DEFAULT_SYNC_TIMEOUT_S = float(os.environ.get("PTO_SYNC_TIMEOUT_S", "60"))
DEFAULT_PROCESS_TIMEOUT_S = float(
    os.environ.get("PTO_PROCESS_TIMEOUT_S", "1800" if os.environ.get("PTO_SIMULATOR") == "1" else "60")
)


def add_reference_to_path(here: Path) -> Path:
    ref = here.parents[1]
    ref_str = str(ref)
    if ref_str not in sys.path:
        sys.path.insert(0, ref_str)
    return ref


def configure_torch_npu(*, simulator_safe: bool = False) -> None:
    torch.npu.config.allow_internal_format = False
    if simulator_safe:
        torch_npu.npu.set_compile_mode(jit_compile=False)


def tensor_ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def stream_ptr() -> ctypes.c_void_p:
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def stream_as_int() -> int:
    return int(torch.npu.current_stream().npu_stream)


def compile_kernel(compile_sh: Path, kernel: str) -> Path:
    out = subprocess.check_output(["bash", str(compile_sh), kernel], text=True)
    return Path(out.strip().splitlines()[-1])


def cube_core_count(device: str) -> int:
    props = torch.npu.get_device_properties(device)
    return int(getattr(props, "cube_core_num", getattr(props, "multi_processor_count", 1)))


def vector_core_count(device: str) -> int:
    props = torch.npu.get_device_properties(device)
    return int(getattr(props, "vector_core_num", cube_core_count(device) * 2))


def _dtype_of(value: torch.Tensor | torch.dtype) -> torch.dtype:
    return value.dtype if isinstance(value, torch.Tensor) else value


def rtol_for(value: torch.Tensor | torch.dtype) -> float:
    return RTOL_BY_DTYPE.get(_dtype_of(value), 1e-3)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert using SKILL.md thresholds unless explicit overrides are documented."""
    ref_dtype = expected.dtype
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol if rtol is not None else rtol_for(ref_dtype),
        atol=atol if atol is not None else DEFAULT_ATOL,
    )


def is_simulator_mode() -> bool:
    return os.environ.get("PTO_SIMULATOR", "").lower() not in ("", "0", "false", "no")


def device_repeats() -> int:
    if is_simulator_mode():
        return 1
    return int(os.environ.get("PTO_DEVICE_REPEATS", "5"))


def synchronize_device(timeout_s: float | None = None) -> None:
    timeout = DEFAULT_SYNC_TIMEOUT_S if timeout_s is None else timeout_s
    with futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(torch.npu.synchronize)
        try:
            fut.result(timeout=timeout)
        except futures.TimeoutError as exc:
            raise TimeoutError(
                f"torch.npu.synchronize() exceeded {timeout}s; possible sync deadlock"
            ) from exc


def run_repeated(launch_fn: Callable[[], None], *, repeats: int | None = None) -> None:
    """Launch a kernel repeatedly on real devices to surface sync bugs."""
    for _ in range(device_repeats() if repeats is None else repeats):
        launch_fn()
        synchronize_device()
