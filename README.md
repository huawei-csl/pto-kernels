# pto-kernels

[![PyPI version](https://img.shields.io/pypi/v/pto-kernels.svg)](https://pypi.org/project/pto-kernels/)
[![Python versions](https://img.shields.io/pypi/pyversions/pto-kernels.svg)](https://pypi.org/project/pto-kernels/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause--Clear-blue.svg)](LICENSE)

A collection of high-performance custom kernels for **Ascend NPUs**, built on top of [pto-isa](https://github.com/PTO-ISA/pto-isa) — the Parallel Tile Operation virtual instruction set architecture designed by Ascend CANN.

PTO focuses on tile-level operations, enabling efficient, composable kernel development targeting Huawei's Ascend AI processors, and ships as ready-to-use PyTorch (`torch-npu`) operators.

---

## Why pto-kernels?

- **Fast** — hand-tuned tile-level kernels for Ascend NPUs, benchmarked against `torch-npu` built-ins.
- **Drop-in** — kernels are exposed as plain Python functions that operate on `torch_npu` tensors.
- **Broad coverage** — everything from elementwise ops (`abs`, `swiglu`) to linear-algebra primitives (`tri_inv`, `matmul`) to gated-linear-attention building blocks (GDN, KDA chunked recurrence, WY representation, KKT).
- **Extensible** — built on [pto-isa](https://github.com/PTO-ISA/pto-isa), so new kernels can be written once at the tile level and reused across ops.

---

## Installation

### From PyPI (recommended)

Prebuilt wheels are published to [PyPI](https://pypi.org/project/pto-kernels/) for Python 3.10–3.12 on `x86_64` and `aarch64`:

```bash
pip install pto-kernels
```

> Requires a working `torch-npu` + Ascend CANN runtime environment to run kernels on-device; the package itself installs without one.

### From source

Building from source is only needed for unreleased kernels or if you plan to contribute.

**Prerequisites:**
- A configured **torch-npu** environment
- Ascend toolkit installed at `/usr/local/Ascend/ascend-toolkit`

```bash
# One-time setup
make setup_once

# Install directly from GitHub
export CMAKE_GENERATOR="Unix Makefiles"
pip install -v git+https://github.com/huawei-csl/pto-kernels.git
```

Or build a wheel locally:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip3 install -r requirements.txt
make wheel                          # produces pto_kernels-X.Y.Z-*.whl
pip install --force-reinstall pto_kernels-*.whl
```

---

## Quickstart

```python
import torch
import torch_npu  # noqa
from pto_kernels import pto_swiglu

x = torch.randn(4, 2048, device="npu", dtype=torch.float16)
y = pto_swiglu(x)  # fused SwiGLU on Ascend NPU
```

## Available kernels

| Category | Kernels |
| --- | --- |
| Elementwise / activation | `pto_abs`, `pto_swiglu` |
| Attention | `pto_fa` |
| Linear algebra | `pto_simple_matmul`, `pto_batch_matrix_square`, `pto_tri_inv`, `pto_tri_inv_ns`, `pto_tri_inv_rec_unroll`, `pto_tri_inv_trick` |
| Scan / gather | `pto_scan_ul1`, `pto_csr_gather` |
| GDN (Gated DeltaNet) | `pto_gdn_chunk_cumsum`, `pto_gdn_chunk_o`, `pto_gdn_scaled_dot_kkt`, `pto_gdn_wy_fast` |
| KDA (Kimi Delta Attention) | `pto_kda_chunk_h`, `pto_kda_chunk_o`, `pto_kda_gate_cumsum`, `pto_kda_kkt`, `pto_kda_wy` |

More end-to-end usage patterns live under [`examples/`](examples) (JIT C++ kernels, AI CPU custom ops) and [`tests/`](tests) (correctness against reference/`torch_npu` implementations).

---

## Testing

```bash
make test
```

---

## Repository structure

```
pto-kernels/
├── csrc/                  # C++ kernel source files
├── python/pto_kernels/    # Python bindings and utilities
├── examples/jit_cpp/      # JIT compilation examples
├── examples/aicpu/        # AI CPU custom-op examples
├── tests/                 # Test suite
├── scripts/               # Helper scripts
├── doxygen/               # API documentation config
└── CMakeLists.txt         # CMake build configuration
```

---

## Contributing

Contributions are welcome! Whether it's a new kernel, a bug fix, or a benchmark, please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Release process

See [RELEASE.md](RELEASE.md) for how new versions are cut and published.

## License

BSD-3-Clause-Clear — see [LICENSE](LICENSE) for details.
