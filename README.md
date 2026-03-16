# pto-kernels

A collection of high-performance custom kernels for **Ascend NPUs**, built on top of [pto-isa](https://github.com/PTO-ISA/pto-isa) — the Parallel Tile Operation virtual instruction set architecture designed by Ascend CANN.

PTO focuses on tile-level operations, enabling efficient, composable kernel development targeting Huawei's Ascend AI processors.

---

## Prerequisites

- A configured **torch-npu** environment
- Ascend toolkit installed at `/usr/local/Ascend/ascend-toolkit`

Run the one-time setup before building:

```bash
make setup_once
```

---

## Remove installation using pip

The repository is "pip installable", i.e.,

```bash
export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v git+https://github.com/huawei-csl/pto-dsl.git
```

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip3 install -r requirements.txt
make build_wheel
```

This produces an installable Python wheel:

```
pto_kernels-0.1.0-*.whl
```

---

## Installation

```bash
pip install --force-reinstall pto_kernels-*.whl
```

---

## Testing

```bash
make test
```

---

## Repository Structure

```
pto-kernels/
├── csrc/                  # C++ kernel source files
├── python/pto_kernels/    # Python bindings and utilities
├── examples/jit_cpp/      # JIT compilation examples
├── tests/                 # Test suite
├── scripts/               # Helper scripts
├── doxygen/               # API documentation config
└── CMakeLists.txt         # CMake build configuration
```

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

BSD-3-Clause-Clear — see [LICENSE](LICENSE) for details.
