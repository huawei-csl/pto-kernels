# scripts

Build and packaging helper scripts for `pto_kernels`.

## build.sh

Compiles the project locally using CMake against an Ascend CANN toolkit installation.

**Usage**

```bash
./scripts/build.sh [--soc-version <version>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-v`, `--soc-version` | `Ascend910B4` | Target SoC version passed to CMake as `-DSOC_VERSION` |

The script resolves the Ascend toolkit path from `$ASCEND_INSTALL_PATH`, `$ASCEND_HOME_PATH`, `~/Ascend/ascend-toolkit/latest`, or `/usr/local/Ascend/ascend-toolkit/latest` (in that order), sources its environment, then runs a clean CMake configure + build.

## gen-wheel.sh

Builds a `manylinux` Python wheel inside a Docker container, replicating the steps from `.github/workflows/python-packaging.yml`.

The container image used is `quay.io/ascend/manylinux:9.0.0-910b-manylinux_2_28-<python_version>`. The repo is mounted at `/workspace` and the following steps are executed inside the container:

1. Source Ascend toolkit and ATB environment files.
2. Install build dependencies (`pyyaml`, `setuptools`, `pytest`, `packaging`, `pybind11[global]`, `requirements.txt`).
3. Run `make clean wheel`.
4. Install and inspect the built wheel.
5. Run `auditwheel repair` with the appropriate `--exclude` flags for Ascend/PyTorch shared libraries.
6. Write the repaired wheel to `wheelhouse/`.

**Usage**

```bash
./scripts/gen-wheel.sh [--arch x86_64|aarch64] [--py-ver 310|311]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--arch` | `x86_64` | Target architecture (`x86_64` or `aarch64`) |
| `--py-ver` | `310` | Python version without dots (`310` or `311`) |

**Examples**

```bash
# x86_64, Python 3.10 (defaults)
./scripts/gen-wheel.sh

# aarch64, Python 3.11
./scripts/gen-wheel.sh --arch aarch64 --py-ver 311
```

The repaired wheel is written to `wheelhouse/pto_kernels*manylinux*.whl` in the repository root.
