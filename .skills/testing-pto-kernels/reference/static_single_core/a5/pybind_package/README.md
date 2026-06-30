# Static A5 Pybind Package

Minimal pip-installable torch extension package for the static A5 demo.

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a5/pybind_package
python3 -m pip install --no-build-isolation -e .
```

The build step first compiles `../add.cpp` and `../matmul.cpp` with `bisheng`,
copies the generated kernel shared libraries into the Python package, then links
the torch C++ extension with `$ORIGIN` rpath so `_C` can import after install.
The public interface accepts torch tensors, not raw `data_ptr()` integers.

`--no-build-isolation` is intentional: the extension must use the same already
installed `torch` and `torch_npu` environment as the runtime.
