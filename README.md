# pto-isa kernels

Ascend NPU kernels using [pto-isa](https://gitcode.com/cann/pto-isa/). Parallel Tile Operation (PTO) is a virtual instruction set architecture designed by Ascend CANN, focusing on tile-level operations.

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip3 install pyyaml setuptools pytest packaging
pip3 install -r requirements.txt
make build_wheel
```

The above commands will generate a wheel (i.e., `pto_kernels-0.1.0-cp310-cp310-linux_x86_64.whl`) that is pip installable.

### Installing

```bash
pip install --force-reinstall pto_isa_kernels-*.whl
```

```bash
make test
```
