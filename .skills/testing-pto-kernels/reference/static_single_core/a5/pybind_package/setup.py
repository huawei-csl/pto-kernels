from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
PACKAGE_DIR = Path(__file__).resolve().parent / "pto_static_a5_demo"


def build_kernel(name: str) -> Path:
    out = subprocess.check_output(["bash", str(ROOT / "compile.sh"), name], text=True)
    return Path(out.strip().splitlines()[-1])


class BuildWithKernels(BuildExtension):
    def build_extensions(self):
        add_lib = build_kernel("add")
        matmul_lib = build_kernel("matmul")
        PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        packaged_add = PACKAGE_DIR / add_lib.name
        packaged_matmul = PACKAGE_DIR / matmul_lib.name
        shutil.copy2(add_lib, packaged_add)
        shutil.copy2(matmul_lib, packaged_matmul)
        for ext in self.extensions:
            ext.extra_link_args = [
                *getattr(ext, "extra_link_args", []),
                str(packaged_add),
                str(packaged_matmul),
                "-Wl,-rpath,$ORIGIN",
            ]
        super().build_extensions()


setup(
    name="pto-static-a5-demo",
    version="0.0.0",
    packages=["pto_static_a5_demo"],
    package_dir={"pto_static_a5_demo": "pto_static_a5_demo"},
    package_data={
        "pto_static_a5_demo": ["libstatic_add_a5.so", "libstatic_matmul_a5.so"]
    },
    ext_modules=[
        CppExtension(
            "pto_static_a5_demo._C",
            [str(ROOT / "pybind.cpp")],
            extra_compile_args=["-O2", "-std=c++17"],
            runtime_library_dirs=[str(BUILD)],
        )
    ],
    cmdclass={"build_ext": BuildWithKernels},
)
