#!/usr/bin/env python3
"""
Copy TileLang-dumped PTO sources from ../tilelang_codegen/kernels/ into *_kernel.cpp here,
applying the static_baseline transforms (include path + namespace).

Run after: ``../tilelang_codegen/scripts/dump_all_kernels.sh`` (needs NPU + TileLang JIT).
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TILELANG_KERNELS = os.path.join(_HERE, "..", "tilelang_codegen", "kernels")

_MAPPINGS = [
    ("opt_gdn_chunk_cumsum.cpp", "chunk_cumsum_kernel.cpp"),
    ("opt_gdn_chunk_scaled_dot_kkt.cpp", "scaled_dot_kkt_kernel.cpp"),
    ("opt_gdn_wy_fast.cpp", "wy_fast_kernel.cpp"),
    ("opt_gdn_chunk_h.cpp", "chunk_h_kernel.cpp"),
    ("opt_gdn_chunk_o.cpp", "chunk_o_kernel.cpp"),
]


def transform_tilelang_cpp(src: str) -> str:
    src = src.replace(
        '#include "tl_templates/pto/common.h"', '#include "common.h"'
    )
    out_lines = []
    for line in src.splitlines():
        if line.strip() == "#include <pto/pto-inst.hpp>":
            continue
        out_lines.append(line)
    src = "\n".join(out_lines)
    return src.replace("tl::ascend_pto::", "chunk_gdn_pto::")


def main():
    for src_name, dst_name in _MAPPINGS:
        src_path = os.path.join(_TILELANG_KERNELS, src_name)
        dst_path = os.path.join(_HERE, dst_name)
        if not os.path.isfile(src_path):
            raise FileNotFoundError(
                f"Missing {src_path!r}; run tilelang_codegen/scripts/dump_all_kernels.sh first."
            )
        with open(src_path, encoding="utf-8") as f:
            raw = f.read()
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(transform_tilelang_cpp(raw))
        print(f"Wrote {dst_path} (from {src_name})")


if __name__ == "__main__":
    main()
