from __future__ import annotations

from run_chunk_cumsum_dynamic_bsnd import main as run_chunk_cumsum_main
from run_chunk_h_dynamic_bsnd import main as run_chunk_h_main
from run_chunk_o_dynamic_bsnd import main as run_chunk_o_main
from run_scaled_dot_kkt_dynamic_bsnd import main as run_scaled_dot_kkt_main
from run_wy_fast_dynamic_bsnd import main as run_wy_fast_main


def main():
    print("`dynamic_bsnd` is being ported stage-by-stage onto PTO vector/tile kernels.")
    print("Implemented stages:")
    print("  - chunk_cumsum (native BSND + packed varlen)")
    print("  - scaled_dot_kkt (cube PTO kernel + exact NPU torch epilogue)")
    print("  - wy_fast (cube PTO matmul kernels + exact NPU torch packing epilogue)")
    print("  - chunk_h (PTO cube matmuls with host-side recurrent sequencing)")
    print("  - chunk_o (PTO qk/qs cube kernels + exact host gating/qkv epilogue)")
    print("")
    run_chunk_cumsum_main()
    print("")
    run_scaled_dot_kkt_main()
    print("")
    run_wy_fast_main()
    print("")
    run_chunk_h_main()
    print("")
    run_chunk_o_main()


if __name__ == "__main__":
    main()
