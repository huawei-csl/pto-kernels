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
    print("  - scaled_dot_kkt (fused PTO cube+vector kernel)")
    print("  - wy_fast (PTO cube matmuls + Torch fallback for dynamic A1/A2 build)")
    print("  - chunk_h (PTO cube matmuls + host-side recurrent sequencing)")
    print("  - chunk_o (fully fused PTO cube+vector kernel)")
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
