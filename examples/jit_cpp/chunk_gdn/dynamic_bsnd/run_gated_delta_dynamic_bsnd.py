from __future__ import annotations

from run_chunk_cumsum_dynamic_bsnd import main as run_chunk_cumsum_main


def main():
    print("`dynamic_bsnd` is being ported stage-by-stage onto PTO vector/tile kernels.")
    print("Implemented stage:")
    print("  - chunk_cumsum (native BSND + packed varlen)")
    print("")
    run_chunk_cumsum_main()
    print("")
    print("Remaining stages:")
    print("  - scaled_dot_kkt")
    print("  - wy_fast")
    print("  - chunk_h")
    print("  - chunk_o")


if __name__ == "__main__":
    main()
