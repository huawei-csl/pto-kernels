from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from jit_util_linear_attention import BLOCK_DIM, get_causal_mask, jit_compile
from linear_attention_shared import run_correctness_cases, run_dynamic_kernel


def run_kernel(src, q, k, v, chunk_size):
    return run_dynamic_kernel(
        src,
        q,
        k,
        v,
        chunk_size,
        jit_compile=jit_compile,
        block_dim=BLOCK_DIM,
        stage_count=1,
        use_mask=True,
        mask_factory=get_causal_mask,
    )


def main():
    test_configs = [(1, 2, 64, 128, 64), (1, 2, 256, 128, 64), (4, 2, 128, 128, 64), (8, 2, 512, 128, 64), (10, 2, 512, 128, 64), (16, 2, 256, 128, 64), (32, 2, 128, 128, 64), (1, 2, 1024, 128, 64), (8, 2, 2048, 128, 64), (2, 2, 4096, 128, 64), (16, 2, 1024, 128, 64), (50, 20, 128, 128, 64), (1, 2, 128, 128, 128), (4, 2, 512, 128, 128), (16, 20, 1024, 128, 128)]
    run_correctness_cases(__file__, test_configs, run_kernel)


if __name__ == "__main__":
    main()
