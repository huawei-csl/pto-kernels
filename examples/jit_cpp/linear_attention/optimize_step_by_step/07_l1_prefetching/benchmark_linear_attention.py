from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from jit_util_linear_attention import BLOCK_DIM, get_causal_mask, jit_compile
from linear_attention_shared import benchmark_cli, benchmark_dynamic_kernel

DEFAULT_SHAPES = [(24, 20, 2048, 128, 128), (48, 20, 1024, 128, 128), (12, 20, 8192, 128, 128), (24, 20, 1536, 128, 128)]

QUICK_SHAPES = [(8, 20, 1024, 128, 128), (16, 20, 1024, 128, 128)]

THROUGHPUT_HUNT_SHAPES = [(32, 20, 2048, 128, 128), (24, 20, 4096, 128, 128), (12, 20, 8192, 128, 128), (24, 20, 6144, 128, 128)]



def benchmark_shape(src: str, *, batch: int, heads: int, seq_len: int, hidden: int, chunk: int, warmup: int, repeats: int):
    return benchmark_dynamic_kernel(
        src,
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        hidden=hidden,
        chunk=chunk,
        warmup=warmup,
        repeats=repeats,
        jit_compile=jit_compile,
        block_dim=BLOCK_DIM,
        stage_count=2,
        use_mask=True,
        include_workspace_bytes=False,
        mask_factory=get_causal_mask,
    )


def main():
    benchmark_cli(
        script_file=__file__,
        default_shapes=DEFAULT_SHAPES,
        quick_shapes=QUICK_SHAPES,
        benchmark_shape=benchmark_shape,
        throughput_hunt_shapes=THROUGHPUT_HUNT_SHAPES
    )


if __name__ == "__main__":
    main()
