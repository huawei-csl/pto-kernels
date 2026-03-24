import importlib
import sys


COMPARE_TARGETS = (
    ("hadamard_quant_int8", "fuse_int8_quant.bench_hadamard_quant"),
    ("hadamard_quant_int4", "fuse_int4_quant.bench_hadamard_quant"),
)


def _dispatch(module_name: str, forwarded_args) -> None:
    module = importlib.import_module(module_name)
    saved_argv = sys.argv[:]
    sys.argv = [f"{module_name.rsplit('.', 1)[-1]}.py", *forwarded_args]
    try:
        module.main()
    finally:
        sys.argv = saved_argv


def main():
    forwarded_args = sys.argv[1:]

    print(
        "Running both fused Hadamard+quant benchmark suites for later int8/int4 comparison."
    )
    for label, module_name in COMPARE_TARGETS:
        print(f"\n=== compare benchmark component: {label} ===")
        _dispatch(module_name, forwarded_args)


if __name__ == "__main__":
    main()
