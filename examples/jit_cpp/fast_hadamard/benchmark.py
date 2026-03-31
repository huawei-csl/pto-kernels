import argparse
import importlib
import sys


CORE_TARGET_MODULES = {
    "standard": "standard.bench_hadamard",
    "copy_vs_hadamard": "copy_vs_hadamard.bench_copy_vs_hadamard",
    "quantize_int8": "fuse_int8_quant.bench_quantize",
    "hadamard_quant_int8": "fuse_int8_quant.bench_hadamard_quant",
    "quantize_int4": "fuse_int4_quant.bench_quantize",
    "hadamard_quant_int4": "fuse_int4_quant.bench_hadamard_quant",
}

TARGET_MODULES = {
    **CORE_TARGET_MODULES,
    "quantize_compare_int8_int4": "bench_quantize_compare",
    "hadamard_quant_compare_int8_int4": "bench_hadamard_quant_compare",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run one or more fast_hadamard benchmark suites."
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=["all", *TARGET_MODULES],
        help="Benchmark target to run. Repeat to run multiple targets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmark targets and exit.",
    )
    return parser.parse_known_args()


def _selected_targets(requested):
    if not requested or "all" in requested:
        return list(CORE_TARGET_MODULES)

    ordered = []
    seen = set()
    for target in requested:
        if target not in seen:
            ordered.append(target)
            seen.add(target)
    return ordered


def _dispatch(module_name: str, forwarded_args) -> None:
    module = importlib.import_module(module_name)
    saved_argv = sys.argv[:]
    sys.argv = [f"{module_name.rsplit('.', 1)[-1]}.py", *forwarded_args]
    try:
        module.main()
    finally:
        sys.argv = saved_argv


def main():
    args, forwarded_args = _parse_args()

    if args.list:
        for target in TARGET_MODULES:
            print(target)
        return

    for target in _selected_targets(args.target):
        print(f"\n=== benchmark target: {target} ===")
        _dispatch(TARGET_MODULES[target], forwarded_args)


if __name__ == "__main__":
    main()
