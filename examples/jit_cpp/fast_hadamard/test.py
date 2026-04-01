import argparse
from pathlib import Path

import pytest


TARGET_TESTS = {
    "standard": ("standard/test_hadamard.py",),
    "quantize_int8": ("fuse_int8_quant/test_quantize.py",),
    "hadamard_quant_int8": ("fuse_int8_quant/test_hadamard_quant.py",),
    "quantize_int4": ("fuse_int4_quant/test_quantize.py",),
    "hadamard_quant_int4": ("fuse_int4_quant/test_hadamard_quant.py",),
    "hadamard_dynamic_quant_int4": (
        "fuse_int4_dynamic_quant/test_hadamard_dynamic_quant.py",
    ),
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run one or more fast_hadamard test suites."
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=["all", *TARGET_TESTS],
        help="Test target to run. Repeat to run multiple targets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test targets and exit.",
    )
    return parser.parse_known_args()


def _selected_targets(requested):
    if not requested or "all" in requested:
        return list(TARGET_TESTS)

    ordered = []
    seen = set()
    for target in requested:
        if target not in seen:
            ordered.append(target)
            seen.add(target)
    return ordered


def main():
    args, forwarded_args = _parse_args()

    if args.list:
        for target in TARGET_TESTS:
            print(target)
        return

    base = Path(__file__).resolve().parent
    test_paths = []
    for target in _selected_targets(args.target):
        test_paths.extend(str(base / rel_path) for rel_path in TARGET_TESTS[target])

    raise SystemExit(pytest.main([*test_paths, *forwarded_args]))


if __name__ == "__main__":
    main()
