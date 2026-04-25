#!/usr/bin/env python3
"""Validate the local 910B PTO environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pto_kernels.utils import detect_env, inspect_ops_transformer_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if the environment does not match the expected 910B target.",
    )
    args = parser.parse_args()

    env = detect_env()
    ops_runtime = inspect_ops_transformer_runtime(toolkit_home=env.toolkit_home)

    if args.json:
        payload = {
            "environment": json.loads(env.to_json()),
            "ops_transformer_runtime": json.loads(ops_runtime.to_json()),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"toolkit_home      : {env.toolkit_home}")
        print(f"toolkit_version   : {env.toolkit_version}")
        print(f"ptoas_path        : {env.ptoas_path}")
        print(f"bisheng_path      : {env.bisheng_path}")
        print(f"torch_npu         : {env.torch_npu_available}")
        print(f"npu_model         : {env.npu_model}")
        print(f"npu_count         : {env.npu_count}")
        print(f"soc_target        : {env.soc_target}")
        print(f"pto_arch          : {env.pto_arch}")
        print(f"npu_arch          : {env.npu_arch}")
        print(f"ops_pkg_metadata  : {ops_runtime.build_dependency_metadata_present}")
        print(f"ops_pkg_effective : {ops_runtime.effective_package_path}")
        print(f"ops_pkg_compat    : {ops_runtime.compat_build_dependency_metadata_present}")
        print(f"ops_pkg_installed : {ops_runtime.package_installed}")
        print(f"ops_pkg_runfiles  : {len(ops_runtime.package_runfiles)}")
        if env.warnings:
            print("warnings:")
            for warning in env.warnings:
                print(f"  - {warning}")

    if args.strict:
        required = [
            env.soc_target == "ascend910b",
            env.pto_arch == "a3",
            env.npu_arch == "dav-2201",
            env.ptoas_path is not None,
            env.bisheng_path is not None,
        ]
        if not all(required):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
