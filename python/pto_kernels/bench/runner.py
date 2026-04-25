"""Side-by-side benchmark orchestration for PTO kernels."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from pto_kernels.config import repo_root

from .specs import KernelBenchmarkSpec, load_spec


def _load_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkRunner:
    def __init__(self, *, results_dir: Path | None = None, generated_dir: Path | None = None):
        self.repo_root = repo_root()
        self.results_dir = results_dir or self.repo_root / "bench" / "results"
        if generated_dir is not None:
            self.generated_dir = generated_dir
        elif results_dir is not None:
            self.generated_dir = results_dir / "generated"
        else:
            self.generated_dir = self.repo_root / "bench" / "generated"

    def _adapter_summary(self, adapter_path: str, spec: KernelBenchmarkSpec) -> dict[str, Any]:
        adapter_file = self.repo_root / adapter_path
        try:
            module = _load_module_from_path(adapter_file)
        except ImportError as exc:
            return {
                "adapter": adapter_path,
                "status": "blocked",
                "reason": f"adapter import failed: {exc}",
            }
        describe = getattr(module, "describe", None)
        summary = describe(self.repo_root, spec) if callable(describe) else {}
        summary["adapter"] = adapter_path
        return summary

    def _call_adapter(
        self,
        adapter_path: str,
        method_name: str,
        spec: KernelBenchmarkSpec,
        artifacts_dir: Path,
    ) -> dict[str, Any]:
        adapter_file = self.repo_root / adapter_path
        module = _load_module_from_path(adapter_file)
        method = getattr(module, method_name, None)
        if not callable(method):
            return {"status": "blocked", "reason": f"{method_name}() is not implemented"}
        try:
            return method(self.repo_root, spec, artifacts_dir)
        except NotImplementedError as exc:
            return {"status": "blocked", "reason": str(exc)}

    def run(
        self,
        spec_path: str | Path,
        *,
        dry_run: bool = False,
        capture_msprof: bool = False,
    ) -> dict[str, Any]:
        spec = load_spec(spec_path)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        artifacts_dir = self.results_dir / spec.family / spec.name / timestamp
        latest_dir = self.generated_dir / spec.family / spec.name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        report: dict[str, Any] = {
            "name": spec.name,
            "family": spec.family,
            "wave": spec.wave,
            "inventory_ref": spec.inventory_ref,
            "status": spec.status,
            "device": spec.device,
            "bench": asdict(spec.bench),
            "correctness": asdict(spec.correctness),
            "capture_msprof": capture_msprof,
            "baseline": self._adapter_summary(spec.baseline.adapter, spec),
            "pto": self._adapter_summary(spec.pto.adapter, spec),
            "artifacts_dir": str(artifacts_dir),
            "latest_artifacts_dir": str(latest_dir),
            "dry_run": dry_run,
        }

        if not dry_run:
            report["baseline"]["compile"] = self._call_adapter(
                spec.baseline.adapter, "compile_kernel", spec, artifacts_dir
            )
            report["pto"]["compile"] = self._call_adapter(
                spec.pto.adapter, "compile_kernel", spec, artifacts_dir
            )
            report["baseline"]["benchmark"] = self._call_adapter(
                spec.baseline.adapter, "benchmark", spec, artifacts_dir
            )
            report["pto"]["benchmark"] = self._call_adapter(
                spec.pto.adapter, "benchmark", spec, artifacts_dir
            )

        report_path = artifacts_dir / "report.json"
        latest_report_path = latest_dir / "report.json"
        report["latest_report_path"] = str(latest_report_path)
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        self._refresh_latest_artifacts(artifacts_dir, latest_dir)
        return report

    def _refresh_latest_artifacts(self, source_dir: Path, latest_dir: Path) -> None:
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(source_dir, latest_dir)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capture-msprof", action="store_true")
    args = parser.parse_args()

    runner = BenchmarkRunner()
    report = runner.run(args.spec, dry_run=args.dry_run, capture_msprof=args.capture_msprof)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
