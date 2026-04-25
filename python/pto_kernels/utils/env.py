"""Environment discovery for the PTO 910B workflow."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


EXPECTED_SOC = "ascend910b"
EXPECTED_PTO_ARCH = "a3"
EXPECTED_NPU_ARCH = "dav-2201"
EXPECTED_MODEL_PREFIX = "910B"


@dataclass
class DetectedEnv:
    toolkit_home: str | None
    toolkit_version: str | None
    ptoas_path: str | None
    bisheng_path: str | None
    torch_npu_available: bool
    npu_model: str | None
    npu_count: int
    soc_target: str | None
    pto_arch: str | None
    npu_arch: str | None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def parse_npu_smi_output(output: str) -> tuple[str | None, int]:
    model_re = re.compile(r"^\|\s*\d+\s+([A-Za-z0-9._-]+)\s+\|", re.MULTILINE)
    models = model_re.findall(output)
    if not models:
        return None, 0
    return models[0], len(models)


def _map_model(model: str | None) -> tuple[str | None, str | None, str | None, list[str]]:
    warnings: list[str] = []
    if not model:
        warnings.append("No NPU model detected from npu-smi output.")
        return None, None, None, warnings
    if model.startswith(EXPECTED_MODEL_PREFIX):
        return EXPECTED_SOC, EXPECTED_PTO_ARCH, EXPECTED_NPU_ARCH, warnings
    warnings.append(
        f"Detected NPU model '{model}' does not match expected {EXPECTED_MODEL_PREFIX} family."
    )
    return None, None, None, warnings


def _find_toolkit_home() -> str | None:
    candidates = [
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        os.environ.get("ASCEND_HOME_PATH"),
        str(Path.home() / "Ascend" / "cann"),
        "/usr/local/Ascend/cann",
        str(Path.home() / "Ascend" / "ascend-toolkit" / "latest"),
        "/usr/local/Ascend/ascend-toolkit/latest",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _read_version(toolkit_home: str | None) -> str | None:
    if not toolkit_home:
        return None
    version_cfg = Path(toolkit_home) / "version.cfg"
    if version_cfg.exists():
        content = version_cfg.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"toolkit_running_version=\[[^:]+:([^\]]+)\]", content)
        if match:
            return match.group(1)
        match = re.search(r"toolkit_installed_version=\[[^:]+:([^\]]+)\]", content)
        if match:
            return match.group(1)
    install_info = Path(toolkit_home) / "aarch64-linux" / "ascend_toolkit_install.info"
    if install_info.exists():
        content = install_info.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^version=(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    runtime_version = Path(toolkit_home) / "share" / "info" / "runtime" / "version.info"
    if runtime_version.exists():
        content = runtime_version.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^Version=(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def _tool_on_path(name: str) -> str | None:
    return shutil.which(name)


def _torch_npu_available() -> bool:
    try:
        subprocess.run(
            ["python3", "-c", "import torch_npu"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def detect_env() -> DetectedEnv:
    warnings: list[str] = []
    npu_model = None
    npu_count = 0
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            capture_output=True,
            text=True,
        )
        npu_model, npu_count = parse_npu_smi_output(completed.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        warnings.append(f"Unable to run npu-smi info: {exc}")

    soc_target, pto_arch, npu_arch, map_warnings = _map_model(npu_model)
    warnings.extend(map_warnings)

    toolkit_home = _find_toolkit_home()
    if not toolkit_home:
        warnings.append("Unable to locate Ascend toolkit installation.")
    toolkit_version = _read_version(toolkit_home)
    if toolkit_version is None:
        warnings.append("Unable to read toolkit version from version metadata.")

    ptoas_path = _tool_on_path("ptoas")
    bisheng_path = _tool_on_path("bisheng")
    if not ptoas_path:
        warnings.append("ptoas is not on PATH.")
    if not bisheng_path:
        warnings.append("bisheng is not on PATH.")

    return DetectedEnv(
        toolkit_home=toolkit_home,
        toolkit_version=toolkit_version,
        ptoas_path=ptoas_path,
        bisheng_path=bisheng_path,
        torch_npu_available=_torch_npu_available(),
        npu_model=npu_model,
        npu_count=npu_count,
        soc_target=soc_target,
        pto_arch=pto_arch,
        npu_arch=npu_arch,
        warnings=warnings,
    )
