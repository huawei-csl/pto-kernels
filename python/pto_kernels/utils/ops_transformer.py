"""Helpers for ops-transformer seed packages in the local workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil

from pto_kernels.config import resolve_workspace_repo


SEED_OPS = [
    "apply_rotary_pos_emb",
    "grouped_matmul",
    "ffn",
    "moe_token_permute",
    "flash_attention_score",
    "matmul_reduce_scatter",
]

REQUIRED_BUILD_INFO_PACKAGES = [
    "runtime",
    "opbase",
    "hcomm",
    "ge-executor",
    "metadef",
    "ge-compiler",
    "asc-devkit",
    "bisheng-compiler",
    "asc-tools",
]

PACKAGE_VERSION_INFO_FALLBACKS = {
    "runtime": ("runtime/version.info",),
    "opbase": ("toolkit/version.info", "runtime/version.info"),
    "hcomm": ("hccl/version.info",),
    "ge-executor": ("compiler/version.info",),
    "metadef": ("compiler/version.info",),
    "ge-compiler": ("compiler/version.info",),
    "asc-devkit": ("toolkit/version.info",),
    "bisheng-compiler": ("bisheng_toolkit/version.info", "compiler/version.info"),
    "asc-tools": ("toolkit/version.info", "tools/aoe/version.info", "tools/ncs/version.info"),
}


def _infer_install_root(toolkit_home: str | None) -> Path | None:
    if not toolkit_home:
        return None
    path = Path(toolkit_home).resolve()
    parts = path.parts
    if "ascend-toolkit" in parts:
        idx = parts.index("ascend-toolkit")
        if idx > 0:
            return Path(*parts[:idx])
    return path.parent


def _discover_runfiles(build_out: Path) -> list[Path]:
    if not build_out.exists():
        return []
    patterns = (
        "cann-*-ops-transformer_*_linux-*.run",
        "cann-ops-transformer*.run",
        "cann-*-ops-transformer*.run",
    )
    runfiles: set[Path] = set()
    for pattern in patterns:
        runfiles.update(build_out.glob(pattern))
    return sorted(runfiles)


def _discover_vendor_root(toolkit_home: str | None) -> Path | None:
    if not toolkit_home:
        return None
    toolkit_root = Path(toolkit_home)
    candidates = (
        toolkit_root / "vendors" / "custom_transformer",
        toolkit_root / "opp" / "vendors" / "custom_transformer",
        toolkit_root / "opp" / "vendors",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def required_version_info_paths(package_path: str | None) -> list[Path]:
    if not package_path:
        return []
    root = Path(package_path)
    return [root / "share" / "info" / pkg / "version.info" for pkg in REQUIRED_BUILD_INFO_PACKAGES]


def _compat_root_default() -> Path:
    repo_root = resolve_workspace_repo("pto-kernels")
    if repo_root is not None:
        return repo_root / "build" / "ops_transformer_cann_compat"
    return Path.cwd() / "build" / "ops_transformer_cann_compat"


def _resolve_version_info_source(toolkit_home: Path, package_name: str) -> Path | None:
    candidates = PACKAGE_VERSION_INFO_FALLBACKS.get(package_name, ())
    for relative_path in candidates:
        candidate = toolkit_home / relative_path
        if candidate.exists():
            return candidate.resolve()
    return None


def _normalize_version_string(raw_version: str, version_dir: str | None) -> str:
    if version_dir:
        match = re.search(r"\d+\.\d+\.\d+", version_dir)
        if match:
            return match.group(0)
        match = re.search(r"\d+\.\d+", version_dir)
        if match:
            return f"{match.group(0)}.0"

    match = re.search(r"\d+\.\d+\.\d+", raw_version)
    if match:
        return match.group(0)
    match = re.search(r"\d+\.\d+", raw_version)
    if match:
        return f"{match.group(0)}.0"
    return raw_version


def _render_compat_version_info(source: Path) -> str:
    lines = source.read_text(encoding="utf-8").splitlines()
    raw_version = ""
    version_dir = None
    for line in lines:
        if line.startswith("Version="):
            raw_version = line.split("=", 1)[1].strip()
        elif line.startswith("version_dir="):
            version_dir = line.split("=", 1)[1].strip()

    normalized_version = _normalize_version_string(raw_version, version_dir)
    rendered_lines = []
    version_written = False
    for line in lines:
        if line.startswith("Version="):
            rendered_lines.append(f"Version={normalized_version}")
            version_written = True
        else:
            rendered_lines.append(line)
    if not version_written:
        rendered_lines.insert(0, f"Version={normalized_version}")
    return "\n".join(rendered_lines) + "\n"


def compat_required_version_info_paths(compat_root: str | os.PathLike[str] | None) -> list[Path]:
    if compat_root is None:
        return []
    return required_version_info_paths(str(compat_root))


def prepare_compat_package_path(
    *,
    toolkit_home: str | None,
    output_root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> Path | None:
    if not toolkit_home:
        return None

    toolkit_root = Path(toolkit_home).resolve()
    if not toolkit_root.exists():
        return None

    compat_root = Path(output_root) if output_root else _compat_root_default()
    compat_root.mkdir(parents=True, exist_ok=True)

    if force:
        share_dir = compat_root / "share"
        if share_dir.exists() or share_dir.is_symlink():
            if share_dir.is_symlink() or share_dir.is_file():
                share_dir.unlink()
            else:
                shutil.rmtree(share_dir)

    for child in toolkit_root.iterdir():
        destination = compat_root / child.name
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(child)

    share_info_root = compat_root / "share" / "info"
    share_info_root.mkdir(parents=True, exist_ok=True)

    for package_name in REQUIRED_BUILD_INFO_PACKAGES:
        source = _resolve_version_info_source(toolkit_root, package_name)
        if source is None:
            continue
        package_dir = share_info_root / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        version_info = package_dir / "version.info"
        if version_info.exists() or version_info.is_symlink():
            version_info.unlink()
        version_info.write_text(_render_compat_version_info(source), encoding="utf-8")

    return compat_root


@dataclass
class OpsTransformerRuntimeStatus:
    ops_transformer_root: str | None
    toolkit_home: str | None
    install_root: str | None
    build_out: str | None
    seed_ops: list[str]
    package_runfiles: list[str]
    package_path: str | None
    effective_package_path: str | None
    compat_package_path: str | None
    share_info_dir: str | None
    uninstall_script: str | None
    vendors_dir: str | None
    vendors_config: str | None
    binary_info_configs: list[str]
    required_version_infos: list[str]
    missing_version_infos: list[str]
    compat_required_version_infos: list[str]
    compat_missing_version_infos: list[str]
    build_dependency_metadata_present: bool
    compat_build_dependency_metadata_present: bool
    package_installed: bool
    vendor_packages_present: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def inspect_ops_transformer_runtime(*, toolkit_home: str | None) -> OpsTransformerRuntimeStatus:
    ops_root = resolve_workspace_repo("ops-transformer")
    install_root = _infer_install_root(toolkit_home)

    build_out = ops_root / "build_out" if ops_root else None
    package_runfiles = _discover_runfiles(build_out) if build_out else []

    share_info_dir = install_root / "share" / "info" / "ops_transformer" if install_root else None
    vendors_dir = _discover_vendor_root(toolkit_home)
    uninstall_candidates = [
        share_info_dir / "script" / "uninstall.sh" if share_info_dir else None,
        vendors_dir / "scripts" / "uninstall.sh" if vendors_dir else None,
        vendors_dir / "uninstall.sh" if vendors_dir else None,
    ]
    uninstall_script = next(
        (path for path in uninstall_candidates if path is not None and path.exists()),
        None,
    )
    vendors_config_candidates = [
        vendors_dir / "config.ini" if vendors_dir else None,
        vendors_dir / "op_impl" / "ai_core" / "tbe" / "kernel" / "config" / "ascend910b" / "binary_info_config.json"
        if vendors_dir
        else None,
    ]
    vendors_config = next(
        (path for path in vendors_config_candidates if path is not None and path.exists()),
        None,
    )
    binary_info_configs = []
    if vendors_dir and vendors_dir.exists():
        binary_info_configs = sorted(str(path) for path in vendors_dir.rglob("binary_info_config.json"))
    required_infos = required_version_info_paths(toolkit_home)
    missing_infos = [str(path) for path in required_infos if not path.exists()]
    compat_root = prepare_compat_package_path(toolkit_home=toolkit_home)
    compat_required_infos = compat_required_version_info_paths(compat_root)
    compat_missing_infos = [str(path) for path in compat_required_infos if not path.exists()]
    effective_package_path = str(compat_root) if compat_root and missing_infos else toolkit_home

    return OpsTransformerRuntimeStatus(
        ops_transformer_root=str(ops_root) if ops_root else None,
        toolkit_home=toolkit_home,
        install_root=str(install_root) if install_root else None,
        build_out=str(build_out) if build_out and build_out.exists() else None,
        seed_ops=list(SEED_OPS),
        package_runfiles=[str(path) for path in package_runfiles],
        package_path=toolkit_home,
        effective_package_path=effective_package_path,
        compat_package_path=str(compat_root) if compat_root else None,
        share_info_dir=str(share_info_dir) if share_info_dir and share_info_dir.exists() else None,
        uninstall_script=str(uninstall_script) if uninstall_script and uninstall_script.exists() else None,
        vendors_dir=str(vendors_dir) if vendors_dir and vendors_dir.exists() else None,
        vendors_config=str(vendors_config) if vendors_config and vendors_config.exists() else None,
        binary_info_configs=binary_info_configs,
        required_version_infos=[str(path) for path in required_infos],
        missing_version_infos=missing_infos,
        compat_required_version_infos=[str(path) for path in compat_required_infos],
        compat_missing_version_infos=compat_missing_infos,
        build_dependency_metadata_present=not missing_infos,
        compat_build_dependency_metadata_present=not compat_missing_infos,
        package_installed=bool(uninstall_script and uninstall_script.exists()),
        vendor_packages_present=bool(binary_info_configs),
    )


__all__ = [
    "OpsTransformerRuntimeStatus",
    "PACKAGE_VERSION_INFO_FALLBACKS",
    "REQUIRED_BUILD_INFO_PACKAGES",
    "SEED_OPS",
    "compat_required_version_info_paths",
    "inspect_ops_transformer_runtime",
    "prepare_compat_package_path",
    "required_version_info_paths",
]
