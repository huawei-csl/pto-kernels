"""Reusable local distributed launch helpers for PTO kernel benchmarks."""

from __future__ import annotations

import json
import os
import socket
import time
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _device_for_rank(rank: int, *, device_type: str) -> str:
    return f"{device_type}:{rank}"


def _spawn_worker(
    rank: int,
    worker_fn,
    world_size: int,
    worker_kwargs: dict[str, Any],
    output_dir: str,
    master_port: int,
    backend: str,
    device_type: str,
):
    rank_report_path = Path(output_dir) / f"rank_{rank}.json"
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        device = _device_for_rank(rank, device_type=device_type)
        if device_type == "npu":
            torch.npu.set_device(device)

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{master_port}",
            rank=rank,
            world_size=world_size,
        )

        report = worker_fn(
            rank=rank,
            world_size=world_size,
            output_dir=Path(output_dir),
            device=device,
            **worker_kwargs,
        )
        if not isinstance(report, dict):
            raise TypeError(f"Distributed worker returned {type(report).__name__}, expected dict.")
        report.setdefault("status", "ok")
        report.setdefault("rank", rank)
        report.setdefault("world_size", world_size)
        report.setdefault("device", device)
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "rank": rank,
            "world_size": world_size,
            "reason": str(exc),
        }
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    rank_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def run_local_ranked_job(
    worker_fn,
    *,
    world_size: int,
    output_dir: Path,
    worker_kwargs: dict[str, Any] | None = None,
    backend: str = "hccl",
    device_type: str = "npu",
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if device_type == "npu":
        npu_count = torch.npu.device_count()
        if npu_count < world_size:
            return {
                "status": "blocked",
                "reason": f"Need {world_size} local NPUs for distributed validation, but only {npu_count} detected.",
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    master_port = _pick_free_port()
    spawn_module = importlib.import_module("torch.multiprocessing.spawn")
    launch_timeout = timeout_seconds
    if launch_timeout is None:
        launch_timeout = float(os.environ.get("PTO_DISTRIBUTED_LAUNCH_TIMEOUT_SEC", "180"))
    context = spawn_module.start_processes(
        _spawn_worker,
        args=(
            worker_fn,
            world_size,
            worker_kwargs or {},
            str(output_dir),
            master_port,
            backend,
            device_type,
        ),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + max(1.0, launch_timeout)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            partial_rank_reports = []
            for rank in range(world_size):
                report_path = output_dir / f"rank_{rank}.json"
                if report_path.exists():
                    partial_rank_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            return {
                "status": "blocked",
                "reason": (
                    f"Distributed ranked job timed out after {launch_timeout:.0f}s before all ranks "
                    "produced reports."
                ),
                "world_size": world_size,
                "rank_reports": partial_rank_reports,
            }
        if context.join(timeout=min(5.0, remaining), grace_period=5.0):
            break

    rank_reports = []
    for rank in range(world_size):
        report_path = output_dir / f"rank_{rank}.json"
        if not report_path.exists():
            return {
                "status": "blocked",
                "reason": f"Distributed worker for rank {rank} did not produce a report.",
                "world_size": world_size,
            }
        rank_reports.append(json.loads(report_path.read_text(encoding="utf-8")))

    failures = [report for report in rank_reports if report.get("status") != "ok"]
    if failures:
        return {
            "status": "blocked",
            "reason": "Distributed ranked job failed.",
            "world_size": world_size,
            "rank_reports": rank_reports,
        }

    return {
        "status": "ok",
        "world_size": world_size,
        "rank_reports": rank_reports,
    }


__all__ = ["run_local_ranked_job"]
