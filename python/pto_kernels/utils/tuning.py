"""Environment-driven tuning knobs for seed PTO kernels."""

from __future__ import annotations

import os


def tuned_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    valid_values: tuple[int, ...] | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if valid_values is not None and value not in valid_values:
        allowed = ", ".join(str(item) for item in valid_values)
        raise ValueError(f"{name} must be one of ({allowed}), got {value}")
    return value
