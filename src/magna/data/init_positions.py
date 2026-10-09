from typing import Literal

import numpy as np

InitDistribution = Literal["perturbed", "uniform", "cluster"]


def build_init_positions(
    targets: np.ndarray,
    distribution: InitDistribution,
    *,
    seed: int | None = 0,
    std: float | None = None,
    margin: float = 0.5,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] not in (2, 3):
        raise ValueError(f"targets must have shape (n, 2 or 3), got {targets.shape}")
    rng = np.random.default_rng(seed)
    span = np.maximum(targets.max(0) - targets.min(0), 1e-6)

    if distribution == "perturbed":
        s = std or float(np.mean(span) * 0.5)
        return (targets + rng.normal(0, s, targets.shape)).astype(np.float32)
    if distribution == "uniform":
        lo, hi = targets.min(0) - margin * span, targets.max(0) + margin * span
        return rng.uniform(lo, hi, targets.shape).astype(np.float32)
    if distribution == "cluster":
        s = std or float(np.linalg.norm(span) * 0.05)
        return (targets.mean(0) + rng.normal(0, s, targets.shape)).astype(np.float32)
    raise ValueError(f"Unknown init distribution: {distribution!r}")
