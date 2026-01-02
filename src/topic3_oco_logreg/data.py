from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OCOData:
    A: np.ndarray  # (T,d)
    y: np.ndarray  # (T,)
    R: float


def generate_oco(seed: int = 1, d: int = 50, T: int = 5000, R: float = 5.0, noise_std: float = 0.5) -> OCOData:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    w = w / (np.linalg.norm(w) + 1e-12) * (R / 2.0)
    A = rng.standard_normal((T, d))
    eps = rng.normal(0.0, noise_std, size=T)
    y = np.sign(A @ w + eps)
    y[y == 0] = 1
    return OCOData(A=A, y=y.astype(float), R=R)
