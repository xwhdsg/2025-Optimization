from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCData:
    M_true: np.ndarray
    Omega: np.ndarray  # boolean mask
    n: int
    r: int
    lam: float


def generate_mc(seed: int = 1, n: int = 300, r: int = 5, p: float = 0.1, lam: float = 1e-3) -> MCData:
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    V = rng.standard_normal((n, r))
    M = U @ V.T
    Omega = rng.random((n, n)) < p
    return MCData(M_true=M, Omega=Omega, n=n, r=r, lam=lam)
