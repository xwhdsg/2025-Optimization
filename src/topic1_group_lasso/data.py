from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupLassoData:
    A: np.ndarray  # (m,n)
    B: np.ndarray  # (m,l)
    U_true: np.ndarray  # (n,l)
    active_rows: np.ndarray  # (k,)
    mu: float


def generate_group_lasso(
    seed: int = 97006855,
    n: int = 512,
    m: int = 256,
    l: int = 2,
    active_frac: float = 0.1,
    mu: float = 1e-2,
) -> GroupLassoData:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    k = int(round(active_frac * n))
    p = rng.permutation(n)[:k]
    U = np.zeros((n, l), dtype=float)
    U[p, :] = rng.standard_normal((k, l))
    B = A @ U
    return GroupLassoData(A=A, B=B, U_true=U, active_rows=p, mu=mu)
