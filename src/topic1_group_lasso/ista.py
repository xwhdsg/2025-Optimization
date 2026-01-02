from __future__ import annotations

import numpy as np

from .prox import prox_l12, group_l12_norm


def objective(A: np.ndarray, B: np.ndarray, X: np.ndarray, mu: float) -> float:
    R = A @ X - B
    return 0.5 * float(np.sum(R * R)) + mu * group_l12_norm(X)


def ista(
    A: np.ndarray,
    B: np.ndarray,
    mu: float,
    step: float,
    max_iter: int = 2000,
    tol: float = 1e-6,
    X0: np.ndarray | None = None,
    callback=None,
) -> np.ndarray:
    m, n = A.shape
    l = B.shape[1]
    X = np.zeros((n, l)) if X0 is None else X0.copy()

    At = A.T
    for it in range(1, max_iter + 1):
        G = At @ (A @ X - B)
        Xn = prox_l12(X - step * G, tau=step * mu)

        if callback is not None:
            callback(it=it, X=Xn, obj=objective(A, B, Xn, mu))

        if np.linalg.norm(Xn - X) / (np.linalg.norm(X) + 1e-12) < tol:
            X = Xn
            break
        X = Xn
    return X
