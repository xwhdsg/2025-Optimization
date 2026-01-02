from __future__ import annotations

import numpy as np

from .prox import prox_l12, group_l12_norm


def objective(A: np.ndarray, B: np.ndarray, X: np.ndarray, mu: float) -> float:
    R = A @ X - B
    return 0.5 * float(np.sum(R * R)) + mu * float(np.sum(np.linalg.norm(X, axis=1)))


def fista(
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
    Y = X.copy()
    t = 1.0

    At = A.T
    for it in range(1, max_iter + 1):
        G = At @ (A @ Y - B)
        Xn = prox_l12(Y - step * G, tau=step * mu)
        tn = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Y = Xn + ((t - 1.0) / tn) * (Xn - X)

        if callback is not None:
            callback(it=it, X=Xn, obj=objective(A, B, Xn, mu))

        if np.linalg.norm(Xn - X) / (np.linalg.norm(X) + 1e-12) < tol:
            X = Xn
            break
        X = Xn
        t = tn
    return X
