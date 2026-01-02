from __future__ import annotations

import numpy as np

from .objective import loss_and_grads


def gd(U: np.ndarray, V: np.ndarray, M: np.ndarray, Omega: np.ndarray, lam: float, lr: float, max_iter: int, callback=None):
    for it in range(1, max_iter + 1):
        loss, gU, gV = loss_and_grads(U, V, M, Omega, lam)
        U -= lr * gU
        V -= lr * gV
        if callback is not None:
            callback(it=it, loss=loss, U=U, V=V)
    return U, V
