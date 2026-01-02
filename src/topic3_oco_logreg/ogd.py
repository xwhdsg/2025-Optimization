from __future__ import annotations

import numpy as np


def proj_l2_ball(w: np.ndarray, R: float) -> np.ndarray:
    n = np.linalg.norm(w)
    if n <= R:
        return w
    return w * (R / (n + 1e-12))


def logistic_loss_grad(a: np.ndarray, y: float, w: np.ndarray):
    z = y * float(a @ w)
    # grad of log(1+exp(-z)) wrt w: -(y a) * sigmoid(-z)
    s = 1.0 / (1.0 + np.exp(z))
    grad = -(y * a) * s
    loss = np.log(1.0 + np.exp(-z))
    return float(loss), grad


def ogd(A: np.ndarray, y: np.ndarray, R: float, eta: float, callback=None):
    T, d = A.shape
    w = np.zeros(d)
    cum_loss = 0.0
    for t in range(T):
        loss, g = logistic_loss_grad(A[t], y[t], w)
        cum_loss += loss
        w = proj_l2_ball(w - eta * g, R)
        if callback is not None:
            callback(t=t+1, loss=loss, cum_loss=cum_loss, w=w)
    return w
