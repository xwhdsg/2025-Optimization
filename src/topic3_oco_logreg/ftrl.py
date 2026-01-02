from __future__ import annotations

import numpy as np

from .ogd import proj_l2_ball, logistic_loss_grad


def ftrl(A: np.ndarray, y: np.ndarray, R: float, eta: float, lam: float, callback=None):
    """Simple FTRL with quadratic regularizer: minimize <g_1..t, w> + (lam/2)||w||^2.
    Closed form (before projection): w = - (1/lam) * sum g.
    """
    T, d = A.shape
    gsum = np.zeros(d)
    w = np.zeros(d)
    cum_loss = 0.0
    for t in range(T):
        loss, g = logistic_loss_grad(A[t], y[t], w)
        cum_loss += loss
        gsum += g
        w = -(eta / (lam + 1e-12)) * gsum
        w = proj_l2_ball(w, R)
        if callback is not None:
            callback(t=t+1, loss=loss, cum_loss=cum_loss, w=w)
    return w
