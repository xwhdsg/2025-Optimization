from __future__ import annotations

import numpy as np


def offline_opt_cvxpy(A: np.ndarray, y: np.ndarray, R: float):
    """Compute comparator: min_{||w||<=R} sum log(1+exp(-y a^T w))."""
    import cvxpy as cp

    T, d = A.shape
    w = cp.Variable(d)
    losses = cp.sum(cp.logistic(cp.multiply(-y, A @ w)))
    prob = cp.Problem(cp.Minimize(losses), [cp.norm(w, 2) <= R])
    prob.solve()
    return w.value, prob.value
