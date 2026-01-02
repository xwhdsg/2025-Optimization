from __future__ import annotations

import numpy as np


def solve_group_lasso_cvxpy(A: np.ndarray, B: np.ndarray, mu: float, solver: str | None = None):
    """Baseline/reference solve via CVXPY. Requires cvxpy installed."""
    import cvxpy as cp

    n = A.shape[1]
    l = B.shape[1]
    X = cp.Variable((n, l))
    obj = 0.5 * cp.sum_squares(A @ X - B) + mu * cp.sum(cp.norm(X, 2, axis=1))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=solver)  # solver can be None
    return X.value, prob.value
