from __future__ import annotations

import numpy as np


def loss_and_grads(U: np.ndarray, V: np.ndarray, M: np.ndarray, Omega: np.ndarray, lam: float):
    """Compute loss and gradients for 0.5||P_Omega(UV^T - M)||_F^2 + lam/2 (||U||^2+||V||^2)."""
    R = (U @ V.T - M)
    R = R * Omega
    loss = 0.5 * float(np.sum(R * R)) + 0.5 * lam * (float(np.sum(U * U)) + float(np.sum(V * V)))
    gU = R @ V + lam * U
    gV = R.T @ U + lam * V
    return loss, gU, gV


def recovery_error(U: np.ndarray, V: np.ndarray, M: np.ndarray) -> float:
    X = U @ V.T
    return float(np.linalg.norm(X - M, ord='fro') / (np.linalg.norm(M, ord='fro') + 1e-12))
