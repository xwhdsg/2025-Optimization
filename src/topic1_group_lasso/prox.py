from __future__ import annotations

import numpy as np


def group_l12_norm(X: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(X, axis=1)))


def prox_l12(X: np.ndarray, tau: float) -> np.ndarray:
    """Row-wise group soft-thresholding: prox_{tau*||.||_{1,2}}(X)."""
    # For each row x_i: (1 - tau/||x_i||)_+ * x_i
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.maximum(0.0, 1.0 - tau / (norms + 1e-12))
    return scale * X
