from __future__ import annotations

import numpy as np


def stochastic_regret(rewards: np.ndarray, best_mean: float) -> np.ndarray:
    T = rewards.shape[0]
    return np.arange(1, T + 1) * best_mean - np.cumsum(rewards)
