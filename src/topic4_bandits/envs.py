from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BernoulliBandit:
    means: np.ndarray  # (K,)

    def pull(self, i: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self.means[i])


def make_env_a(seed: int = 1, K: int = 10):
    rng = np.random.default_rng(seed)
    means = np.linspace(0.45, 0.58, K)
    means[-1] = 0.60
    rng.shuffle(means)
    return BernoulliBandit(means=means)


def make_env_b_nonstationary(seed: int = 2, K: int = 10, T: int = 20000, period: int = 4000):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.45, 0.58, K)
    base[-1] = 0.60
    rng.shuffle(base)

    # piecewise constant means, best arm changes by permutation
    means_t = np.zeros((T, K))
    cur = base.copy()
    for t in range(T):
        if t % period == 0 and t > 0:
            cur = rng.permutation(cur)
        means_t[t] = cur
    return means_t
