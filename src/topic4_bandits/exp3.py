from __future__ import annotations

import numpy as np


def exp3(T: int, K: int, pull_fn, rng: np.random.Generator, gamma: float = 0.07, callback=None):
    w = np.ones(K)
    rewards = []
    for t in range(1, T + 1):
        p = (1 - gamma) * (w / np.sum(w)) + gamma / K
        arm = int(rng.choice(K, p=p))
        r = pull_fn(arm, rng)
        rewards.append(r)
        # importance-weighted reward estimate
        x_hat = r / (p[arm] + 1e-12)
        w[arm] *= np.exp((gamma * x_hat) / K)
        if callback is not None:
            callback(t=t, arm=arm, reward=r)
    return np.array(rewards)
