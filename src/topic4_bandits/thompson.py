from __future__ import annotations

import numpy as np


def thompson_bernoulli(T: int, K: int, pull_fn, rng: np.random.Generator, callback=None):
    a = np.ones(K)
    b = np.ones(K)
    rewards = []
    for t in range(1, T + 1):
        theta = rng.beta(a, b)
        arm = int(np.argmax(theta))
        r = pull_fn(arm, rng)
        rewards.append(r)
        a[arm] += r
        b[arm] += 1.0 - r
        if callback is not None:
            callback(t=t, arm=arm, reward=r)
    return np.array(rewards)
