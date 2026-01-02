from __future__ import annotations

import numpy as np


def ucb1(T: int, K: int, pull_fn, rng: np.random.Generator, callback=None):
    n = np.zeros(K)
    s = np.zeros(K)
    rewards = []

    # init: play each arm once
    for i in range(K):
        r = pull_fn(i, rng)
        n[i] += 1
        s[i] += r
        rewards.append(r)
        if callback is not None:
            callback(t=len(rewards), arm=i, reward=r)

    for t in range(K + 1, T + 1):
        avg = s / np.maximum(1.0, n)
        bonus = np.sqrt(2.0 * np.log(t) / np.maximum(1.0, n))
        arm = int(np.argmax(avg + bonus))
        r = pull_fn(arm, rng)
        n[arm] += 1
        s[arm] += r
        rewards.append(r)
        if callback is not None:
            callback(t=t, arm=arm, reward=r)

    return np.array(rewards)
