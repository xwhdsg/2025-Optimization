from __future__ import annotations

import numpy as np

from src.common.logger import RunLogger
from src.common.plotting import plot_from_json
from src.topic4_bandits.envs import make_env_a
from src.topic4_bandits.ucb1 import ucb1
from src.topic4_bandits.thompson import thompson_bernoulli
from src.topic4_bandits.exp3 import exp3
from src.topic4_bandits.metrics import stochastic_regret


def main():
    T = 5000
    K = 10
    rng = np.random.default_rng(1)
    env = make_env_a(seed=1, K=K)
    best_mean = float(np.max(env.means))

    def pull(i, rng):
        return env.pull(i, rng)

    # UCB1
    log = RunLogger('ucb1')
    rewards = ucb1(T, K, pull, rng, callback=lambda **kw: log.log(**kw))
    reg = stochastic_regret(rewards, best_mean)
    for t in range(T):
        log.rows[t]['regret'] = float(reg[t])
    log.save_json('results/topic4_ucb1.json')

    # Thompson
    rng = np.random.default_rng(2)
    log2 = RunLogger('ts')
    rewards2 = thompson_bernoulli(T, K, pull, rng, callback=lambda **kw: log2.log(**kw))
    reg2 = stochastic_regret(rewards2, best_mean)
    for t in range(T):
        log2.rows[t]['regret'] = float(reg2[t])
    log2.save_json('results/topic4_ts.json')

    # EXP3
    rng = np.random.default_rng(3)
    log3 = RunLogger('exp3')
    rewards3 = exp3(T, K, pull, rng, gamma=0.07, callback=lambda **kw: log3.log(**kw))
    reg3 = stochastic_regret(rewards3, best_mean)
    for t in range(T):
        log3.rows[t]['regret'] = float(reg3[t])
    log3.save_json('results/topic4_exp3.json')

    plot_from_json('results/topic4_ucb1.json', 't', 'regret', 'Topic4: UCB1 regret', 'figures/topic4_ucb1_regret.png')
    plot_from_json('results/topic4_ts.json', 't', 'regret', 'Topic4: Thompson regret', 'figures/topic4_ts_regret.png')
    plot_from_json('results/topic4_exp3.json', 't', 'regret', 'Topic4: EXP3 regret', 'figures/topic4_exp3_regret.png')

    print('Done. Logs in results/, plots in figures/.')


if __name__ == '__main__':
    main()
