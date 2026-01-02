from __future__ import annotations

import numpy as np

from src.common.logger import RunLogger
from src.common.plotting import plot_from_json
from src.common.seed import SeedConfig, set_global_seed
from src.topic1_group_lasso.data import generate_group_lasso
from src.topic1_group_lasso.ista import ista, objective
from src.topic1_group_lasso.fista import fista


def main():
    set_global_seed(SeedConfig(seed=97006855))
    data = generate_group_lasso()
    A, B, mu = data.A, data.B, data.mu

    # conservative stepsize using spectral norm estimate
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / (L + 1e-12)

    # ISTA
    log1 = RunLogger('ista')
    X_ista = ista(A, B, mu, step=step, max_iter=300, callback=lambda **kw: log1.log(**kw))
    log1.save_json('results/topic1_ista.json')

    # FISTA
    log2 = RunLogger('fista')
    X_fista = fista(A, B, mu, step=step, max_iter=300, callback=lambda **kw: log2.log(**kw))
    log2.save_json('results/topic1_fista.json')

    # Plots
    plot_from_json('results/topic1_ista.json', 'it', 'obj', 'Topic1: ISTA objective', 'figures/topic1_ista_obj.png')
    plot_from_json('results/topic1_fista.json', 'it', 'obj', 'Topic1: FISTA objective', 'figures/topic1_fista_obj.png')

    print('Done. Logs in results/, plots in figures/.')
    print('ISTA final obj:', objective(A, B, X_ista, mu))
    print('FISTA final obj:', objective(A, B, X_fista, mu))


if __name__ == '__main__':
    main()
