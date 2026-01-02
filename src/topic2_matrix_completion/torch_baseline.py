from __future__ import annotations

import numpy as np


def torch_lbfgs_baseline(M: np.ndarray, Omega: np.ndarray, r: int = 5, lam: float = 1e-3, max_iter: int = 200):
    """Baseline using torch.optim.LBFGS on the factorized objective."""
    import torch

    device = torch.device('cpu')
    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    Om_t = torch.tensor(Omega.astype(np.float32), dtype=torch.float32, device=device)
    n = M.shape[0]

    U = torch.randn(n, r, device=device, requires_grad=True)
    V = torch.randn(n, r, device=device, requires_grad=True)

    opt = torch.optim.LBFGS([U, V], max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        R = (U @ V.t() - M_t) * Om_t
        loss = 0.5 * (R * R).sum() + 0.5 * lam * ((U * U).sum() + (V * V).sum())
        loss.backward()
        return loss

    loss = opt.step(closure)
    return U.detach().cpu().numpy(), V.detach().cpu().numpy(), float(loss.detach().cpu().item())
