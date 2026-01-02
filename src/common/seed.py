from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # torch optional
    torch = None


@dataclass(frozen=True)
class SeedConfig:
    seed: int
    deterministic_torch: bool = True


def set_global_seed(cfg: SeedConfig) -> None:
    """Best-effort reproducibility across numpy/random/(optional) torch."""
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if torch is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        if cfg.deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
