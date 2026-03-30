from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_worker_init_fn(base_seed: int, rank: int = 0):
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + rank * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)

    return _worker_init_fn
