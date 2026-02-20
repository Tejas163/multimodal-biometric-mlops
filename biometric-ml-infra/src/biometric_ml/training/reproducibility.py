"""
reproducibility.py
------------------
Utilities to make training runs fully reproducible.

Calling ``seed_everything(seed)`` before any PyTorch / NumPy / Python
operations ensures that:
  * Random number generators (Python, NumPy, PyTorch CPU + CUDA) are seeded.
  * CuDNN uses deterministic algorithms (at a small speed cost).
  * DataLoader worker seeds are set via a worker_init_fn.

The Hydra config snapshot is saved as an MLflow artifact at run start so
the exact configuration that produced a checkpoint is always recoverable.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed all RNGs and optionally enable deterministic CuDNN algorithms.

    Args:
        seed:          Integer seed value (from config.training.seed).
        deterministic: If True, forces deterministic CUDA ops.  May reduce
                       throughput on some GPU architectures; disable for
                       large-scale pre-training runs if speed matters more
                       than exact reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Available in PyTorch ≥ 1.8; raises on non-deterministic ops
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # Older PyTorch — best-effort


def get_worker_init_fn(base_seed: int):
    """
    Returns a DataLoader ``worker_init_fn`` that derives a unique seed
    for each worker from the base training seed.

    Usage::

        DataLoader(..., worker_init_fn=get_worker_init_fn(cfg.training.seed))
    """

    def _init(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init
