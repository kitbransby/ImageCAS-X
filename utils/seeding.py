"""Central seeding. The seed is fixed rather than exposed as a config option, so any
two runs of the same config are directly comparable.
"""
import os
import random

import numpy as np
import SimpleITK as sitk
import torch

SEED = 42

def seed_everything(seed: int = SEED) -> None:
    """Seed every RNG and force deterministic cuDNN kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Re-seed python/numpy per worker and cap ITK to one thread there.

    PyTorch reseeds its own per-worker RNG, but forked workers otherwise inherit
    identical `random`/`numpy` state, correlating augmentation draws across workers.
    ITK separately sizes its thread pool to the full core count PER PROCESS;
    parallelism should come from num_workers, not from each worker's own ITK calls.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
