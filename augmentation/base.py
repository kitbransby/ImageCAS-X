from abc import ABC, abstractmethod
import numpy as np


class AugmentationStep(ABC):
    """A single randomised transform over a sample dict of (X, Y, Z) arrays plus
    `spacing`. Spatial transforms must touch every spatial field consistently (see
    steps.py::_apply_spatial); intensity transforms touch 'volume' only."""

    def __init__(self, p: float = 0.5, **kwargs):
        self.p = p  # probability of applying this transform

    def __call__(self, sample: dict) -> dict:
        if np.random.random() < self.p:
            return self.apply(sample)
        return sample

    @abstractmethod
    def apply(self, sample: dict) -> dict:
        raise NotImplementedError
