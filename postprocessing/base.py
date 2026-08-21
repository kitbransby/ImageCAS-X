from abc import ABC, abstractmethod
import numpy as np


class PostprocessingStep(ABC):
    """A single transform from raw model output toward a final lumen mask."""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """`sample` holds 'pred' and 'scan_id'; steps update 'pred', or add 'mask'
        once binarised."""
        raise NotImplementedError
