from abc import ABC, abstractmethod


class PreprocessingStep(ABC):
    """A single, stateless transform applied to a sample dict."""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """Modifies `sample` in place or returns a new dict."""
        raise NotImplementedError
