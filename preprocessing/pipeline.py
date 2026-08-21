from utils.config import BenchmarkConfig
from preprocessing.steps import (
    Resample, ResampleToShape, DilateMask, NormaliseToRange,
)

_STEP_REGISTRY = {
    "resample": Resample,
    "resample_to_shape": ResampleToShape,
    "normalise_to_range": NormaliseToRange,
    # ImageCAS Stage 2 (dilated vessel seg)
    "dilate_mask": DilateMask,
}


class PreprocessingPipeline:
    def __init__(self, steps: list):
        self.steps = steps

    def __call__(self, sample: dict) -> dict:
        for step in self.steps:
            sample = step(sample)
        return sample


def build_preprocessing(config: BenchmarkConfig) -> PreprocessingPipeline:
    steps = []
    for step_name in config.preprocessing.steps:
        cls = _STEP_REGISTRY.get(step_name)
        if cls is None:
            raise ValueError(f"Unknown preprocessing step '{step_name}'")
        params = config.preprocessing.params.get(step_name, {})
        steps.append(cls(**params))
    return PreprocessingPipeline(steps)
