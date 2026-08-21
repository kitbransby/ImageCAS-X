from utils.config import BenchmarkConfig
from postprocessing.steps import (
    Threshold, KeepComponentsLargerThan100Voxels, ArgmaxBinarize, ConnectivityFuse,
)

_STEP_REGISTRY = {
    "threshold": Threshold,
    "keep_components_larger_than_100_voxels": KeepComponentsLargerThan100Voxels,
    # CAS-Net
    "argmax_binarize": ArgmaxBinarize,
    # ADE-HTL
    "connectivity_fuse": ConnectivityFuse,
}


class PostprocessingPipeline:
    def __init__(self, steps: list):
        self.steps = steps

    def __call__(self, sample: dict) -> dict:
        for step in self.steps:
            sample = step(sample)
        return sample


def build_postprocessing(config: BenchmarkConfig) -> PostprocessingPipeline:
    steps = []
    for step_name in config.postprocessing.steps:
        cls = _STEP_REGISTRY.get(step_name)
        if cls is None:
            raise ValueError(f"Unknown postprocessing step '{step_name}'")
        params = config.postprocessing.params.get(step_name, {})
        steps.append(cls(**params))
    return PostprocessingPipeline(steps)
