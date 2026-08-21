"""Augmentation pipeline factory, driven by a config's 'augmentation' block.
Applied to the training split only; the shared set lives in configs/pipeline.json.
"""

from augmentation.steps import (
    RandomFlip, MultiplicativeBrightness, Contrast, GaussianBlur, GaussianNoise,
    GammaTransform, SimulateLowResolution,
)

_STEP_REGISTRY = {
    "random_flip":              RandomFlip,
    "multiplicative_brightness": MultiplicativeBrightness,
    "contrast":                 Contrast,
    "gaussian_blur":            GaussianBlur,
    "gaussian_noise":           GaussianNoise,
    "gamma":                    GammaTransform,
    "gamma_invert":             GammaTransform,
    "simulate_low_resolution":  SimulateLowResolution,
}


class AugmentationPipeline:
    """Chains AugmentationStep instances."""

    def __init__(self, steps: list):
        self.steps = steps

    def __call__(self, sample: dict) -> dict:
        for step in self.steps:
            sample = step(sample)
        return sample

    def __len__(self):
        return len(self.steps)


def build_augmentation(aug_config: dict) -> AugmentationPipeline:
    steps = []
    for name in aug_config.get("steps", []):
        cls = _STEP_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown augmentation step: '{name}'")
        params = aug_config.get("params", {}).get(name, {})
        steps.append(cls(**params))
    return AugmentationPipeline(steps)
