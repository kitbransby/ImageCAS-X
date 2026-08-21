"""Augmentation steps. Spatial transforms apply to every spatial array in the
sample, intensity transforms to 'volume' only. AugmentationStep gates each on its
probability `p`, so `apply` assumes it has already been selected to run. Ranges and
probabilities follow nnU-Net's defaults.
"""

import numpy as np
from augmentation.base import AugmentationStep


# Discrete arrays go through the label function (nearest neighbour) so their values
# stay exactly the set they started as; continuous ones through the volume function.
# Neither is touched by intensity steps, which only look at 'volume'.
_LABEL_KEYS = ("mask", "centerline", "kp_map")
_CONTINUOUS_KEYS = ("cl_heatmap",)
_CHANNEL_KEYS = ("ade",)  # continuous, but with a leading channel axis


def _apply_spatial(sample: dict, fn_volume, fn_label) -> dict:
    """Keep every spatial array registered with the image. 'ade' is transformed
    channel by channel, since the transforms assume a bare 3D array and would
    otherwise treat its channel axis as spatial."""
    if "volume" in sample:
        sample["volume"] = fn_volume(sample["volume"])
    for key in _LABEL_KEYS:
        if sample.get(key) is not None:
            sample[key] = fn_label(sample[key])
    for key in _CONTINUOUS_KEYS:
        if sample.get(key) is not None:
            sample[key] = fn_volume(sample[key])
    for key in _CHANNEL_KEYS:
        if sample.get(key) is not None:
            sample[key] = np.stack([fn_volume(c) for c in sample[key]], axis=0)
    return sample


class RandomFlip(AugmentationStep):
    """Random flip along one or more spatial axes, each drawn independently."""
    def __init__(self, p: float = 1.0, axes=(0, 1, 2), p_per_axis: float = 0.5, **kw):
        super().__init__(p=p)
        self.axes = list(axes)
        self.p_per_axis = p_per_axis

    def apply(self, sample: dict) -> dict:
        flip_axes = [ax for ax in self.axes if np.random.random() < self.p_per_axis]
        if not flip_axes:
            return sample
        fn = lambda a: np.ascontiguousarray(np.flip(a, axis=flip_axes))
        return _apply_spatial(sample, fn, fn)


class MultiplicativeBrightness(AugmentationStep):
    """Scale the volume by a random factor, no additive offset. Independent of
    Contrast; nnU-Net gates the two separately rather than bundling them."""
    def __init__(self, p: float = 0.15, multiplier_range=(0.75, 1.25),
                 clip=(0.0, 1.0), **kw):
        super().__init__(p=p)
        self.multiplier_range = tuple(multiplier_range)
        self.clip = tuple(clip) if clip is not None else None

    def apply(self, sample: dict) -> dict:
        if "volume" not in sample:
            return sample
        factor = float(np.random.uniform(*self.multiplier_range))
        vol = sample["volume"].astype(np.float32) * factor
        if self.clip is not None:
            vol = np.clip(vol, self.clip[0], self.clip[1])
        sample["volume"] = vol
        return sample


class Contrast(AugmentationStep):
    """Scale values around the volume's own mean, then clip back to its original
    min/max so the perturbation cannot leave the range the network was normalised
    to."""
    def __init__(self, p: float = 0.15, contrast_range=(0.75, 1.25),
                 preserve_range: bool = True, **kw):
        super().__init__(p=p)
        self.contrast_range = tuple(contrast_range)
        self.preserve_range = preserve_range

    def apply(self, sample: dict) -> dict:
        if "volume" not in sample:
            return sample
        vol = sample["volume"].astype(np.float32)
        mn, mx = vol.min(), vol.max()
        mean = vol.mean()
        factor = float(np.random.uniform(*self.contrast_range))
        vol = (vol - mean) * factor + mean
        if self.preserve_range:
            vol = np.clip(vol, mn, mx)
        sample["volume"] = vol
        return sample


class GaussianBlur(AugmentationStep):
    """Random Gaussian blur."""
    def __init__(self, p: float = 0.5, sigma_range=(0.1, 1.0), **kw):
        super().__init__(p=p)
        self.sigma_range = tuple(sigma_range)

    def apply(self, sample: dict) -> dict:
        if "volume" not in sample:
            return sample
        from scipy.ndimage import gaussian_filter
        sigma = float(np.random.uniform(*self.sigma_range))
        sample["volume"] = gaussian_filter(sample["volume"].astype(np.float32), sigma=sigma)
        return sample


class GaussianNoise(AugmentationStep):
    """Additive Gaussian noise. `variance_range` is a VARIANCE range, matching
    nnU-Net — sampling std uniformly instead would skew the distribution."""
    def __init__(self, p: float = 0.1, variance_range=(0.0, 0.1), clip=(0.0, 1.0), **kw):
        super().__init__(p=p)
        self.variance_range = tuple(variance_range)
        self.clip = tuple(clip) if clip is not None else None

    def apply(self, sample: dict) -> dict:
        if "volume" not in sample:
            return sample
        variance = float(np.random.uniform(*self.variance_range))
        std = variance ** 0.5
        vol = sample["volume"].astype(np.float32)
        vol = vol + np.random.normal(0.0, std, vol.shape).astype(np.float32)
        if self.clip is not None:
            vol = np.clip(vol, self.clip[0], self.clip[1])
        sample["volume"] = vol
        return sample


class GammaTransform(AugmentationStep):
    """Gamma correction: rescale to [0, 1] by the volume's own min/max, raise to a
    random power, rescale back.

    When gamma_range starts below 1, gamma is drawn from either side of 1 with equal
    probability, matching nnU-Net — this biases the draw away from the no-op at
    gamma == 1. `invert` negates the volume around the warp so the same transform
    darkens rather than brightens; nnU-Net applies gamma both ways, as two steps.
    """
    def __init__(self, p: float = 0.3, gamma_range=(0.7, 1.5), invert: bool = False,
                 retain_stats: bool = True, epsilon: float = 1e-7,
                 clip=(0.0, 1.0), **kw):
        super().__init__(p=p)
        self.gamma_range = tuple(gamma_range)
        self.invert = invert
        self.retain_stats = retain_stats
        self.epsilon = epsilon
        self.clip = tuple(clip) if clip is not None else None

    def apply(self, sample: dict) -> dict:
        if "volume" not in sample:
            return sample
        vol = sample["volume"].astype(np.float32)
        if self.invert:
            vol = -vol
        mean, std = vol.mean(), vol.std()

        lo, hi = self.gamma_range
        if lo < 1 and np.random.random() < 0.5:
            gamma = float(np.random.uniform(lo, 1))
        else:
            gamma = float(np.random.uniform(max(lo, 1), hi))

        mn, rng = vol.min(), vol.max() - vol.min()
        vol = np.power((vol - mn) / (rng + self.epsilon), gamma) * rng + mn

        if self.retain_stats:
            vol = (vol - vol.mean()) / (vol.std() + 1e-8) * std + mean
        if self.invert:
            vol = -vol
        if self.clip is not None:
            vol = np.clip(vol, self.clip[0], self.clip[1])
        sample["volume"] = vol
        return sample


class SimulateLowResolution(AugmentationStep):
    """Simulate low-resolution artefacts by downsampling then upsampling."""
    def __init__(self, p: float = 0.25, zoom_range=(0.5, 1.0), **kw):
        super().__init__(p=p)
        self.zoom_range = tuple(zoom_range)

    def apply(self, sample: dict) -> dict:
        if "volume" not in sample:
            return sample
        from scipy.ndimage import zoom
        vol = sample["volume"].astype(np.float32)
        f = float(np.random.uniform(*self.zoom_range))
        if f >= 1.0:
            return sample
        down = zoom(vol, f, order=1)
        up = zoom(down, [s / d for s, d in zip(vol.shape, down.shape)], order=1)
        sample["volume"] = up
        return sample
