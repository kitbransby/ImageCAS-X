import numpy as np


def dilate_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Dilation with a spherical structuring element. Shared by DilateMask and the
    ImageCAS pipeline's test-time dilation, so both use the same element."""
    from skimage.morphology import ball
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask.astype(bool), structure=ball(radius)).astype(np.uint8)
