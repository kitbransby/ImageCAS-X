import numpy as np
import torch
from postprocessing.base import PostprocessingStep


class Threshold(PostprocessingStep):
    """Sigmoid + threshold to a binary mask. Params: threshold (default 0.5)."""
    def __call__(self, sample: dict) -> dict:
        pred = sample["pred"]
        threshold = self.params.get("threshold", 0.5)
        if isinstance(pred, torch.Tensor):
            mask = (pred.sigmoid() > threshold).cpu().numpy().astype(np.uint8)
        else:
            # numpy inputs are already in probability space.
            mask = (pred > threshold).astype(np.uint8)
        # Drop a singleton channel axis to align with the (X, Y, Z) GT mask.
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]
        sample["mask"] = mask
        return sample


class KeepComponentsLargerThan100Voxels(PostprocessingStep):
    """Keep foreground components of at least `min_size` voxels. Size-based rather
    than keep-top-N, so separate L/R coronary trees survive without hardcoding how
    many components to expect."""

    def __call__(self, sample: dict) -> dict:
        from scipy.ndimage import label

        mask = sample["mask"].astype(np.uint8)
        min_size = self.params.get("min_size", 100)

        labeled, n_labels = label(mask)
        if n_labels == 0:
            return sample

        out = np.zeros_like(mask)
        for i in range(1, n_labels + 1):
            if (labeled == i).sum() >= min_size:
                out[labeled == i] = 1
        sample["mask"] = out
        return sample


class ArgmaxBinarize(PostprocessingStep):
    """2-channel sigmoid output to a binary mask via argmax (CAS-Net). Unlike a
    threshold, this is correct even when the channels do not sum to 1."""

    def __call__(self, sample: dict) -> dict:
        pred = sample["pred"]
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        sample["mask"] = np.argmax(pred, axis=0).astype(np.uint8)
        return sample


class ConnectivityFuse(PostprocessingStep):
    """Binarise ADE-HTL's fused connectivity mask on the original scan grid.

    Neither half of the fusion rule runs here — the vote and its cut both happen in
    inference.py::_fuse_connectivity, on the model's own grid. What arrives is that
    0/1 indicator, linearly resampled, so the cut below is the ordinary 0.5 every
    other method binarises at, just at full original resolution.

    `threshold` and `min_connections` are consumed upstream but live in this step's
    params so the whole rule reads as one block. min_connections is >= rather than >:
    that is the reading making the paper's own worked cases come out right, given the
    centre point counts.

    Must run FIRST in the chain: it takes the prediction from 4D to 3D, and every
    later step assumes 3D.
    """

    def __call__(self, sample: dict) -> dict:
        pred = sample["pred"]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        pred = np.asarray(pred)
        if pred.ndim != 4:
            raise ValueError(
                f"ConnectivityFuse expects a (C, X, Y, Z) map with the fused connectivity "
                f"indicator in channel 0 (see inference.py::_fuse_connectivity), got "
                f"shape {pred.shape}."
            )

        # Channel 1, when present, is the lumen probability for --save-probs only.
        sample["mask"] = (pred[0] >= 0.5).astype(np.uint8)
        return sample
