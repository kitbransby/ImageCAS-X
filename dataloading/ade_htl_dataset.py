"""ADE-HTL's dataset: ROI-constrained crops carrying ADE's distance fields as extra
input channels, plus the topology targets its three decoders are supervised by.

Registered as input_type "ade_htl_crop". Subclasses RandomCropDataset, differing only
in where crops may land, what rides along in the input, and what comes out as
targets. Needs both offline assets: precompute_ade.py's and precompute_targets.py's
per-scan .npz files.

Train/val emit fixed-shape volume (6ch), mask, connectivity (27ch), cl_heatmap,
kp_edt, kp_coords and n_kp. Test returns the whole ROI instead of a crop, with
crop_origin/pre_crop_shape so inference.py can re-embed the prediction. Cropping to
the ROI is what Sec. III-A.1 describes and is also what keeps the 6-channel input
affordable — the ADE channels at full grid are ~1.5 GB per scan.

Connectivity and kp_edt are derived here rather than precomputed, both AFTER
augmentation: flipping a volume permutes which neighbour each connectivity channel
refers to, so deriving from the already-flipped mask stops the channel semantics
desyncing from the image.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from dataloading.random_crop_dataset import RandomCropDataset
from models.ade_htl import NEIGHBOUR_OFFSETS, N_CONNECTIVITY
from utils.config import BenchmarkConfig


# Eq. 2 clips at 1, so 1.0 is "at least tau away". Voxels outside the coarse mask
# (the paper's point set P) have no stored distance and are filled with it.
ADE_FILL = 1.0


def _connectivity_target(mask: np.ndarray) -> np.ndarray:
    """Sec. III-B.3's encoding: channel c is 1 at P when P and its neighbour at
    NEIGHBOUR_OFFSETS[c] are both vessel. Neighbours outside the crop count as
    background, matching what the network sees in its input."""
    out = np.zeros((N_CONNECTIVITY, *mask.shape), dtype=np.uint8)
    for c, offset in enumerate(NEIGHBOUR_OFFSETS):
        src, dst = [], []
        for axis, d in enumerate(offset):
            n = mask.shape[axis]
            # Source window is the neighbour's position, destination is P's.
            src.append(slice(max(d, 0), n + min(d, 0)))
            dst.append(slice(max(-d, 0), n + min(-d, 0)))
        out[c][tuple(dst)] = mask[tuple(src)]
    out &= mask[None]
    return out


class ADEHTLDataset(RandomCropDataset):
    """ROI-constrained crops carrying ADE input channels and HTL targets."""

    # Only this class can assemble the 6-channel input, so build_eval_dataloader uses
    # it rather than VolumeDataset at evaluation time.
    supplies_eval_volume = True

    def __init__(self, config: BenchmarkConfig, split: str = "train",
                 preprocessing=None, augmentation=None):
        super().__init__(config, split=split, preprocessing=preprocessing,
                         augmentation=augmentation)
        p = config.data.params
        if "patch_size" not in p:
            self.patch_size = [256, 256, 64]
        self.ade_dir = self._resolve(p.get("ade_dir", "ade_2026"))
        self.targets_dir = self._resolve(p.get("htl_targets_dir", "ade_htl_targets_2026"))
        self.max_keypoints = int(p.get("max_keypoints", 64))
        # Set per __getitem__ and read by the _random_origin override below.
        self._roi = None

    # -- ADE / target assets ------------------------------------------------

    def _load_assets(self, scan_id: str) -> tuple:
        """Loaded per access rather than cached: every worker draws a fresh random
        scan, so a cache would grow towards the whole dataset in each one."""
        import os

        ade = np.load(os.path.join(self.ade_dir, f"{scan_id}.npz"))
        targets = np.load(os.path.join(self.targets_dir, f"{scan_id}.npz"))
        return ade, targets

    @staticmethod
    def _scatter(idx: np.ndarray, values: np.ndarray, origin: tuple,
                 patch_size: list, n_channels: int, fill: float) -> np.ndarray:
        """Scatter sparse (voxel index, value) pairs into a dense crop, dropping rows
        outside it. How the ADE fields and centerline heatmap get from their sparse
        on-disk form to a dense array without materialising a full-grid volume."""
        out = np.full((n_channels, *patch_size), fill, dtype=np.float32)
        if len(idx) == 0:
            return out
        local = idx.astype(np.int64) - np.asarray(origin, dtype=np.int64)[None, :]
        inside = np.all((local >= 0) & (local < np.asarray(patch_size)[None, :]), axis=1)
        if not inside.any():
            return out
        local = local[inside]
        vals = np.atleast_2d(values[inside].astype(np.float32))
        if vals.shape[0] != n_channels:
            vals = vals.T
        out[:, local[:, 0], local[:, 1], local[:, 2]] = vals
        return out

    # -- crop placement -----------------------------------------------------

    def _random_origin(self, shape) -> tuple:
        """Uniform crop origin restricted to ADE's ROI. Overrides the whole-volume
        version so the inherited retry loop picks up the restriction for free."""
        import random

        if self._roi is None:
            return super()._random_origin(shape)
        lo, hi = self._roi
        origin = []
        for axis in range(3):
            span = self.patch_size[axis]
            # Keep the crop inside the ROI where possible; when the ROI is narrower
            # than the patch, centre on it and let _crop_at zero-pad the overhang.
            top = min(int(hi[axis]) - span, int(shape[axis]) - span)
            bottom = int(lo[axis])
            if top < bottom:
                centre = (int(lo[axis]) + int(hi[axis])) // 2 - span // 2
                origin.append(int(np.clip(centre, 0, max(0, shape[axis] - span))))
            else:
                origin.append(random.randint(bottom, top))
        return tuple(origin)

    # -- sample assembly ----------------------------------------------------

    def __getitem__(self, idx) -> dict:
        import random

        endless = self.split in ("train", "val")
        scan_id = random.choice(self.scan_ids) if endless else self.scan_ids[idx]

        volume, sitk_img = self._load_volume(scan_id)
        mask, sitk_mask = self._load_gt_mask(scan_id)
        mask = self._binarise_mask(np.asarray(mask))

        sample = {
            "volume": np.asarray(volume).astype(np.float32),
            "mask": mask.astype(np.uint8),
            "scan_id": scan_id,
            "spacing": self._spacing(sitk_img),
            "sitk_img": sitk_img,
            "sitk_mask": self._binarised_sitk_mask(mask, sitk_mask),
        }
        sample = self._run_preprocessing(sample)
        vol, msk = sample["volume"], sample["mask"]

        ade, targets = self._load_assets(scan_id)
        roi_lo, roi_hi = ade["roi_lo"], ade["roi_hi"]

        if endless:
            # Cleared afterwards so it cannot leak into another call on this worker.
            self._roi = (roi_lo, roi_hi)
            try:
                vol_patch, msk_patch, origin = self._sample_crop_with_fg_quota(
                    vol, msk, require_fg=bool(idx))
            finally:
                self._roi = None
            patch_size = list(self.patch_size)
        else:
            # The whole ROI in one piece; inference.py tiles it internally.
            origin = tuple(int(v) for v in np.maximum(roi_lo, 0))
            hi = np.minimum(roi_hi, np.array(vol.shape))
            patch_size = [int(h - o) for h, o in zip(hi, origin)]
            sl = tuple(slice(o, o + s) for o, s in zip(origin, patch_size))
            vol_patch, msk_patch = vol[sl], msk[sl]

        ade_patch = self._scatter(ade["idx"], ade["dist"], origin, patch_size,
                                  n_channels=5, fill=ADE_FILL)

        if not endless:
            # Inference never reads the topology targets, and at ROI scale the
            # connectivity target alone is ~300 MB per scan.
            stacked = np.concatenate([np.ascontiguousarray(vol_patch)[None], ade_patch], axis=0)
            return {
                "volume": torch.from_numpy(np.ascontiguousarray(stacked)).float(),
                "mask": torch.from_numpy(np.ascontiguousarray(msk_patch).astype(np.uint8)).long(),
                "scan_id": scan_id,
                "spacing": sample["spacing"],
                # Consumed by inference.py::_embed_crop.
                "crop_origin": tuple(int(o) for o in origin),
                "pre_crop_shape": tuple(int(s) for s in vol.shape),
                "centerlines": self._load_centerlines(scan_id),
            }

        cl_patch = self._scatter(targets["cl_idx"], targets["cl_val"], origin,
                                 patch_size, n_channels=1, fill=0.0)[0]
        kp_patch = self._scatter(targets["kp_idx"],
                                 np.ones(len(targets["kp_idx"]), dtype=np.float32),
                                 origin, patch_size, n_channels=1, fill=0.0)[0]

        if self.split == "train" and self.augmentation is not None:
            aug = self.augmentation({
                "volume": np.ascontiguousarray(vol_patch),
                "mask": np.ascontiguousarray(msk_patch),
                "cl_heatmap": cl_patch,
                "kp_map": kp_patch,
                "ade": ade_patch,
                "spacing": sample["spacing"],
            })
            vol_patch, msk_patch = aug["volume"], aug["mask"]
            cl_patch, kp_patch, ade_patch = aug["cl_heatmap"], aug["kp_map"], aug["ade"]

        msk_patch = np.ascontiguousarray(msk_patch).astype(np.uint8)

        # Derived from the (possibly flipped) crop, never loaded.
        connectivity = _connectivity_target(msk_patch)
        kp_bool = kp_patch > 0.5
        kp_coords, n_kp = self._keypoint_coords(kp_bool)
        kp_edt = self._keypoint_edt(kp_bool, patch_size)

        stacked = np.concatenate([np.ascontiguousarray(vol_patch)[None], ade_patch], axis=0)

        return {
            "volume": torch.from_numpy(np.ascontiguousarray(stacked)).float(),
            "mask": torch.from_numpy(msk_patch).long(),
            "connectivity": torch.from_numpy(connectivity),
            "cl_heatmap": torch.from_numpy(np.ascontiguousarray(cl_patch)).float(),
            "kp_edt": torch.from_numpy(kp_edt).float(),
            "kp_coords": torch.from_numpy(kp_coords).float(),
            "n_kp": torch.tensor(n_kp, dtype=torch.long),
            "scan_id": scan_id,
            "spacing": sample["spacing"],
        }

    def _keypoint_coords(self, kp_bool: np.ndarray) -> tuple:
        """Zero-padded to a fixed shape so default_collate can stack them; `n_kp` says
        how many rows are real."""
        coords = np.argwhere(kp_bool).astype(np.float32)
        n_kp = min(len(coords), self.max_keypoints)
        padded = np.zeros((self.max_keypoints, 3), dtype=np.float32)
        if n_kp:
            padded[:n_kp] = coords[:n_kp]
        return padded, n_kp

    @staticmethod
    def _keypoint_edt(kp_bool: np.ndarray, patch_size: list) -> np.ndarray:
        """Eq. 3's first term. A crop with no key points gets the diagonal everywhere,
        the limit of "nearest key point is infinitely far"."""
        if not kp_bool.any():
            d_max = float(np.linalg.norm(np.asarray(patch_size, dtype=np.float64)))
            return np.full(tuple(patch_size), d_max, dtype=np.float32)
        return distance_transform_edt(~kp_bool).astype(np.float32)
