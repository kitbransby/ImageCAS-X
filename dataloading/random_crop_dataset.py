import math
import random

import numpy as np
import torch
from torch.utils.data import Sampler

from dataloading.base_dataset import BaseLumenDataset
from utils.config import BenchmarkConfig
from utils.seeding import SEED


class RandomCropDataset(BaseLumenDataset):
    """Randomly crops fixed-size patches from full volumes; shared by every
    patch-based method, which differ only in `patch_size`.

    On train/val every access draws a fresh random scan and origin. `idx` is NOT a
    scan selector here — it carries a bool from ForegroundQuotaBatchSampler saying
    whether this batch slot must land on foreground. Test returns the full volume,
    for sliding-window inference.
    """

    def __init__(self, config: BenchmarkConfig, split: str = "train",
                 preprocessing=None, augmentation=None):
        super().__init__(config, split)
        self.preprocessing = preprocessing
        self.augmentation = augmentation  # AugmentationPipeline or None

        p = config.data.params
        raw_ps = p.get("patch_size", [128, 160, 160])
        self.patch_size = raw_ps if isinstance(raw_ps, list) else [raw_ps] * 3

    def _random_origin(self, shape) -> tuple:
        """Uniform crop origin, no foreground bias."""
        px, py, pz = self.patch_size
        vx, vy, vz = shape
        x0 = random.randint(0, max(0, vx - px))
        y0 = random.randint(0, max(0, vy - py))
        z0 = random.randint(0, max(0, vz - pz))
        return x0, y0, z0

    def _crop_at(self, arr: np.ndarray, origin: tuple) -> np.ndarray:
        px, py, pz = self.patch_size
        x0, y0, z0 = origin
        patch = arr[x0:x0 + px, y0:y0 + py, z0:z0 + pz]
        if patch.shape != (px, py, pz):
            pad = [
                (0, max(0, px - patch.shape[0])),
                (0, max(0, py - patch.shape[1])),
                (0, max(0, pz - patch.shape[2])),
            ]
            patch = np.pad(patch, pad, constant_values=0)
        return patch[:px, :py, :pz]

    def _sample_crop_with_fg_quota(self, volume: np.ndarray, mask: np.ndarray,
                                   require_fg: bool, max_attempts: int = 20):
        """Uniform random crop, retried at a new origin if `require_fg` and the draw
        missed the vessel — without which batches would skew almost entirely
        background at small patch sizes.

        Retries only crop the cheap uint8 mask; the heavier, often mmap-backed volume
        is cropped once, after a location is accepted. The origin is returned because
        subclasses crop other per-scan arrays at the same place."""
        origin = self._random_origin(volume.shape)
        msk_patch = self._crop_at(mask, origin)
        if require_fg:
            for _ in range(max_attempts - 1):
                if msk_patch.any():
                    break
                origin = self._random_origin(volume.shape)
                msk_patch = self._crop_at(mask, origin)
        vol_patch = self._crop_at(volume, origin)
        return vol_patch, msk_patch, origin

    def __getitem__(self, idx) -> dict:
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

        if endless:
            vol, msk, _ = self._sample_crop_with_fg_quota(vol, msk, require_fg=bool(idx))

            if self.split == "train" and self.augmentation is not None:
                aug = self.augmentation({"volume": vol, "mask": msk,
                                         "spacing": sample["spacing"]})
                vol, msk = aug["volume"], aug["mask"]

        out = {
            "volume": torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(0),
            "mask": torch.from_numpy(np.ascontiguousarray(msk)).long(),
            "scan_id": scan_id,
            "spacing": sample["spacing"],
        }

        out["centerlines"] = self._load_centerlines(scan_id)

        return out


class ForegroundQuotaBatchSampler(Sampler):
    """Yields `n_batches` batches of `batch_size` bools, each marking whether that
    slot must be a foreground-containing crop. `n_batches` is fixed regardless of
    dataset size, which is what makes the dataset "endless" to the DataLoader."""

    def __init__(self, n_batches: int, batch_size: int, min_fg_fraction: float = 0.5,
                seed: int = SEED):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_fg = min(batch_size, math.ceil(batch_size * min_fg_fraction))
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            flags = [True] * self.n_fg + [False] * (self.batch_size - self.n_fg)
            self._rng.shuffle(flags)
            yield flags
