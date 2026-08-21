import json
import os
from abc import ABC, abstractmethod

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from utils.config import BenchmarkConfig
from utils import io as bio
from preprocessing.steps import Resample, ResampleToShape, DilateMask


class BaseLumenDataset(Dataset, ABC):
    """Common interface for all benchmark datasets, differing in what __getitem__
    returns and how preprocessing is applied.

    Two offline .npy caches replace re-running sitk load+resample per access, and the
    steps they bake in are skipped at runtime: 'resample' selects the 0.5mm-iso cache,
    'resample_to_shape' the 128x128x64 one. Both keep a dilated mask alongside the
    plain one, chosen by whether 'dilate_mask' is also configured.
    """

    def __init__(self, config: BenchmarkConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.scan_ids = self._get_split_ids()

        self.use_resampled_cache = "resample" in self.config.preprocessing.steps
        if self.use_resampled_cache:
            self.target_spacing = float(
                self.config.preprocessing.params.get("resample", {}).get("target_spacing", 0.5)
            )
            p = self.config.data.params
            self._volumes_resampled_dir = self._resolve(p.get("volumes_resampled_dir", "volumes_resampled"))
            self._masks_resampled_dir = self._resolve(p.get("masks_resampled_dir", "segmentations_resampled"))
            # Reading the precomputed dilated sibling, because a radius-2 ball over
            # a 0.5mm volume takes ~7s per __getitem__.
            self._use_dilated_mask_resampled = "dilate_mask" in self.config.preprocessing.steps

        self.use_128_cache = "resample_to_shape" in self.config.preprocessing.steps
        if self.use_128_cache:
            p = self.config.data.params
            self._volumes_128_dir = self._resolve(p.get("volumes_128_dir", "volumes_resampled_128"))
            self._masks_128_dir = self._resolve(p.get("masks_128_dir", "segmentations_resampled_128"))
            self._use_dilated_mask_128 = "dilate_mask" in self.config.preprocessing.steps
            spacing_path = os.path.join(self._volumes_128_dir, "_spacing.json")
            with open(spacing_path, encoding="utf-8") as f:
                self._spacing_128 = json.load(f)

    def _get_split_ids(self) -> list:
        mapping = {
            "train": self.config.data.train_ids,
            "val": self.config.data.val_ids,
            "test": self.config.data.test_ids,
        }
        ids = mapping.get(self.split)
        if ids is None:
            raise ValueError(f"Unknown split: {self.split}")
        return ids

    def _resolve(self, rel_or_abs: str) -> str:
        """Join with data_root when the path is relative."""
        if not rel_or_abs or os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.config.data.data_root, rel_or_abs)

    def _load_volume(self, scan_id: str):
        """sitk_img is None in cache mode: no resample step is left to run."""
        if self.use_128_cache:
            path = os.path.join(self._volumes_128_dir, f"{scan_id}.npy")
            return np.load(path, mmap_mode="r"), None
        if self.use_resampled_cache:
            path = os.path.join(self._volumes_resampled_dir, f"{scan_id}.npy")
            return np.load(path, mmap_mode="r"), None
        suffix = self.config.data.volume_suffix or self.config.data.file_extension
        path = bio.resolve_scan_path(self._resolve(self.config.data.volume_dir), scan_id, suffix)
        return bio.load_volume(path)

    def _load_gt_mask(self, scan_id: str):
        """Returns the RAW multi-label mask, so `_binarise_mask()` must be applied
        downstream. The dilated variants are already binarised, dilation being a
        binary op; re-binarising one is a harmless no-op under the default labels."""
        if self.use_128_cache:
            fname = f"{scan_id}_dilated.npy" if self._use_dilated_mask_128 else f"{scan_id}.npy"
            path = os.path.join(self._masks_128_dir, fname)
            return np.load(path, mmap_mode="r"), None
        if self.use_resampled_cache:
            fname = (f"{scan_id}_dilated.npy" if self._use_dilated_mask_resampled
                     else f"{scan_id}.npy")
            path = os.path.join(self._masks_resampled_dir, fname)
            return np.load(path, mmap_mode="r"), None
        suffix = self.config.data.mask_suffix or self.config.data.file_extension
        path = bio.resolve_scan_path(self._resolve(self.config.data.gt_mask_dir), scan_id, suffix)
        return bio.load_mask(path)

    def _binarise_mask(self, mask: np.ndarray) -> np.ndarray:
        p = self.config.data.params
        if not p.get("binarise_lumen", True):
            return mask
        return bio.binarise_lumen(
            mask,
            background_label=p.get("background_label", bio.LUMEN_BACKGROUND_LABEL),
        )

    def _spacing(self, sitk_img, scan_id: str = None) -> tuple:
        if sitk_img is not None:
            return sitk_img.GetSpacing()
        if self.use_128_cache:
            # Anisotropic and scan-specific, unlike the 0.5mm-iso cache.
            return tuple(self._spacing_128[scan_id])
        return (self.target_spacing,) * 3

    def _binarised_sitk_mask(self, mask_bin: np.ndarray, sitk_mask):
        """Rebuild sitk_mask's geometry around an already-binarised array.

        The resample steps operate on this sitk image, not sample['mask'], so it must
        carry the binarised values or the resampled mask silently reverts to
        multi-label. No-op in cache mode, where no such step runs.
        """
        if sitk_mask is None:
            return None
        bin_img = sitk.GetImageFromArray(mask_bin.transpose(2, 1, 0))
        bin_img.CopyInformation(sitk_mask)
        return bin_img

    def _run_preprocessing(self, sample: dict) -> dict:
        """Run preprocessing, skipping steps already baked into the cache — they
        would be redundant and need a sitk image neither cache path has."""
        preprocessing = getattr(self, "preprocessing", None)
        if preprocessing is None:
            return sample
        skip_types = ()
        if self.use_resampled_cache:
            skip_types += (Resample,)
            if self._use_dilated_mask_resampled:
                skip_types += (DilateMask,)  # already dilated on disk
        if self.use_128_cache:
            skip_types += (ResampleToShape, DilateMask)
        if not skip_types:
            return preprocessing(sample)
        for step in preprocessing.steps:
            if isinstance(step, skip_types):
                continue
            sample = step(sample)
        return sample

    def _centerlines_dir(self) -> str:
        rel = self.config.data.params.get("centerlines_dir", "centerlines")
        return self._resolve(rel)

    def _load_centerlines(self, scan_id: str) -> dict:
        """Left and right GT centerline VTKs, combined and per-side, in LPS mm.
        Missing files come back as zeros((0, 3))."""
        cdir = self._centerlines_dir()
        left_pts = right_pts = None
        left_scalars = right_scalars = {}

        for side in ("left", "right"):
            path = os.path.join(cdir, f"{scan_id}.coronary_{side}_centerline.vtk")
            if not os.path.exists(path):
                print(f"    [warn] centerline not found: {path}")
                continue
            try:
                pts, scl = bio.load_vtk_centerline(path)
                if side == "left":
                    left_pts, left_scalars = pts, scl
                else:
                    right_pts, right_scalars = pts, scl
            except Exception as e:
                print(f"    [warn] failed to load centerline {path}: {e}")

        _empty = np.zeros((0, 3), dtype=np.float64)
        left_pts = left_pts if left_pts is not None else _empty
        right_pts = right_pts if right_pts is not None else _empty

        available = [p for p in (left_pts, right_pts) if len(p) > 0]
        combined = np.concatenate(available, axis=0) if available else _empty

        return {
            "combined_points": combined,
            "left_points":     left_pts,
            "left_scalars":    left_scalars,
            "right_points":    right_pts,
            "right_scalars":   right_scalars,
        }

    def __len__(self) -> int:
        return len(self.scan_ids)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Must return a dict with at least: 'volume', 'mask', 'scan_id', 'spacing'."""
        raise NotImplementedError
