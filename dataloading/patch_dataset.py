import os
import random
import numpy as np
import torch
from dataloading.base_dataset import BaseLumenDataset
from utils.config import BenchmarkConfig


class PatchDataset(BaseLumenDataset):
    """Fixed-size 3D patches centred on per-scan {scan_id}.npy centre-point files in
    data.params.centers_dir, written by stage3_generate_centers.py."""

    def __init__(self, config: BenchmarkConfig, split: str = "train",
                 preprocessing=None, augmentation=None):
        super().__init__(config, split)
        self.preprocessing = preprocessing
        self.augmentation = augmentation

        p = config.data.params
        raw_ps = p.get("patch_size", 64)
        self.patch_size = raw_ps if isinstance(raw_ps, int) else int(raw_ps[0])
        self.centers_dir = p.get("centers_dir", "")
        self.skip_empty = p.get("skip_empty", True)

        self._index = self._build_index()
        self._centerline_cache: dict = {}

    def _build_index(self) -> list:
        if not self.centers_dir:
            raise ValueError(
                "data.params.centers_dir must be set for PatchDataset. "
                "Generate per-scan .npy center files (shape (N, 3) voxel coords) "
                "and set centers_dir to the containing directory."
            )

        index = []
        for scan_id in self.scan_ids:
            path = os.path.join(self.centers_dir, f"{scan_id}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Centre file not found: {path}\n"
                    f"Generate skeleton/center .npy files for all {self.split} scan IDs "
                    f"before training the patch model."
                )
            centers = np.load(path)
            if centers.ndim != 2 or centers.shape[1] != 3:
                raise ValueError(
                    f"Expected (N, 3) array in {path}, got shape {centers.shape}."
                )
            for c in centers:
                index.append((scan_id, c))

        if len(index) == 0:
            raise ValueError(
                f"No patch centres found for {self.split} split. "
                f"All .npy files in {self.centers_dir} appear to be empty."
            )
        return index

    def __len__(self) -> int:
        return len(self._index)

    def _extract_patch(self, volume: np.ndarray, mask: np.ndarray,
                       center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A patch_size^3 region around `center`, zero-padded at boundaries."""
        half = self.patch_size // 2
        cx, cy, cz = int(round(center[0])), int(round(center[1])), int(round(center[2]))

        x0, x1 = cx - half, cx + half + 1
        y0, y1 = cy - half, cy + half + 1
        z0, z1 = cz - half, cz + half + 1

        pad = [
            (max(0, -x0), max(0, x1 - volume.shape[0])),
            (max(0, -y0), max(0, y1 - volume.shape[1])),
            (max(0, -z0), max(0, z1 - volume.shape[2])),
        ]
        vx0, vx1 = max(0, x0), min(volume.shape[0], x1)
        vy0, vy1 = max(0, y0), min(volume.shape[1], y1)
        vz0, vz1 = max(0, z0), min(volume.shape[2], z1)

        vol_patch = volume[vx0:vx1, vy0:vy1, vz0:vz1]
        msk_patch = mask[vx0:vx1, vy0:vy1, vz0:vz1]

        if any(p[0] > 0 or p[1] > 0 for p in pad):
            vol_patch = np.pad(vol_patch, pad, mode="constant", constant_values=0)
            msk_patch = np.pad(msk_patch, pad, mode="constant", constant_values=0)

        ps = self.patch_size
        return vol_patch[:ps, :ps, :ps], msk_patch[:ps, :ps, :ps]

    def __getitem__(self, idx: int, _depth: int = 0) -> dict:
        scan_id, center = self._index[idx]

        if self.use_resampled_cache:
            # Slice straight out of the mmap rather than reloading+resampling the
            # whole volume on every one of a scan's centre-point draws.
            volume, sitk_img = self._load_volume(scan_id)
            mask, _ = self._load_gt_mask(scan_id)
            vol_patch, msk_patch = self._extract_patch(volume, mask, center)
            msk_patch = self._binarise_mask(msk_patch)
            sample = {
                "volume": vol_patch.astype(np.float32),
                "mask": msk_patch.astype(np.uint8),
                "spacing": self._spacing(sitk_img),
            }
            sample = self._run_preprocessing(sample)  # normalise-type steps only
        else:
            volume, sitk_img = self._load_volume(scan_id)
            mask, sitk_mask = self._load_gt_mask(scan_id)
            mask = self._binarise_mask(np.asarray(mask))
            full_sample = {
                "volume": volume.astype(np.float32),
                "mask": mask.astype(np.uint8),
                "scan_id": scan_id,
                "spacing": self._spacing(sitk_img),
                "sitk_img": sitk_img,
                "sitk_mask": self._binarised_sitk_mask(mask, sitk_mask),
            }
            full_sample = self._run_preprocessing(full_sample)

            vol_patch, msk_patch = self._extract_patch(
                full_sample["volume"], full_sample["mask"], center
            )
            sample = {
                "volume": vol_patch.astype(np.float32),
                "mask": msk_patch.astype(np.uint8),
                "spacing": full_sample["spacing"],
            }

        if self.skip_empty and not sample["mask"].any():
            # Centre outside the GT mask, e.g. a stale centre file. Retry rather
            # than crash the run; bounded so the recursion cannot run away.
            if _depth < 10 and len(self._index) > 1:
                return self.__getitem__(random.randrange(len(self._index)), _depth + 1)
            print(
                f"    [warn] empty mask patch for scan '{scan_id}' at center "
                f"{center.tolist()} (retries exhausted); returning it as-is. "
                f"Review the .npy centre file at "
                f"{os.path.join(self.centers_dir, scan_id + '.npy')}."
            )

        sample["scan_id"] = scan_id
        sample["center"] = center.astype(np.float32)

        if self.augmentation is not None and self.split == "train":
            sample = self.augmentation(sample)

        sample["volume"] = torch.from_numpy(sample["volume"]).float().unsqueeze(0)
        sample["mask"] = torch.from_numpy(sample["mask"]).long()

        if scan_id not in self._centerline_cache:
            self._centerline_cache[scan_id] = self._load_centerlines(scan_id)
        sample["centerlines"] = self._centerline_cache[scan_id]

        return sample
