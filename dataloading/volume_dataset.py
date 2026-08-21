import numpy as np
import torch
from dataloading.base_dataset import BaseLumenDataset
from utils.config import BenchmarkConfig


class VolumeDataset(BaseLumenDataset):
    """Returns the full 3D volume and mask for methods that take whole-scan input."""

    def __init__(self, config: BenchmarkConfig, split: str = "train",
                 preprocessing=None, augmentation=None):
        super().__init__(config, split)
        self.preprocessing = preprocessing
        self.augmentation = augmentation  # AugmentationPipeline or None

    def __getitem__(self, idx: int) -> dict:
        scan_id = self.scan_ids[idx]

        volume, sitk_img = self._load_volume(scan_id)
        mask, sitk_mask = self._load_gt_mask(scan_id)
        mask = self._binarise_mask(np.asarray(mask))

        sample = {
            "volume": np.asarray(volume),
            "mask": mask,
            "scan_id": scan_id,
            "spacing": self._spacing(sitk_img, scan_id),
            "sitk_img": sitk_img,  # Resample needs the geometry
            "sitk_mask": self._binarised_sitk_mask(mask, sitk_mask),
        }

        sample = self._run_preprocessing(sample)

        if self.augmentation is not None and self.split == "train":
            sample = self.augmentation(sample)

        sample["volume"] = torch.from_numpy(sample["volume"]).float().unsqueeze(0)
        sample["mask"] = torch.from_numpy(sample["mask"]).long()
        # sitk objects are not batch-collatable
        sample.pop("sitk_img", None)
        sample.pop("sitk_mask", None)

        sample["centerlines"] = self._load_centerlines(scan_id)

        return sample
