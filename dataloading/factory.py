import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataloader import default_collate

from utils.config import BenchmarkConfig
from utils.seeding import SEED, worker_init_fn
from dataloading.volume_dataset import VolumeDataset
from dataloading.patch_dataset import PatchDataset
from dataloading.random_crop_dataset import RandomCropDataset, ForegroundQuotaBatchSampler
from dataloading.ade_htl_dataset import ADEHTLDataset


def _collate_fn(batch):
    """default_collate, but 'centerlines' stays a list: its arrays are
    variable-length and hold nested dicts, so they cannot be stacked."""
    centerlines = [item.pop("centerlines", None) for item in batch]
    collated = default_collate(batch)
    if any(c is not None for c in centerlines):
        collated["centerlines"] = centerlines
    return collated

_DATASET_REGISTRY = {
    "volume": VolumeDataset,
    "patch": PatchDataset,
    "random_crop": RandomCropDataset,
    "ade_htl_crop": ADEHTLDataset,
}


def build_dataset(config: BenchmarkConfig, split: str, preprocessing=None, augmentation=None):
    cls = _DATASET_REGISTRY.get(config.input_type)
    if cls is None:
        raise ValueError(f"Unknown input_type '{config.input_type}'. "
                         f"Options: {list(_DATASET_REGISTRY)}")
    return cls(config, split=split, preprocessing=preprocessing, augmentation=augmentation)


_FIXED_SHAPE_STEPS = {"resample_to_shape"}


def _is_batchable(config: BenchmarkConfig, split: str) -> bool:
    """Whether samples have a fixed shape and can be collated into a batch > 1.
    Variable-shape whole-volume samples must run at batch_size=1."""
    it = config.input_type
    if it == "patch":
        return True
    if it in ("random_crop", "ade_htl_crop"):
        return split in ("train", "val")
    if it == "volume":
        return any(s in _FIXED_SHAPE_STEPS for s in config.preprocessing.steps)
    return False


def build_dataloader(config: BenchmarkConfig, split: str,
                     preprocessing=None, augmentation=None) -> DataLoader:
    dataset = build_dataset(config, split, preprocessing, augmentation)
    batch_size = config.training.batch_size if _is_batchable(config, split) else 1

    # Train/val are "endless": an epoch is a fixed number of batches, not one pass
    # over the dataset, so cadence is comparable across methods.
    if split in ("train", "val"):
        n_iters = (config.training.train_iters_per_epoch if split == "train"
                  else config.training.val_iters_per_epoch)

        if config.input_type in ("random_crop", "ade_htl_crop"):
            # Small patches rarely land on foreground under uniform cropping.
            # ADEHTLDataset reads the same per-slot bool out of `idx`, so it needs
            # this sampler too — a RandomSampler would put a scan index there.
            min_fg = config.data.params.get("min_fg_fraction", 0.5)
            batch_sampler = ForegroundQuotaBatchSampler(
                n_batches=n_iters, batch_size=batch_size, min_fg_fraction=min_fg,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=config.training.num_workers,
                pin_memory=True,
                collate_fn=_collate_fn,
                worker_init_fn=worker_init_fn,
            )

        sampler = RandomSampler(
            dataset, replacement=True, num_samples=n_iters * batch_size,
            generator=torch.Generator().manual_seed(SEED),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config.training.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
            worker_init_fn=worker_init_fn,
        )

    # test split: one deterministic pass over every scan, in order.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        worker_init_fn=worker_init_fn,
    )


def build_eval_dataloader(config: BenchmarkConfig, preprocessing=None,
                          split: str = "test") -> DataLoader:
    """Whole preprocessed volumes, one scan at a time, whatever the training
    input_type — run_inference tiles internally, so metrics always see a full-volume
    prediction. A class setting `supplies_eval_volume` opts out of VolumeDataset,
    because its test branch carries extra input channels VolumeDataset knows nothing
    about. Non-test splits exist so a cascade's earlier stage can be run over the
    scans a later one needs."""
    cls = _DATASET_REGISTRY.get(config.input_type)
    if getattr(cls, "supplies_eval_volume", False):
        dataset = cls(config, split=split, preprocessing=preprocessing)
    else:
        dataset = VolumeDataset(config, split=split, preprocessing=preprocessing)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        worker_init_fn=worker_init_fn,
    )
