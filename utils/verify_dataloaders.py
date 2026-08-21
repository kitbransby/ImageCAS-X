"""Visually sanity-check a method's dataloaders, augmentation included: builds the
same pipeline train.py uses and saves one PNG per batch, showing each sample's middle
axial slice alone and with the GT mask boundary overlaid.

Usage:
    python -m utils.verify_dataloaders -c configs/cas_net.json
    python -m utils.verify_dataloaders -c configs/cas_net.json --n-batches 5 --out-dir /tmp/check
"""
import argparse
import os

# Must be set before numpy/torch/SimpleITK -- see train.py's identical guard.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from utils.config import BenchmarkConfig
from utils.seeding import seed_everything
from dataloading.factory import build_dataloader
from preprocessing.pipeline import build_preprocessing
from augmentation.pipeline import build_augmentation


def _plot_batch(batch: dict, title: str, out_path: str) -> None:
    """Rows are [volume, volume + mask boundary]; one column per sample."""
    volume = batch["volume"]  # (B, 1, X, Y, Z)
    mask = batch["mask"]      # (B, X, Y, Z)
    scan_ids = batch["scan_id"]
    B = volume.shape[0]

    fig, axes = plt.subplots(2, B, figsize=(3 * B, 6), squeeze=False)
    for i in range(B):
        vol = volume[i, 0].numpy()
        msk = mask[i].numpy()
        z = vol.shape[2] // 2
        vol_slice = vol[:, :, z].T
        msk_slice = msk[:, :, z].T

        boundary = find_boundaries((msk_slice > 0), mode="inner")
        boundary = np.ma.masked_where(~boundary, boundary.astype(np.float32))

        axes[0, i].imshow(vol_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, i].set_title(scan_ids[i], fontsize=8)

        axes[1, i].imshow(vol_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, i].imshow(boundary, cmap="autumn", origin="lower", alpha=0.5, vmin=0, vmax=1)

        for row in (0, 1):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    axes[0, 0].set_ylabel("volume", fontsize=8)
    axes[1, 0].set_ylabel("volume + mask boundary", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _print_batch_stats(batch: dict, split: str, batch_idx: int) -> None:
    volume = batch["volume"]
    mask = batch["mask"]
    for i, scan_id in enumerate(batch["scan_id"]):
        vol = volume[i, 0].numpy()
        msk = mask[i].numpy()
        fg_frac = float((msk > 0).mean())
        print(f"    [{split} batch {batch_idx}] {scan_id}: "
              f"vol shape={tuple(vol.shape)} min={vol.min():.4f} max={vol.max():.4f} "
              f"mean={vol.mean():.4f} | mask fg_frac={fg_frac:.5f}")


def verify_dataloaders(config_path: str, out_dir: str, n_batches: int,
                       num_workers: int = None) -> None:
    seed_everything()
    cfg = BenchmarkConfig.from_json(config_path)
    cfg.validate()
    if num_workers is not None:
        cfg.training.num_workers = num_workers

    os.makedirs(out_dir, exist_ok=True)

    preprocessing = build_preprocessing(cfg)
    augmentation = build_augmentation(cfg.augmentation) if cfg.augmentation.get("steps") else None

    # As in train.py, augmentation is only applied to the train split.
    train_loader = build_dataloader(cfg, split="train", preprocessing=preprocessing,
                                    augmentation=augmentation)
    val_loader = build_dataloader(cfg, split="val", preprocessing=preprocessing)

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break
            _print_batch_stats(batch, split, batch_idx)
            out_path = os.path.join(out_dir, f"{split}_batch{batch_idx}.png")
            _plot_batch(batch, f"{cfg.method_name} - {split} batch {batch_idx}", out_path)
            print(f"  -> saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to method config JSON")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to save PNGs (default: dataloader_check/<method_name>)")
    parser.add_argument("--n-batches", type=int, default=3,
                        help="Number of batches to dump per split (default: 3)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override config's training.num_workers (default: use config value)")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        method_name = BenchmarkConfig.from_json(args.config).method_name
        out_dir = os.path.join("dataloader_check", method_name)

    verify_dataloaders(args.config, out_dir, args.n_batches, num_workers=args.num_workers)
