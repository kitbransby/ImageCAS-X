"""Offline: resample every volume + GT mask to a fixed isotropic spacing and cache
them as uncompressed .npy, so training can mmap patches straight off disk instead of
re-running sitk load+resample per __getitem__. Every config resamples to 0.5mm, so
this cache is shared across methods — run it once.

Masks are cached multi-label rather than binarised, so methods needing per-branch
labels are not stuck with a pre-binarised cache; binarising is cheap at load time.

Usage:
    python -m utils.offline_resample_images_to_disk -c configs/ffr_unet.json
    python -m utils.offline_resample_images_to_disk -c configs/ffr_unet.json \
        --target-spacing 0.5 --workers 8 --overwrite
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must be set before SimpleITK: ITK sizes its thread pool to the full core count PER
# PROCESS, oversubscribing the CPU once several workers run.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import SimpleITK as sitk

from utils.config import BenchmarkConfig
from utils import io as bio
from preprocessing.steps import Resample


def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def _worker_init():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def _process_scan(args: tuple) -> tuple[str, bool, str]:
    """Resample one scan and save as .npy. Returns (scan_id, ok, message)."""
    (scan_id, vol_path, mask_path, out_vol_path, out_mask_path, target_spacing) = args
    try:
        volume, sitk_img = bio.load_volume(vol_path)
        mask, sitk_mask = bio.load_mask(mask_path)

        sample = {
            "volume": volume.astype(np.float32),
            "mask": mask.astype(np.uint8),
            "sitk_img": sitk_img,
            "sitk_mask": sitk_mask,
        }
        sample = Resample(target_spacing=target_spacing)(sample)

        os.makedirs(os.path.dirname(out_vol_path), exist_ok=True)
        os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
        np.save(out_vol_path, sample["volume"].astype(np.float32))
        np.save(out_mask_path, sample["mask"].astype(np.uint8))
        return scan_id, True, f"shape={sample['volume'].shape}"
    except Exception as e:
        return scan_id, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to method config JSON")
    parser.add_argument("--target-spacing", type=float, default=None,
                        help="Isotropic spacing in mm. Defaults to the config's "
                             "preprocessing.params.resample.target_spacing, or 0.5 if absent.")
    parser.add_argument("--volumes-out-dir", default="volumes_resampled",
                        help="Output dir for resampled volumes, relative to data_root unless absolute.")
    parser.add_argument("--masks-out-dir", default="segmentations_resampled",
                        help="Output dir for resampled masks, relative to data_root unless absolute.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process scans whose .npy outputs already exist.")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)

    target_spacing = args.target_spacing
    if target_spacing is None:
        target_spacing = cfg.preprocessing.params.get("resample", {}).get("target_spacing", 0.5)

    scan_ids = sorted(set(cfg.data.train_ids) | set(cfg.data.val_ids) | set(cfg.data.test_ids))
    if not scan_ids:
        raise ValueError(
            "No scan IDs found across train/val/test splits — check data.filelist_dir "
            f"in {args.config}."
        )

    vol_suffix = cfg.data.volume_suffix or cfg.data.file_extension
    mask_suffix = cfg.data.mask_suffix or cfg.data.file_extension
    vol_dir = _resolve(cfg.data.data_root, cfg.data.volume_dir)
    mask_dir = _resolve(cfg.data.data_root, cfg.data.gt_mask_dir)
    out_vol_dir = _resolve(cfg.data.data_root, args.volumes_out_dir)
    out_mask_dir = _resolve(cfg.data.data_root, args.masks_out_dir)

    print(f"[offline_resample] {len(scan_ids)} scans  target_spacing={target_spacing}mm")
    print(f"[offline_resample] volumes:  {vol_dir}  ->  {out_vol_dir}")
    print(f"[offline_resample] masks:    {mask_dir}  ->  {out_mask_dir}")

    jobs = []
    skipped = 0
    for scan_id in scan_ids:
        out_vol_path = os.path.join(out_vol_dir, f"{scan_id}.npy")
        out_mask_path = os.path.join(out_mask_dir, f"{scan_id}.npy")
        if not args.overwrite and os.path.exists(out_vol_path) and os.path.exists(out_mask_path):
            skipped += 1
            continue
        vol_path = bio.resolve_scan_path(vol_dir, scan_id, vol_suffix)
        mask_path = bio.resolve_scan_path(mask_dir, scan_id, mask_suffix)
        jobs.append((scan_id, vol_path, mask_path, out_vol_path, out_mask_path, target_spacing))

    print(f"[offline_resample] {skipped} already cached, {len(jobs)} to process "
          f"(workers={args.workers})\n")

    t0 = time.perf_counter()
    ok_count, fail_count = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_process_scan, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            scan_id, ok, message = future.result()
            status = "ok" if ok else "FAILED"
            print(f"  [{i}/{len(jobs)}] {scan_id}: {status}  {message}")
            if ok:
                ok_count += 1
            else:
                fail_count += 1

    total = time.perf_counter() - t0
    print(f"\n[offline_resample] done in {total:.1f}s  "
          f"ok={ok_count}  failed={fail_count}  skipped={skipped}")
    if fail_count:
        print("[offline_resample] re-run with --overwrite to retry failed scans "
              "after investigating the errors above.")


if __name__ == "__main__":
    main()
