"""Offline: dilated GT masks at 0.5mm for ADE-HTL's coarse stage (Sec. III-A.1).

    python -m models.ade_htl.precompute_dilated_masks -c configs/ade_htl_stage1_coarse.json

Reads the shared 0.5mm mask cache and writes a dilated sibling next to it, following
the same <scan_id>_dilated.npy convention base_dataset.py picks up automatically
whenever `dilate_mask` is configured.

An offline pass rather than the `dilate_mask` step at load time because a radius-2
dilation over a 0.5mm volume takes ~7s, once per sample.

Radius: the paper dilates with "a dilated kernel of size 5 x 5 x 5", and
skimage.morphology.ball(2) is exactly a 5x5x5 structuring element, so its kernel is
radius 2. Radius 5 would be an 11x11x11 ball, ~7x the lumen volume instead of ~2.5x.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must be set before SimpleITK/numpy: ITK sizes its thread pool to the full core
# count PER PROCESS, oversubscribing the CPU once several workers run.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from utils.config import BenchmarkConfig
from utils import io as bio
from utils.morphology import dilate_mask


# ball(DEFAULT_RADIUS) is a 5x5x5 structuring element -- the paper's kernel size.
DEFAULT_RADIUS = 2


def _process_scan(args: tuple) -> tuple:
    scan_id, in_path, out_path, radius, binarise_kwargs = args
    try:
        # The cache holds the raw multi-label mask, and dilation is a binary op.
        mask = np.load(in_path, mmap_mode="r")
        lumen = bio.binarise_lumen(np.asarray(mask), **binarise_kwargs)
        if not lumen.any():
            return scan_id, False, "empty GT lumen mask"
        dilated = dilate_mask(lumen, radius).astype(np.uint8)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, dilated)
        ratio = dilated.sum() / max(int(lumen.sum()), 1)
        return scan_id, True, (f"{int(lumen.sum())} -> {int(dilated.sum())} voxels "
                               f"({ratio:.1f}x), shape={tuple(dilated.shape)}")
    except Exception as e:
        return scan_id, False, f"{type(e).__name__}: {e}"


def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to the ADE-HTL coarse-stage config JSON")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                        help="Default 'all' — the coarse net trains on train/val and is "
                             "predicted over every split to build the ADE assets.")
    parser.add_argument("--radius", type=int, default=None,
                        help=f"Spherical SE radius in voxels. Defaults to the config's "
                             f"preprocessing.params.dilate_mask.radius, else {DEFAULT_RADIUS} "
                             f"(= the paper's 5x5x5 kernel).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)
    p = cfg.data.params

    masks_dir = _resolve(cfg.data.data_root,
                         p.get("masks_resampled_dir", "segmentations_resampled"))
    radius = args.radius
    if radius is None:
        radius = int(cfg.preprocessing.params.get("dilate_mask", {})
                     .get("radius", DEFAULT_RADIUS))

    binarise_kwargs = {
        "background_label": p.get("background_label", bio.LUMEN_BACKGROUND_LABEL),
    }

    if args.split == "all":
        scan_ids = sorted(set(cfg.data.train_ids) | set(cfg.data.val_ids) | set(cfg.data.test_ids))
    else:
        scan_ids = {"train": cfg.data.train_ids, "val": cfg.data.val_ids,
                    "test": cfg.data.test_ids}[args.split]
    if not scan_ids:
        raise ValueError(f"No scan IDs for split '{args.split}' — check data.filelist_dir in the config.")

    print(f"[dilate] {len(scan_ids)} scans  split={args.split}  radius={radius} "
          f"(ball {2 * radius + 1}^3)")
    print(f"[dilate] masks_dir={masks_dir}  (reads <id>.npy, writes <id>_dilated.npy)")

    jobs, skipped = [], 0
    for scan_id in scan_ids:
        in_path = os.path.join(masks_dir, f"{scan_id}.npy")
        out_path = os.path.join(masks_dir, f"{scan_id}_dilated.npy")
        if not os.path.exists(in_path):
            print(f"  {scan_id}: SKIP — no 0.5mm mask cache at {in_path}; run "
                  f"`python -m utils.offline_resample_images_to_disk -c {args.config}` first")
            continue
        if not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        jobs.append((scan_id, in_path, out_path, radius, binarise_kwargs))

    print(f"[dilate] {skipped} already cached, {len(jobs)} to process (workers={args.workers})\n")

    t0 = time.perf_counter()
    ok_count, failed = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_scan, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            scan_id, ok, message = future.result()
            print(f"  [{i}/{len(jobs)}] {scan_id}: {'ok' if ok else 'FAILED'}  {message}")
            ok_count += ok
            if not ok:
                failed.append(scan_id)

    print(f"\n[dilate] done in {time.perf_counter() - t0:.1f}s  ok={ok_count}  "
          f"failed={len(failed)}  skipped={skipped}")
    if failed:
        print(f"[dilate] scans with no dilated mask ({len(failed)}): {' '.join(failed)}")


if __name__ == "__main__":
    main()
