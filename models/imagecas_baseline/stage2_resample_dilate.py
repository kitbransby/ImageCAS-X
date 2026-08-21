"""Offline: resample the 0.5mm cache down to the fixed 128x128x64 grid ImageCAS
Stage 1/2 train on, so training can mmap it instead of re-running load + dilate +
resample per __getitem__.

Both a plain and a dilated binary lumen mask are written, the latter being Stage 2's
Eq. 2 training target. Dilation runs BEFORE the resize, at the source resolution:
radius is a physical distance, so dilating after downsampling would inflate it well
beyond the intended ~2.5mm.

Usage:
    python -m models.imagecas_baseline.stage2_resample_dilate -c configs/imagecas_stage2_coarse_dilated.json
    python -m models.imagecas_baseline.stage2_resample_dilate -c configs/imagecas_stage2_coarse_dilated.json \
        --target-shape 128 128 64 --dilate-radius 5 --workers 8 --overwrite
"""
import argparse
import json
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
from utils.morphology import dilate_mask


def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def _worker_init():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def _resample_nn(mask_img: sitk.Image, reference_img: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask_img)


def _process_scan(args: tuple) -> tuple:
    """Resample one scan's cached volume + mask to target_shape, dilate the mask
    for the Stage 2 variant, and save volume/plain-mask/dilated-mask as .npy.
    Returns (scan_id, ok, message, new_spacing_or_None)."""
    (scan_id, in_vol_path, in_mask_path, out_vol_path, out_mask_path, out_mask_dilated_path,
     target_shape, source_spacing, dilate_radius) = args
    try:
        volume = np.load(in_vol_path).astype(np.float32)          # (X,Y,Z), 0.5mm iso
        mask_multilabel = np.load(in_mask_path).astype(np.uint8)  # (X,Y,Z), 0.5mm iso, raw multi-label
        mask_bin = bio.binarise_lumen(mask_multilabel)
        mask_dilated = dilate_mask(mask_bin, radius=dilate_radius)

        spacing = (source_spacing, source_spacing, source_spacing)
        img = sitk.GetImageFromArray(volume.transpose(2, 1, 0))
        img.SetSpacing(spacing)

        mask_bin_img = sitk.GetImageFromArray(mask_bin.transpose(2, 1, 0))
        mask_bin_img.CopyInformation(img)
        mask_dilated_img = sitk.GetImageFromArray(mask_dilated.transpose(2, 1, 0))
        mask_dilated_img.CopyInformation(img)

        orig_size = img.GetSize()
        orig_spacing = img.GetSpacing()
        tx, ty, tz = (int(v) for v in target_shape)
        new_spacing = [orig_size[i] * orig_spacing[i] / target_shape[i] for i in range(3)]

        vol_resampler = sitk.ResampleImageFilter()
        vol_resampler.SetOutputSpacing(new_spacing)
        vol_resampler.SetSize([tx, ty, tz])
        vol_resampler.SetOutputDirection(img.GetDirection())
        vol_resampler.SetOutputOrigin(img.GetOrigin())
        vol_resampler.SetInterpolator(sitk.sitkLinear)
        vol_resampler.SetDefaultPixelValue(0)
        img_r = vol_resampler.Execute(img)

        mask_bin_r = _resample_nn(mask_bin_img, img_r)
        mask_dilated_r = _resample_nn(mask_dilated_img, img_r)

        volume_r = sitk.GetArrayFromImage(img_r).transpose(2, 1, 0).astype(np.float32)
        mask_bin_r_arr = sitk.GetArrayFromImage(mask_bin_r).transpose(2, 1, 0).astype(np.uint8)
        mask_dilated_r_arr = sitk.GetArrayFromImage(mask_dilated_r).transpose(2, 1, 0).astype(np.uint8)

        for p in (out_vol_path, out_mask_path, out_mask_dilated_path):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(out_vol_path, volume_r)
        np.save(out_mask_path, mask_bin_r_arr)
        np.save(out_mask_dilated_path, mask_dilated_r_arr)

        return scan_id, True, f"shape={volume_r.shape}", new_spacing
    except Exception as e:
        return scan_id, False, str(e), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to a method config JSON (e.g. configs/imagecas_stage2_coarse_dilated.json) "
                             "-- used for data_root/splits, and to default target-shape/dilate-radius/"
                             "target-spacing from its preprocessing.params if not passed explicitly.")
    parser.add_argument("--target-spacing", type=float, default=None,
                        help="Isotropic spacing (mm) of the SOURCE volumes_resampled/segmentations_resampled "
                             "cache. Defaults to the config's preprocessing.params.resample.target_spacing, or 0.5.")
    parser.add_argument("--target-shape", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Output voxel shape. Defaults to the config's "
                             "preprocessing.params.resample_to_shape.target_shape, or [128, 128, 64].")
    parser.add_argument("--dilate-radius", type=int, default=None,
                        help="Spherical structuring element radius (voxels, at source spacing) for the "
                             "dilated mask variant. Defaults to the config's "
                             "preprocessing.params.dilate_mask.radius, or 5.")
    parser.add_argument("--volumes-in-dir", default="volumes_resampled")
    parser.add_argument("--masks-in-dir", default="segmentations_resampled")
    parser.add_argument("--volumes-out-dir", default="volumes_resampled_128")
    parser.add_argument("--masks-out-dir", default="segmentations_resampled_128")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process scans whose .npy outputs already exist.")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)

    target_spacing = args.target_spacing
    if target_spacing is None:
        target_spacing = cfg.preprocessing.params.get("resample", {}).get("target_spacing", 0.5)

    target_shape = args.target_shape
    if target_shape is None:
        target_shape = cfg.preprocessing.params.get("resample_to_shape", {}).get("target_shape", [128, 128, 64])
    target_shape = tuple(int(v) for v in target_shape)

    dilate_radius = args.dilate_radius
    if dilate_radius is None:
        dilate_radius = cfg.preprocessing.params.get("dilate_mask", {}).get("radius", 5)

    scan_ids = sorted(set(cfg.data.train_ids) | set(cfg.data.val_ids) | set(cfg.data.test_ids))
    if not scan_ids:
        raise ValueError(
            "No scan IDs found across train/val/test splits -- check data.filelist_dir "
            f"in {args.config}."
        )

    in_vol_dir = _resolve(cfg.data.data_root, args.volumes_in_dir)
    in_mask_dir = _resolve(cfg.data.data_root, args.masks_in_dir)
    out_vol_dir = _resolve(cfg.data.data_root, args.volumes_out_dir)
    out_mask_dir = _resolve(cfg.data.data_root, args.masks_out_dir)

    print(f"[resample_dilate_128] {len(scan_ids)} scans  target_shape={target_shape}  "
          f"source_spacing={target_spacing}mm  dilate_radius={dilate_radius}")
    print(f"[resample_dilate_128] volumes: {in_vol_dir}  ->  {out_vol_dir}")
    print(f"[resample_dilate_128] masks:   {in_mask_dir}  ->  {out_mask_dir}  "
          f"(<scan_id>.npy = plain, <scan_id>_dilated.npy = dilated)")

    jobs = []
    skipped = 0
    for scan_id in scan_ids:
        in_vol_path = os.path.join(in_vol_dir, f"{scan_id}.npy")
        in_mask_path = os.path.join(in_mask_dir, f"{scan_id}.npy")
        out_vol_path = os.path.join(out_vol_dir, f"{scan_id}.npy")
        out_mask_path = os.path.join(out_mask_dir, f"{scan_id}.npy")
        out_mask_dilated_path = os.path.join(out_mask_dir, f"{scan_id}_dilated.npy")
        if not args.overwrite and all(os.path.exists(p) for p in
                                       (out_vol_path, out_mask_path, out_mask_dilated_path)):
            skipped += 1
            continue
        jobs.append((scan_id, in_vol_path, in_mask_path, out_vol_path, out_mask_path,
                     out_mask_dilated_path, target_shape, target_spacing, dilate_radius))

    print(f"[resample_dilate_128] {skipped} already cached, {len(jobs)} to process "
          f"(workers={args.workers})\n")

    t0 = time.perf_counter()
    ok_count, fail_count = 0, 0
    spacing_map = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_process_scan, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            scan_id, ok, message, new_spacing = future.result()
            status = "ok" if ok else "FAILED"
            print(f"  [{i}/{len(jobs)}] {scan_id}: {status}  {message}")
            if ok:
                ok_count += 1
                spacing_map[scan_id] = list(new_spacing)
            else:
                fail_count += 1

    # Per-scan spacing sidecar: ResampleToShape's output spacing is anisotropic
    # and scan-specific (depends on each scan's original size), unlike the
    # plain 0.5mm-iso cache where a single constant applies to every scan.
    # Consumed by dataloading/base_dataset.py's `_spacing()` in 128-cache mode.
    spacing_path = os.path.join(out_vol_dir, "_spacing.json")
    existing = {}
    if os.path.exists(spacing_path):
        with open(spacing_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(spacing_map)
    bio.save_results(existing, spacing_path)

    total = time.perf_counter() - t0
    print(f"\n[resample_dilate_128] done in {total:.1f}s  "
          f"ok={ok_count}  failed={fail_count}  skipped={skipped}")
    if fail_count:
        print("[resample_dilate_128] re-run with --overwrite to retry failed scans "
              "after investigating the errors above.")


if __name__ == "__main__":
    main()
