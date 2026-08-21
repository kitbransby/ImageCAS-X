"""Offline: ADE's region-of-interest and five anatomical distance field maps.

    python -m models.ade_htl.precompute_ade -c configs/ade_htl.json \
        --coarse-run <coarse run dir> --split all

Writes one <data_root>/<ade_dir>/<scan_id>.npz per scan holding the ROI bounds, the
full grid shape, the coarse-vessel voxel indices (the paper's point set P) and Eq.
2's normalised distance from each to each chamber surface.

Two inputs, neither trained here. Heart chambers come from TotalSegmentator's
`heartchambers_highres` output: its licence does not permit training new models on
its outputs, so the paper's own chamber ResUNet is deliberately not reimplemented and
the masks are an input at both train and test time. A missing chamber file therefore
fails that scan loudly rather than emitting a zero distance field, which would read
as "touching every chamber". The coarse vessel mask comes from the Stage 1 config,
predicted over every split.

The fields are stored sparsely because the paper defines them only over P: a dense
fp16 ROI array is ~0.5 GB per scan against ~2 MB here. The dataset scatters them back
into a dense 5-channel crop, filling non-P voxels with Eq. 2's clipped maximum.

Everything is on the shared 0.5mm isotropic grid the HTL network trains on.
Regenerate if the target spacing changes.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk

from utils.config import BenchmarkConfig
from utils import io as bio
from utils.precompute_centerline_samples import _resample_mask_onto


DEFAULT_ADE_DIR = "ade_2026"
DEFAULT_CHAMBERS_DIR = "TotalSegmentator"
DEFAULT_CHAMBERS_SUFFIX = ".heart.nii.gz"

# TotalSegmentator heartchambers_highres label indices, in Sec. III-A.2's order.
# The task also emits 1=myocardium and 7=pulmonary_artery; neither is one of ADE's
# five structures, but both count towards the heart bounding box.
CHAMBER_LABELS = (3, 5, 2, 4, 6)
CHAMBER_NAMES = ("LV", "RV", "LA", "RA", "aorta")
N_CHAMBERS = len(CHAMBER_LABELS)

# Eq. 2 divides distances by tau and clips at 1, so anything further than tau from a
# chamber reads as "far" and stops varying.
DEFAULT_TAU_MM = 50.0


def _bbox(mask: np.ndarray) -> tuple:
    """Half-open bounds of a mask's tight bounding box. Per-axis any() rather than
    argwhere, which would materialise an (N, 3) index array of millions of voxels."""
    lo, hi = [], []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        hit = np.flatnonzero(mask.any(axis=others))
        lo.append(int(hit[0]))
        hi.append(int(hit[-1]) + 1)
    return np.array(lo), np.array(hi)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Inner boundary voxels — the voxel equivalent of the paper's chamber surface
    point set. Its subsampling to 8000 points exists only to bound a brute-force
    distance computation that an exact EDT does not need."""
    from scipy.ndimage import binary_erosion

    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def _distance_fields(chambers: np.ndarray, spacing: np.ndarray,
                     tau_mm: float) -> tuple:
    """Eqs. 1-2 over a cropped chamber label volume, plus the names of any structures
    absent from this scan. Eq. 1's min over a surface point set is exactly an EDT
    seeded on that surface, so one EDT per structure replaces the paper's brute-force
    pass and is exact rather than subsampled."""
    from scipy.ndimage import distance_transform_edt

    dist = np.empty((N_CHAMBERS, *chambers.shape), dtype=np.float32)
    missing = []
    for c, label in enumerate(CHAMBER_LABELS):
        surface = _surface(chambers == label)
        if not surface.any():
            # Every voxel is infinitely far, which Eq. 2 clips to 1. Filling 0
            # instead would read as lying on the chamber surface.
            dist[c] = 1.0
            missing.append(CHAMBER_NAMES[c])
            continue
        s = distance_transform_edt(~surface, sampling=spacing)
        np.clip(s / tau_mm, 0.0, 1.0, out=s)
        dist[c] = s
    return dist, missing


def _process_scan(args: tuple) -> tuple:
    (scan_id, vol_path, chambers_path, coarse_path, out_path,
     target_spacing, tau_mm, margin_mm) = args
    try:
        if not os.path.exists(chambers_path):
            return scan_id, False, (
                f"no TotalSegmentator chamber mask at {chambers_path} — ADE cannot be "
                f"built for this scan and it cannot be predicted; add it to "
                f"filelist/exclude.txt if the file will never exist")
        if coarse_path is None:
            return scan_id, False, (
                "no coarse vessel prediction found — run `python -m inference -c "
                "configs/ade_htl_stage1_coarse.json -r <run> --split <this scan's split>` first")

        # Geometry only: the same 0.5mm grid, without an interpolation pass.
        grid = bio.resampled_geometry(bio.read_image_geometry(vol_path), target_spacing)
        spacing = np.array(grid.GetSpacing(), dtype=np.float64)
        shape = np.array(grid.GetSize(), dtype=np.int64)

        chambers = _resample_mask_onto(sitk.ReadImage(chambers_path), grid)
        coarse = _resample_mask_onto(sitk.ReadImage(coarse_path), grid).astype(bool)

        heart = chambers > 0
        if not heart.any():
            return scan_id, False, "TotalSegmentator chamber mask is empty"
        if not coarse.any():
            return scan_id, False, "coarse vessel prediction is empty"

        # Sec. III-A.1: the union of the coarse-vessel and heart bounding boxes.
        heart_lo, heart_hi = _bbox(heart)
        vessel_lo, vessel_hi = _bbox(coarse)
        margin_vox = np.ceil(margin_mm / spacing).astype(np.int64)
        roi_lo = np.maximum(np.minimum(heart_lo, vessel_lo) - margin_vox, 0)
        roi_hi = np.minimum(np.maximum(heart_hi, vessel_hi) + margin_vox, shape)
        sl = tuple(slice(int(l), int(h)) for l, h in zip(roi_lo, roi_hi))

        # Exact, not an approximation: the ROI contains the whole heart mask by
        # construction, so every surface voxel a P voxel could be nearest to is in it.
        dist, missing = _distance_fields(chambers[sl], spacing, tau_mm)

        p_local = np.argwhere(coarse[sl])
        if len(p_local) == 0:
            return scan_id, False, "coarse vessel mask has no voxels inside the ROI"
        values = dist[:, p_local[:, 0], p_local[:, 1], p_local[:, 2]].T  # (N, 5)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            roi_lo=roi_lo.astype(np.int32),
            roi_hi=roi_hi.astype(np.int32),
            shape=shape.astype(np.int32),
            # Relative to the full grid, so the dataset can index a crop taken
            # anywhere without first knowing the ROI origin.
            idx=(p_local + roi_lo).astype(np.int16),
            dist=values.astype(np.float16),
            tau_mm=np.float32(tau_mm),
        )
        note = (f"|P|={len(p_local)}, roi={tuple((roi_hi - roi_lo).tolist())}, "
                f"grid={tuple(shape.tolist())}")
        if missing:
            note += f"  [FLAG] structures absent from TotalSegmentator output: {', '.join(missing)}"
        return scan_id, True, note
    except Exception as e:
        return scan_id, False, f"{type(e).__name__}: {e}"


def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def _find_coarse(coarse_run: str, scan_id: str, extension: str) -> str | None:
    """Find a scan's coarse prediction across inference.py's split-suffixed
    directories. A scan is in exactly one split, so the first hit is unambiguous."""
    for sub in ("predictions", "predictions_train", "predictions_val"):
        path = os.path.join(coarse_run, sub, f"{scan_id}{extension}")
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to the ADE-HTL method config JSON")
    parser.add_argument("--coarse-run", required=True,
                        help="Run folder of the trained ade_htl_stage1_coarse net, containing "
                             "predictions/ and predictions_{train,val}/. Relative to results_root "
                             "in the config, or an absolute path.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                        help="Which split to build ADE assets for. Default 'all' — the HTL net "
                             "needs them for every scan it trains, validates or predicts on.")
    parser.add_argument("--tau-mm", type=float, default=DEFAULT_TAU_MM,
                        help=f"Eq. 2's normalisation distance in mm. Default {DEFAULT_TAU_MM} (the paper's value).")
    parser.add_argument("--margin-mm", type=float, default=5.0,
                        help="Margin added around the union bounding box, in mm. Default 5.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)
    p = cfg.data.params

    coarse_run = args.coarse_run
    if cfg.results_root and not os.path.isabs(coarse_run):
        coarse_run = os.path.join(cfg.results_root, coarse_run)

    out_dir = _resolve(cfg.data.data_root, p.get("ade_dir", DEFAULT_ADE_DIR))
    vol_dir = _resolve(cfg.data.data_root, cfg.data.volume_dir)
    chambers_dir = _resolve(cfg.data.data_root, p.get("chambers_dir", DEFAULT_CHAMBERS_DIR))
    chambers_suffix = p.get("chambers_suffix", DEFAULT_CHAMBERS_SUFFIX)
    vol_suffix = cfg.data.volume_suffix or cfg.data.file_extension
    target_spacing = float(cfg.preprocessing.params.get("resample", {}).get("target_spacing", 0.5))

    if args.split == "all":
        scan_ids = sorted(set(cfg.data.train_ids) | set(cfg.data.val_ids) | set(cfg.data.test_ids))
    else:
        scan_ids = {"train": cfg.data.train_ids, "val": cfg.data.val_ids,
                    "test": cfg.data.test_ids}[args.split]
    if not scan_ids:
        raise ValueError(f"No scan IDs for split '{args.split}' — check data.filelist_dir in the config.")

    print(f"[ade] {len(scan_ids)} scans  split={args.split}  tau={args.tau_mm}mm  "
          f"margin={args.margin_mm}mm  target_spacing={target_spacing}mm")
    print(f"[ade] vol_dir={vol_dir}")
    print(f"[ade] chambers_dir={chambers_dir}  suffix={chambers_suffix}")
    print(f"[ade] coarse_run={coarse_run}")
    print(f"[ade] out_dir={out_dir}")

    jobs, skipped = [], 0
    for scan_id in scan_ids:
        out_path = os.path.join(out_dir, f"{scan_id}.npz")
        if not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        jobs.append((scan_id,
                     bio.resolve_scan_path(vol_dir, scan_id, vol_suffix),
                     os.path.join(chambers_dir, f"{scan_id}{chambers_suffix}"),
                     _find_coarse(coarse_run, scan_id, cfg.data.file_extension),
                     out_path, target_spacing, args.tau_mm, args.margin_mm))

    print(f"[ade] {skipped} already cached, {len(jobs)} to process (workers={args.workers})\n")

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

    print(f"\n[ade] done in {time.perf_counter() - t0:.1f}s  ok={ok_count}  "
          f"failed={len(failed)}  skipped={skipped}")
    if failed:
        # A scan with no ADE assets cannot be trained or predicted at all; this is
        # the list to reconcile against exclude.txt.
        print(f"[ade] scans with no ADE assets ({len(failed)}): {' '.join(failed)}")
        print("[ade] re-run with --overwrite to retry after investigating the errors above.")


if __name__ == "__main__":
    main()
