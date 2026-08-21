"""Offline: ADE-HTL's key-point set and adaptive-sigma centerline heatmap.

    python -m models.ade_htl.precompute_targets -c configs/ade_htl.json --split all

Writes one <data_root>/<htl_targets_dir>/<scan_id>.npz per scan holding the key-point
indices and the sparse non-negligible part of the heatmap.

These supervise two of the three HTL branches. The third, 27-channel connectivity, is
deliberately not precomputed — flipping a volume permutes which neighbour each
channel refers to, so the dataset derives it after augmentation instead.

Key points are bifurcations and endpoints (Sec. IV-A.1), taken from the VTKs'
per-vertex `branch_points`/`end_points` flags. utils/precompute_centerline_samples.py
drops `branch_points`, so its .npz files cannot be reused and the VTK is read again.

Heatmap sigma (Eqs. 5-6) is the local vessel radius. The paper defines it as the RMS
distance from C_t to the perpendicular cross-section's boundary, then notes it "can
be approximated as the vascular radius" — so it is taken as the GT lumen's EDT at
C_t, which is that radius by definition and needs no cross-section extraction.

Everything is on the shared 0.5mm grid, matching precompute_ade.py.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk

from utils.config import BenchmarkConfig
from utils import io as bio
from utils.precompute_centerline_samples import (
    MAX_SNAP_MM, SIDES, _GTGeometry, _read_centerline, _resample_mask_onto,
)


DEFAULT_TARGETS_DIR = "ade_htl_targets_2026"

# At 3 sigma the Gaussian is already ~0.011, so dropping below this keeps
# essentially the full support.
HEATMAP_EPS = 0.01
# Gaussians are rendered out to this many sigma.
HEATMAP_EXTENT_SIGMA = 3.0


def _mm_to_voxel(pts_mm: np.ndarray, ref_img) -> np.ndarray:
    """LPS mm -> continuous voxel coordinates."""
    origin = np.array(ref_img.GetOrigin())
    spacing = np.array(ref_img.GetSpacing())
    direction = np.array(ref_img.GetDirection()).reshape(3, 3)
    rel = np.asarray(pts_mm, dtype=np.float64) - origin[None, :]
    return (np.linalg.inv(direction) @ rel.T).T / spacing[None, :]


def _render_heatmap(centers_vox: np.ndarray, sigma_mm: np.ndarray,
                    shape: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Eq. 6's Gaussians at each centerline voxel, combined by max.

    Centres snap to the nearest voxel first, so the peak is exactly 1.0; evaluating
    at continuous coordinates would put it between voxels and cap the achievable
    target below 1. Max rather than sum keeps the target in [0, 1] where two branches
    run close together.
    """
    heat = np.zeros(tuple(shape), dtype=np.float32)
    centers = np.rint(centers_vox).astype(np.int64)

    for centre, sigma in zip(centers, sigma_mm):
        half = np.ceil(HEATMAP_EXTENT_SIGMA * sigma / spacing).astype(np.int64)
        lo = np.maximum(centre - half, 0)
        hi = np.minimum(centre + half + 1, shape)
        if np.any(hi <= lo):
            continue
        grids = np.meshgrid(*[(np.arange(l, h) - c) * s
                              for l, h, c, s in zip(lo, hi, centre, spacing)],
                            indexing="ij")
        d2 = grids[0] ** 2 + grids[1] ** 2 + grids[2] ** 2
        g = np.exp(-d2 / (2.0 * sigma ** 2))
        sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        np.maximum(heat[sl], g, out=heat[sl])
    return heat


def _collect_vertices(centerline_paths: list, grid, geom: _GTGeometry) -> tuple:
    """Every side's centerline vertices and key points on the voxel grid. Vertices
    whose nearest lumen voxel is beyond MAX_SNAP_MM are dropped: the centerline has
    strayed outside the annotated mask, so the radius would come from elsewhere."""
    centers, sigmas, keypoints, notes = [], [], [], []

    for side, path in centerline_paths:
        if not os.path.exists(path):
            notes.append(f"{side}: missing")
            continue
        cl = _read_centerline(path)
        points = cl["points"]
        if len(points) == 0:
            notes.append(f"{side}: no points")
            continue

        snap_mm, near_idx, _ = geom.nearest(points)
        keep = snap_mm <= MAX_SNAP_MM
        if not keep.any():
            notes.append(f"{side}: all {len(points)} vertices fell outside the GT mask")
            continue

        radius = geom.radius_at(near_idx[keep])
        centers.append(_mm_to_voxel(points[keep], grid))
        # Floor at one voxel: a sub-voxel radius would be a target the regression
        # head cannot represent.
        sigmas.append(np.maximum(radius.astype(np.float64), float(grid.GetSpacing()[0])))

        # Already marked per-vertex, so no re-derivation from a skeleton is needed.
        flags = np.zeros(len(points), dtype=bool)
        for name in ("branch_points", "end_points"):
            arr = cl[name]
            if arr is not None:
                flags |= arr != 0
        kp_keep = flags & keep
        if kp_keep.any():
            keypoints.append(_mm_to_voxel(points[kp_keep], grid))

        n_kp = int(kp_keep.sum())
        notes.append(f"{side}: {int(keep.sum())}/{len(points)} vertices, {n_kp} key points")

    if not centers:
        return None, None, None, notes

    centers_vox = np.concatenate(centers, axis=0)
    sigma_mm = np.concatenate(sigmas, axis=0)
    kp_vox = np.concatenate(keypoints, axis=0) if keypoints else np.zeros((0, 3))
    return centers_vox, sigma_mm, kp_vox, notes


def _process_scan(args: tuple) -> tuple:
    (scan_id, vol_path, mask_path, centerline_paths, out_path,
     target_spacing, binarise_kwargs) = args
    try:
        grid = bio.resampled_geometry(bio.read_image_geometry(vol_path), target_spacing)
        spacing = np.array(grid.GetSpacing(), dtype=np.float64)
        shape = np.array(grid.GetSize(), dtype=np.int64)

        gt_raw = _resample_mask_onto(sitk.ReadImage(mask_path), grid)
        lumen = bio.binarise_lumen(gt_raw, **binarise_kwargs).astype(bool)
        if not lumen.any():
            return scan_id, False, "empty GT lumen mask"

        geom = _GTGeometry(lumen, grid)
        centers_vox, sigma_mm, kp_vox, notes = _collect_vertices(centerline_paths, grid, geom)
        if centers_vox is None:
            return scan_id, False, "no usable centerline vertices; " + "; ".join(notes)

        heat = _render_heatmap(centers_vox, sigma_mm, shape, spacing)
        cl_idx = np.argwhere(heat > HEATMAP_EPS)
        if len(cl_idx) == 0:
            return scan_id, False, "centerline heatmap is empty"
        cl_val = heat[cl_idx[:, 0], cl_idx[:, 1], cl_idx[:, 2]]

        kp_idx = np.rint(kp_vox).astype(np.int64)
        if len(kp_idx):
            in_bounds = np.all((kp_idx >= 0) & (kp_idx < shape[None, :]), axis=1)
            kp_idx = np.unique(kp_idx[in_bounds], axis=0)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            shape=shape.astype(np.int32),
            kp_idx=kp_idx.astype(np.int16),
            cl_idx=cl_idx.astype(np.int16),
            cl_val=cl_val.astype(np.float16),
        )
        return scan_id, True, (f"{len(kp_idx)} key points, {len(cl_idx)} heatmap voxels, "
                               f"sigma={sigma_mm.min():.2f}-{sigma_mm.max():.2f}mm; "
                               + "; ".join(notes))
    except Exception as e:
        return scan_id, False, f"{type(e).__name__}: {e}"


def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to the ADE-HTL method config JSON")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                        help="Which split to build targets for. Default 'all' — the HTL net is "
                             "supervised by these on train and val alike.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)
    p = cfg.data.params

    out_dir = _resolve(cfg.data.data_root, p.get("htl_targets_dir", DEFAULT_TARGETS_DIR))
    vol_dir = _resolve(cfg.data.data_root, cfg.data.volume_dir)
    mask_dir = _resolve(cfg.data.data_root, cfg.data.gt_mask_dir)
    cl_dir = _resolve(cfg.data.data_root, p.get("centerlines_dir", "centerlines"))
    vol_suffix = cfg.data.volume_suffix or cfg.data.file_extension
    mask_suffix = cfg.data.mask_suffix or cfg.data.file_extension
    target_spacing = float(cfg.preprocessing.params.get("resample", {}).get("target_spacing", 0.5))

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

    print(f"[htl-targets] {len(scan_ids)} scans  split={args.split}  target_spacing={target_spacing}mm")
    print(f"[htl-targets] mask_dir={mask_dir}")
    print(f"[htl-targets] centerlines_dir={cl_dir}")
    print(f"[htl-targets] out_dir={out_dir}")

    jobs, skipped = [], 0
    for scan_id in scan_ids:
        out_path = os.path.join(out_dir, f"{scan_id}.npz")
        if not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        centerline_paths = [
            (side, os.path.join(cl_dir, f"{scan_id}.coronary_{side}_centerline.vtk"))
            for side in SIDES
        ]
        jobs.append((scan_id,
                     bio.resolve_scan_path(vol_dir, scan_id, vol_suffix),
                     bio.resolve_scan_path(mask_dir, scan_id, mask_suffix),
                     centerline_paths, out_path, target_spacing, binarise_kwargs))

    print(f"[htl-targets] {skipped} already cached, {len(jobs)} to process (workers={args.workers})\n")

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

    print(f"\n[htl-targets] done in {time.perf_counter() - t0:.1f}s  ok={ok_count}  "
          f"failed={len(failed)}  skipped={skipped}")
    if failed:
        print(f"[htl-targets] scans with no HTL targets ({len(failed)}): {' '.join(failed)}")
        print("[htl-targets] re-run with --overwrite to retry after investigating the errors above.")


if __name__ == "__main__":
    main()
