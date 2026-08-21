"""Offline: sample the GT centerline tree and record, per sample vertex, the
covariates the stratified results are grouped by — coronary segment, lumen HU and
geodesic distance from the ostium.

    python -m utils.precompute_centerline_samples -c configs/<method>.json --split test

Writes one <out_dir>/<scan_id>.npz per scan. Everything here depends on the ground
truth alone, so one run serves every method; evaluate.py adds the only
prediction-dependent column, a local Dice around each vertex.

The VTKs' line cells are inter-branchpoint segments in inconsistent orientation, so
cell ordering says nothing about which end is proximal — but they index a shared
point array, so a point ID is already a graph node. Traversal is therefore: build
adjacency from the cells, then breadth-first outward from the vertices the file's own
`start_points` array marks. Every vertex gets a unique hop count and arc length,
the centerline being a tree.

Sampling every `--step-vertices` vertices BY HOP COUNT means a shared proximal trunk
is sampled once no matter how many tips lie beyond it, and stays phase-aligned across
a bifurcation. A side may hold more than one tree — a left system with no left main
has separate LAD and LCx ostia — so BFS runs from all roots at once and each vertex
measures from whichever ostium reaches it.

Points are stored in physical mm (LPS) of the original scan, so nothing here depends
on the resample target spacing.
"""
import argparse
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk

from utils.config import BenchmarkConfig
from utils import io as bio


DEFAULT_OUT_DIR = "centerline_samples"
# Beyond this, the centerline has strayed outside the annotated mask, so diameter
# and HU would both be read from an unrelated place.
MAX_SNAP_MM = 3.0
SIDES = ("left", "right")


def _read_centerline(path: str) -> dict:
    """Points, line-cell connectivity and point-data arrays. Local to this module
    because it needs all three together, plus the string array `segment_name`, which
    utils/io.py's loaders each miss some of. Missing arrays come back as None."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise ImportError("vtk is required to load centerline VTK files: pip install vtk")

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    poly = reader.GetOutput()

    n_pts = poly.GetNumberOfPoints()
    if n_pts == 0:
        return {"points": np.zeros((0, 3)), "cells": []}

    out = {"points": vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)}

    pd = poly.GetPointData()
    for name in ("segment_label", "start_points", "end_points", "branch_points"):
        arr = pd.GetAbstractArray(name)
        out[name] = vtk_to_numpy(arr).astype(np.int64) if arr is not None else None
    sn = pd.GetAbstractArray("segment_name")
    out["segment_name"] = (np.array([sn.GetValue(i) for i in range(sn.GetNumberOfTuples())])
                           if sn is not None else None)

    cells = []
    lines = poly.GetLines()
    if lines is not None and lines.GetNumberOfCells():
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        while lines.GetNextCell(id_list):
            ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
            if len(ids) > 1:
                cells.append(ids)
    out["cells"] = cells
    return out


def _adjacency(cells: list, n_points: int) -> list:
    """Undirected adjacency over point IDs. Cells index one shared point array, so a
    point ID IS a node and two cells meeting at a bifurcation join automatically;
    discarding direction is what makes traversal immune to their orientation."""
    adj = [set() for _ in range(n_points)]
    for ids in cells:
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def _traverse(points: np.ndarray, adj: list, roots: list) -> tuple:
    """Breadth-first from every ostium at once, -1/inf where none reaches. `depth`
    counts hops and is what sampling steps along; `dist_mm` is true arc length and is
    the reported covariate."""
    n = len(points)
    depth = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf, dtype=np.float64)
    queue = deque()
    for r in roots:
        depth[r] = 0
        dist[r] = 0.0
        queue.append(r)
    while queue:
        v = queue.popleft()
        for w in adj[v]:
            if depth[w] < 0:
                depth[w] = depth[v] + 1
                dist[w] = dist[v] + float(np.linalg.norm(points[w] - points[v]))
                queue.append(w)
    return depth, dist


def _sample_vertices(depth: np.ndarray, end_points: np.ndarray, step_vertices: int) -> np.ndarray:
    """Vertices every `step_vertices` hops from the ostium, plus every end point, so
    each branch's most distal vertex is represented even when it misses a step
    boundary."""
    reached = depth >= 0
    on_step = reached & (depth % step_vertices == 0)
    if end_points is not None:
        on_step |= reached & (end_points != 0)
    return np.flatnonzero(on_step)


def _voxel_to_mm(idx: np.ndarray, ref_img) -> np.ndarray:
    """Voxel indices -> LPS mm."""
    origin = np.array(ref_img.GetOrigin())
    spacing = np.array(ref_img.GetSpacing())
    direction = np.array(ref_img.GetDirection()).reshape(3, 3)
    return origin + (direction @ (np.asarray(idx, dtype=np.float64) * spacing).T).T


def _resample_mask_onto(mask_img: sitk.Image, ref_img: sitk.Image) -> np.ndarray:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref_img)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(r.Execute(mask_img)).transpose(2, 1, 0).astype(np.uint8)


def _radius_field(lumen: np.ndarray, spacing: tuple) -> tuple:
    """EDT inside the lumen — the local radius in mm. Computed on the lumen's bbox
    padded by a voxel, so the border is background either way and the result is
    identical to the full-volume EDT, whose cost scales with array size rather than
    the under-1% the tree occupies. Returns the field and the crop's origin."""
    from scipy.ndimage import distance_transform_edt

    coords = np.argwhere(lumen)
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, np.array(lumen.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    return distance_transform_edt(lumen[sl], sampling=spacing), lo


class _GTGeometry:
    """The EDT and lumen KD-tree sample lookups need. Built once per scan and shared
    by both sides, neither depending on which centerline file is processed."""

    def __init__(self, lumen: np.ndarray, ref_img):
        from scipy.spatial import cKDTree

        self.spacing = np.array(ref_img.GetSpacing())
        self.radius_crop, self.lo = _radius_field(lumen, tuple(self.spacing))
        self.lumen_idx = np.argwhere(lumen)
        self.tree = cKDTree(_voxel_to_mm(self.lumen_idx, ref_img))

    def nearest(self, pts_mm: np.ndarray) -> tuple:
        """(snap distance, nearest lumen voxel's indices, its tree index)."""
        snap_mm, j = self.tree.query(np.atleast_2d(pts_mm))
        return snap_mm, self.lumen_idx[j], j

    def radius_at(self, near_idx: np.ndarray) -> np.ndarray:
        ci = near_idx - self.lo
        return self.radius_crop[ci[:, 0], ci[:, 1], ci[:, 2]].astype(np.float32)


def _lookup_covariates(pts_mm: np.ndarray, volume: np.ndarray, geom: _GTGeometry) -> dict:
    """Local lumen diameter and attenuation at each sample vertex.

    Lookups go through the nearest LUMEN voxel, not the one the vertex lands in:
    vertices are continuous coordinates that can fall a fraction of a voxel outside
    the mask, where calibre is undefined. `snap_mm` records how far each moved, so
    ones that strayed far can be dropped.

    `diameter_mm` is twice the EDT, doubled here and only here since the EDT stays a
    radius for the HU ball. It is the true local calibre except within about one
    radius of a mask end — at the ostium, cut flat against the aorta, and at each
    distal tip, the EDT decays to the distance to that cut face instead. That affects
    at most the first and last sample of a branch and is left uncorrected.

    HU averages the LUMEN VOXELS inside a ball of 0.7x the local radius. A fixed-size
    ball would sample mostly myocardium in a 1mm distal vessel, and including
    non-lumen voxels would make measured HU fall with calibre through partial-volume
    dilution alone — manufacturing exactly the confound this table exists to separate.
    """
    snap_mm, near_idx, nearest = geom.nearest(pts_mm)
    radius = geom.radius_at(near_idx)

    lumen_hu = volume[geom.lumen_idx[:, 0], geom.lumen_idx[:, 1],
                      geom.lumen_idx[:, 2]].astype(np.float64)
    hu = np.empty(len(pts_mm), dtype=np.float32)
    min_r = float(geom.spacing.min())
    for i, p in enumerate(pts_mm):
        ball = geom.tree.query_ball_point(p, max(0.7 * float(radius[i]), min_r))
        hu[i] = np.mean(lumen_hu[ball]) if ball else lumen_hu[nearest[i]]

    return {"diameter_mm": (2.0 * radius).astype(np.float32),
            "hu": hu, "snap_mm": snap_mm.astype(np.float32)}


def _process_side(path: str, step_vertices: int) -> tuple:
    """Sample one side. Returns (columns or None, note, flags)."""
    flags = []
    if not os.path.exists(path):
        return None, "missing", flags

    cl = _read_centerline(path)
    points, cells = cl["points"], cl["cells"]
    if len(points) == 0 or not cells:
        return None, "no points/line cells", flags

    if cl["start_points"] is None:
        flags.append("no 'start_points' array in the file — cannot locate the ostium, "
                     "side skipped")
        return None, "no start_points array", flags
    roots = np.flatnonzero(cl["start_points"]).tolist()
    if not roots:
        flags.append("'start_points' array marks no vertex — cannot locate the ostium, "
                     "side skipped")
        return None, "no marked start vertex", flags

    adj = _adjacency(cells, len(points))
    depth, dist = _traverse(points, adj, roots)

    n_unreachable = int((depth < 0).sum())
    if n_unreachable:
        flags.append(f"{n_unreachable}/{len(points)} vertices unreachable from any marked "
                     f"start point — they are excluded from sampling")

    sampled = _sample_vertices(depth, cl["end_points"], step_vertices)
    if len(sampled) == 0:
        return None, "no vertices sampled", flags

    cols = {
        "xyz_mm": points[sampled],
        "dist_mm": dist[sampled].astype(np.float32),
        "depth": depth[sampled].astype(np.int32),
    }
    # `branch_points` is deliberately dropped: bifurcations are only sampled when one
    # lands on a step boundary, so the flag would be zero almost everywhere and
    # misleading where it was not. Force-sampling them would make it a real covariate.
    for name, key, dtype in (("segment_label", "seg", np.uint8),
                             ("end_points", "is_end_point", np.uint8)):
        arr = cl[name]
        cols[key] = (arr[sampled].astype(dtype) if arr is not None
                     else np.zeros(len(sampled), dtype=dtype))
    cols["segment_name"] = (cl["segment_name"][sampled] if cl["segment_name"] is not None
                            else np.full(len(sampled), "", dtype="<U1"))

    note = (f"{len(sampled)} pts, trees={len(roots)}, vertices={len(points)}, "
            f"max_dist={dist[np.isfinite(dist)].max():.0f}mm")
    return cols, note, flags


def _process_scan(args: tuple) -> tuple:
    (scan_id, vol_path, mask_path, centerline_paths, out_path,
     step_vertices, binarise_kwargs) = args
    try:
        volume, ref_img = bio.load_volume(vol_path)
        gt_raw = _resample_mask_onto(sitk.ReadImage(mask_path), ref_img)
        lumen = bio.binarise_lumen(gt_raw, **binarise_kwargs).astype(bool)
        if not lumen.any():
            return scan_id, False, "empty GT lumen mask"

        geom = _GTGeometry(lumen, ref_img)

        per_side, notes = [], []
        for side_code, (side, path) in enumerate(centerline_paths):
            cols, note, flags = _process_side(path, step_vertices)
            notes.append(f"{side}: {note}" + "".join(f" [FLAG] {side}: {m}" for m in flags))
            if cols is None:
                continue
            cols["side"] = np.full(len(cols["dist_mm"]), side_code, dtype=np.uint8)
            per_side.append(cols)

        if not per_side:
            return scan_id, False, "no centerline samples; " + "; ".join(notes)

        cols = {k: np.concatenate([s[k] for s in per_side]) for k in per_side[0]}
        pts_mm = cols["xyz_mm"].astype(np.float64)

        cols.update(_lookup_covariates(pts_mm, volume, geom))

        keep = cols["snap_mm"] <= MAX_SNAP_MM
        n_dropped = int((~keep).sum())
        if not keep.any():
            return scan_id, False, "every sample vertex fell outside the GT mask"
        cols = {k: v[keep] for k, v in cols.items()}

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            xyz_mm=cols["xyz_mm"].astype(np.float32),
            step_vertices=np.int32(step_vertices),
            **{k: v for k, v in cols.items() if k != "xyz_mm"},
        )
        return scan_id, True, (f"{len(cols['dist_mm'])} samples, dropped={n_dropped}; "
                               + "; ".join(notes))
    except Exception as e:
        return scan_id, False, f"{type(e).__name__}: {e}"


def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to any method config JSON")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test",
                        help="Which split to sample. Default: test (the only split evaluate.py scores).")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir, relative to data_root unless absolute. Defaults to the "
                             f"config's data.params.centerline_samples_dir, or '{DEFAULT_OUT_DIR}'.")
    parser.add_argument("--step-vertices", type=int, default=5,
                        help="Sample every Nth centerline vertex, counted in hops outward from the "
                             "ostium. Default 5 (~2.2mm at this dataset's ~0.45mm vertex spacing). "
                             "End points are always sampled regardless.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)

    out_dir = _resolve(cfg.data.data_root,
                       args.out_dir or cfg.data.params.get("centerline_samples_dir") or DEFAULT_OUT_DIR)
    vol_dir = _resolve(cfg.data.data_root, cfg.data.volume_dir)
    mask_dir = _resolve(cfg.data.data_root, cfg.data.gt_mask_dir)
    cl_dir = _resolve(cfg.data.data_root, cfg.data.params.get("centerlines_dir", "centerlines"))
    vol_suffix = cfg.data.volume_suffix or cfg.data.file_extension
    mask_suffix = cfg.data.mask_suffix or cfg.data.file_extension

    binarise_kwargs = {
        "background_label": cfg.data.params.get("background_label", bio.LUMEN_BACKGROUND_LABEL),
    }

    if args.split == "all":
        scan_ids = sorted(set(cfg.data.train_ids) | set(cfg.data.val_ids) | set(cfg.data.test_ids))
    else:
        scan_ids = {"train": cfg.data.train_ids, "val": cfg.data.val_ids,
                    "test": cfg.data.test_ids}[args.split]
    if not scan_ids:
        raise ValueError(f"No scan IDs for split '{args.split}' — check data.filelist_dir in the config.")

    print(f"[samples] {len(scan_ids)} scans  split={args.split}  step_vertices={args.step_vertices}")
    print(f"[samples] vol_dir={vol_dir}")
    print(f"[samples] mask_dir={mask_dir}")
    print(f"[samples] centerlines_dir={cl_dir}")
    print(f"[samples] out_dir={out_dir}")

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
                     centerline_paths, out_path,
                     args.step_vertices, binarise_kwargs))

    print(f"[samples] {skipped} already cached, {len(jobs)} to process (workers={args.workers})\n")

    t0 = time.perf_counter()
    ok_count = fail_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_scan, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            scan_id, ok, message = future.result()
            print(f"  [{i}/{len(jobs)}] {scan_id}: {'ok' if ok else 'FAILED'}  {message}")
            ok_count += ok
            fail_count += not ok

    print(f"\n[samples] done in {time.perf_counter() - t0:.1f}s  ok={ok_count}  "
          f"failed={fail_count}  skipped={skipped}")
    if fail_count:
        print("[samples] re-run with --overwrite to retry failed scans after investigating the errors above.")


if __name__ == "__main__":
    main()
