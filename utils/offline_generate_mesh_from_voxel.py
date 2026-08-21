"""Offline: generate a GT lumen surface mesh (verts + faces) from the GT voxel
mask, via marching cubes + non-shrinking (Taubin-style) smoothing. Writes one
`<scan_id>.coronary_surface.vtk` per scan.

Pipeline per scan:
  1. Load the GT lumen mask in its *original* (unresampled) voxel grid --
     native scan spacing, not `preprocessing.params.resample.target_spacing`.
  2. Binarise to a single foreground label (`bio.binarise_lumen`).
  3. `vtkMarchingCubes` at isovalue 0.5 -> one watertight, manifold surface for
     the whole coronary tree in a single pass. No centerline, no per-branch
     construction, no branch merging -- the mask is already one connected
     volume, so there's nothing to stitch.
  4. `vtkWindowedSincPolyDataFilter` (VTK's standard non-shrinking low-pass
     mesh filter, the practical equivalent of Taubin 1995) to remove
     voxel-staircase artifacts without the shrinkage plain Laplacian smoothing
     causes.

An earlier revision built a per-branch tube mesh from the GT centerline tree
instead (spline fit + rotation-minimizing frames + ray casting against the mask,
one tube per disjoint branch segment, merged via
`vtkBooleanOperationPolyDataFilter`). That was dropped because the boolean
filter is fragile on this geometry -- it segfaults outright on some
non-manifold/self-intersecting input (killing the worker with no Python
traceback) and legitimately returns an empty result for sibling branches that
only touch at a point rather than overlapping in volume. Marching cubes on the
mask is both simpler and more robust.

Coordinate spaces:
  - The mesh's point geometry is physical mm (LPS), using the *original*
    (unresampled) mask's own sitk affine -- "original volume space". This is
    resample-spacing-agnostic: regenerating with a different
    `preprocessing.params.resample.target_spacing` doesn't require rebuilding
    this file.
  - It also carries a point-data array, `voxel_coords_resampled`: the same
    vertices' coordinates in the *resampled* grid's voxel-index space, computed
    via `bio.resampled_geometry` (which reproduces
    `preprocessing.steps.Resample`'s exact origin/spacing/direction/size formula
    without paying for the full image interpolation, since only the resulting
    affine is needed here). A consumer reading this array needs no coordinate
    conversion of its own, and the array is always regenerated correctly
    whenever this script reruns with the config's current target_spacing.

Reads the GT voxel mask (data.gt_mask_dir) plus the raw volume (data.volume_dir)
-- the volume is only used for its sitk header, to compute `voxel_coords_resampled`
in the same grid geometry `preprocessing.steps.Resample` produces (which derives
that grid from the volume's header, not the mask's). The GT centerline tree is
not read by this script at all. Output goes to `<data_root>/surfaces/`.

Takes no config: the dataset dirs, filename suffixes, filelist location and
target_spacing it needs are all in the shared base configs/pipeline.json, identical
across every method config, and data_root comes from $ImageCAS_X_data_path.

Usage:
    python -m utils.offline_generate_mesh_from_voxel
    python -m utils.offline_generate_mesh_from_voxel --split train --workers 16 --overwrite
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from utils.config import BenchmarkConfig
from utils import io as bio


def _silence_vtk_warnings():
    """Suppress VTK's C++-side warning/error stream. Must be called inside each
    worker process (ProcessPoolExecutor spawns fresh interpreters)."""
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()


def _dbg(scan_id: str, msg: str):
    print(f"[{scan_id}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Coordinate conversion (voxel-index <-> physical mm)
# ---------------------------------------------------------------------------

def _voxel_index_to_physical(sitk_img, points_vox: np.ndarray) -> np.ndarray:
    """Voxel index -> physical mm (LPS): phys = origin + direction @ (index * spacing).

    Expects/returns the same [x,y,z] index order this codebase uses everywhere
    (`sitk.GetArrayFromImage(img).transpose(2,1,0)`).
    """
    origin = np.array(sitk_img.GetOrigin())
    spacing = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    return origin[None, :] + (direction @ (points_vox * spacing[None, :]).T).T


def _physical_to_voxel_index(sitk_img, points_phys: np.ndarray) -> np.ndarray:
    """Vectorised inverse of `_voxel_index_to_physical`. Returns continuous
    (not rounded) voxel indices."""
    origin = np.array(sitk_img.GetOrigin())
    spacing = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    inv_direction = np.linalg.inv(direction)
    rel = points_phys - origin[None, :]
    return (inv_direction @ rel.T).T / spacing[None, :]


# ---------------------------------------------------------------------------
# Mask -> surface (marching cubes + Taubin-style smoothing)
# ---------------------------------------------------------------------------

def _mask_to_vtk_image(mask: np.ndarray):
    """Wrap a binary [x,y,z] numpy mask as a vtkImageData in raw voxel-index
    space (unit spacing, zero origin). Marching cubes runs purely on array
    indices here; real physical geometry is applied afterwards via
    `_voxel_index_to_physical`, since vtkImageData can't represent a
    non-axis-aligned direction matrix.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    img = vtk.vtkImageData()
    img.SetDimensions(*mask.shape)
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    flat = np.asarray(mask, dtype=np.float32).ravel(order="F")
    vtk_arr = numpy_to_vtk(flat, deep=True)
    img.GetPointData().SetScalars(vtk_arr)
    return img


def _marching_cubes_and_smooth(mask: np.ndarray, isovalue: float,
                                smoothing_iterations: int, passband: float) -> tuple:
    """Binary mask -> single watertight surface (vtkMarchingCubes) -> non-shrinking
    smoothing (vtkWindowedSincPolyDataFilter, VTK's standard equivalent of Taubin
    1995's low-pass mesh filter). Output verts are in the mask's own raw
    voxel-index space (see `_mask_to_vtk_image`); the caller converts to physical
    mm afterwards.
    """
    import vtk

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(_mask_to_vtk_image(mask))
    mc.SetValue(0, isovalue)
    mc.ComputeNormalsOff()
    mc.ComputeGradientsOff()
    mc.ComputeScalarsOff()
    mc.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(mc.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.SetPassBand(passband)
    smoother.BoundarySmoothingOn()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.Update()

    return _vtk_polydata_to_numpy(smoother.GetOutput())


def _numpy_to_vtk_polydata(verts: np.ndarray, faces: np.ndarray):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64)))
    cells = vtk.vtkCellArray()
    for f in faces:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(int(f[0]))
        cells.InsertCellPoint(int(f[1]))
        cells.InsertCellPoint(int(f[2]))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def _vtk_polydata_to_numpy(poly) -> tuple:
    from vtk.util.numpy_support import vtk_to_numpy

    verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    # Connectivity is a flat vertex-id stream with no per-cell count prefix (the old
    # GetPolys().GetData() layout, deprecated in VTK 9.6). Marching cubes emits only
    # triangles, so it reshapes directly.
    conn = vtk_to_numpy(poly.GetPolys().GetConnectivityArray())
    faces = conn.reshape(-1, 3).astype(np.int64)
    return verts, faces


def _write_vtk_mesh(verts: np.ndarray, faces: np.ndarray, path: str, point_data: dict = None):
    """point_data: optional {name: (V,3)-shaped array} written as extra vtkPolyData
    point-data arrays alongside the geometry -- see module docstring's note on the
    `voxel_coords_resampled` array."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    poly = _numpy_to_vtk_polydata(verts, faces)
    if point_data:
        for name, arr in point_data.items():
            vtk_arr = numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float64), deep=True)
            vtk_arr.SetName(name)
            poly.GetPointData().AddArray(vtk_arr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.Write()


# ---------------------------------------------------------------------------
# Per-scan orchestration
# ---------------------------------------------------------------------------

def _process_scan(args: tuple) -> tuple:
    (scan_id, vol_path, mask_path, out_mesh_path,
     target_spacing, isovalue, smoothing_iterations, passband) = args
    _silence_vtk_warnings()
    try:
        _, sitk_img = bio.load_volume(vol_path)
        mask, sitk_mask = bio.load_mask(mask_path)
        _dbg(scan_id, f"loaded mask: shape={mask.shape} dtype={mask.dtype} "
                       f"spacing={sitk_mask.GetSpacing()} origin={sitk_mask.GetOrigin()} "
                       f"unique_labels={np.unique(mask)}")

        lumen_mask = bio.binarise_lumen(mask)
        n_true = int(lumen_mask.sum())
        _dbg(scan_id, f"lumen_mask: shape={lumen_mask.shape} n_true_voxels={n_true}")
        if n_true == 0:
            return scan_id, False, "empty lumen mask (no foreground voxels)"

        verts_vox_orig, faces = _marching_cubes_and_smooth(
            lumen_mask, isovalue, smoothing_iterations, passband)
        _dbg(scan_id, f"marching_cubes+smoothed: verts={verts_vox_orig.shape} faces={faces.shape}")

        # Main mesh geometry: original scan's physical mm (LPS) -- same coordinate
        # system as the scan itself and every other pipeline VTK.
        verts_phys = _voxel_index_to_physical(sitk_mask, verts_vox_orig)

        # Extra point-data array: same vertices in the *resampled* grid's voxel-index
        # space (see module docstring) -- so a consumer can read them directly
        # without recomputing the conversion, while the written mesh itself stays
        # resample-spacing-agnostic.
        #
        # Geometry must be derived from the volume's header (sitk_img), not the
        # mask's, to match preprocessing.steps.Resample: that step resamples the
        # mask onto the *volume's* resampled grid (SetReferenceImage(sitk_img_r))
        # rather than an independently-computed one, specifically so it doesn't
        # depend on the mask's header matching the volume's. Using sitk_mask here
        # instead produces silently misaligned voxel_coords_resampled whenever a
        # scan's volume/mask headers aren't identical.
        resampled_geom = bio.resampled_geometry(sitk_img, target_spacing)
        verts_vox_resampled = _physical_to_voxel_index(resampled_geom, verts_phys)
        _dbg(scan_id, f"resampled voxel-index coords (target_spacing={target_spacing}mm): "
                       f"range_min={verts_vox_resampled.min(axis=0)} "
                       f"range_max={verts_vox_resampled.max(axis=0)}")

        _write_vtk_mesh(verts_phys, faces, out_mesh_path,
                         point_data={"voxel_coords_resampled": verts_vox_resampled})

        return scan_id, True, f"verts={len(verts_phys)} faces={len(faces)}"
    except Exception as e:
        return scan_id, False, str(e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def _pipeline_config() -> BenchmarkConfig:
    """The shared base config (configs/pipeline.json) — dataset dirs, filename
    suffixes, filelist location and target_spacing. Everything this script needs
    already lives there and is identical across methods, so there is no method
    config to pass in. `method_name` is injected only because BenchmarkConfig
    requires it; nothing here reads it."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "configs", "pipeline.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw["method_name"] = "offline_generate_mesh_from_voxel"
    return BenchmarkConfig(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--isovalue", type=float, default=0.5,
                         help="Marching cubes contour value on the binary [0,1] lumen mask.")
    parser.add_argument("--smoothing-iterations", type=int, default=50,
                         help="vtkWindowedSincPolyDataFilter iteration count.")
    parser.add_argument("--passband", type=float, default=0.05,
                         help="vtkWindowedSincPolyDataFilter pass band (0,2]; smaller = more smoothing.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = _pipeline_config()

    target_spacing = float(cfg.preprocessing.params.get("resample", {}).get("target_spacing", 0.5))
    out_dir = _resolve(cfg.data.data_root, "surfaces")

    if args.split == "all":
        scan_ids = sorted(set(cfg.data.train_ids) | set(cfg.data.val_ids) | set(cfg.data.test_ids))
    else:
        scan_ids = {"train": cfg.data.train_ids, "val": cfg.data.val_ids, "test": cfg.data.test_ids}[args.split]

    if not scan_ids:
        raise ValueError(
            f"No scan IDs found — expected train/val/test .txt files in "
            f"{_resolve(cfg.data.data_root, cfg.data.filelist_dir)}."
        )

    vol_suffix = cfg.data.volume_suffix or cfg.data.file_extension
    mask_suffix = cfg.data.mask_suffix or cfg.data.file_extension
    vol_dir = _resolve(cfg.data.data_root, cfg.data.volume_dir)
    mask_dir = _resolve(cfg.data.data_root, cfg.data.gt_mask_dir)

    print(f"[gen_mesh] {len(scan_ids)} scans  target_spacing={target_spacing}mm  "
          f"isovalue={args.isovalue}  smoothing_iterations={args.smoothing_iterations}  "
          f"passband={args.passband}")
    print(f"[gen_mesh] vol_dir={vol_dir}")
    print(f"[gen_mesh] mask_dir={mask_dir}")
    print(f"[gen_mesh] out_dir={out_dir}")

    jobs = []
    skipped = 0
    for scan_id in scan_ids:
        out_mesh_path = os.path.join(out_dir, f"{scan_id}.coronary_surface.vtk")
        if not args.overwrite and os.path.exists(out_mesh_path):
            skipped += 1
            continue
        vol_path = bio.resolve_scan_path(vol_dir, scan_id, vol_suffix)
        mask_path = bio.resolve_scan_path(mask_dir, scan_id, mask_suffix)
        jobs.append((scan_id, vol_path, mask_path, out_mesh_path,
                     target_spacing, args.isovalue, args.smoothing_iterations, args.passband))


    print(f"[gen_mesh] {skipped} already cached, {len(jobs)} to process (workers={args.workers})\n")

    t0 = time.perf_counter()
    ok_count, fail_count = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
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
    print(f"\n[gen_mesh] done in {total:.1f}s  ok={ok_count}  failed={fail_count}  skipped={skipped}")
    if fail_count:
        print("[gen_mesh] re-run with --overwrite to retry failed scans after investigating the errors above.")


if __name__ == "__main__":
    main()
