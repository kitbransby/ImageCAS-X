"""Score a method's predictions against GT.

    python -m evaluate -c configs/<method>.json -r <run_dir>

Scores <run_dir>/predictions/<scan_id><ext> against GT for every test scan that has
one; scans without a prediction are skipped and counted, and the scan count actually
evaluated is reported at the end and stored under "coverage" in the results JSON.
This never loads a model, which is what lets externally-produced predictions be
scored here too.

With --multi_class it instead scores <run_dir>/predictions/<scan_id>.multi_class.nii.gz
against the GT labels for each of the 14 coronary segments separately, reporting every
segmentation metric plus how often each segment is present. That mode skips the
descriptor stratification and the centerline/point breakdowns.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from utils.config import BenchmarkConfig
from utils.seeding import seed_everything
from utils.metrics import (evaluate as compute_metrics, compute_centerline_metrics,
                                     compute_centerline_point_metrics,
                                     compute_local_dice, CENTERLINE_METRICS,
                                     METRIC_REGISTRY)
from utils import io as bio


CENTERLINE_SIDES = ("left", "right")


def _centerline_dir(cfg: BenchmarkConfig) -> str:
    rel = cfg.data.params.get("centerlines_dir", "centerlines")
    return rel if os.path.isabs(rel) else os.path.join(cfg.data.data_root, rel)


def _load_gt_centerline_pts(cfg: BenchmarkConfig, scan_id: str) -> np.ndarray:
    """Left + right GT centerline VTKs combined into one (N, 3) LPS mm array."""
    cdir = _centerline_dir(cfg)
    all_pts = []
    for side in CENTERLINE_SIDES:
        path = os.path.join(cdir, f"{scan_id}.coronary_{side}_centerline.vtk")
        if not os.path.exists(path):
            print(f"    [warn] GT centerline not found: {path}")
            continue
        try:
            pts, _ = bio.load_vtk_centerline(path)
            if len(pts) > 0:
                all_pts.append(pts)
        except Exception as e:
            print(f"    [warn] failed to load centerline {path}: {e}")
    return np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float64)


def _load_gt_labels(cfg: BenchmarkConfig, scan_id: str, ref_img: sitk.Image) -> np.ndarray:
    """The GT label mask resampled onto `ref_img`'s grid, labels untouched.

    `ref_img` comes from the saved prediction, which carries the original volume's
    header. The GT mask's own origin/direction is not assumed to match — resampling
    explicitly is what stops a header mismatch silently misaligning GT and prediction.
    """
    base = cfg.data.gt_mask_dir
    if base and not os.path.isabs(base):
        base = os.path.join(cfg.data.data_root, base)
    suffix = cfg.data.mask_suffix or cfg.data.file_extension
    path = bio.resolve_scan_path(base, scan_id, suffix)
    mask_img = sitk.ReadImage(path)

    mask_resampler = sitk.ResampleImageFilter()
    mask_resampler.SetReferenceImage(ref_img)
    mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampler.SetDefaultPixelValue(0)
    mask_img_r = mask_resampler.Execute(mask_img)

    return sitk.GetArrayFromImage(mask_img_r).transpose(2, 1, 0).astype(np.uint8)


def _load_gt(cfg: BenchmarkConfig, scan_id: str, ref_img: sitk.Image):
    """Binary GT lumen plus its centerline points, for the scan-level evaluation."""
    gt_raw = _load_gt_labels(cfg, scan_id, ref_img)
    p = cfg.data.params
    if p.get("binarise_lumen", True):
        gt_bin = bio.binarise_lumen(
            gt_raw,
            background_label=p.get("background_label", bio.LUMEN_BACKGROUND_LABEL),
        )
    else:
        gt_bin = gt_raw.copy()
    gt_centerline_pts = _load_gt_centerline_pts(cfg, scan_id)
    return gt_bin, gt_centerline_pts


POINT_COVARIATES = ("dist_mm", "depth", "seg", "segment_name", "diameter_mm", "hu",
                    "side", "is_end_point")
# The subset meaningful to average within a bin; the rest are categorical.
POINT_CONTINUOUS = ("dist_mm", "diameter_mm", "hu")
# Covariate the per-segment breakdown groups on, best first.
POINT_SEGMENT_KEYS = ("segment_name", "seg")


def _evaluate_points(cfg: BenchmarkConfig, scan_id: str, pred: np.ndarray,
                     gt: np.ndarray, ref_img) -> dict | None:
    """Local Dice at each precomputed GT centerline sample. The .npz holds every
    covariate depending on the GT alone; this adds the one column that depends on the
    prediction. Returns None when absent, so scan-level metrics still run."""
    rel = cfg.data.params.get("centerline_samples_dir", "centerline_samples")
    sdir = rel if os.path.isabs(rel) else os.path.join(cfg.data.data_root, rel)
    path = os.path.join(sdir, f"{scan_id}.npz")
    if not os.path.exists(path):
        print(f"    [warn] centerline samples not found: {path} — skipping point metrics")
        return None

    with np.load(path) as f:
        cols = {k: f[k] for k in POINT_COVARIATES if k in f}
        pts_mm = f["xyz_mm"].astype(np.float64)

    roi_size_mm = float(cfg.evaluation.point_metrics.get("roi_size_mm", 8.0))
    cols["local_dice"] = compute_local_dice(pred, gt, pts_mm, ref_img, roi_size_mm)
    return cols


def _evaluate_scan(config_path: str, results_dir: str, scan_id: str) -> tuple:
    """Evaluate one scan. Rebuilds its own config so it can run in a worker process
    without BenchmarkConfig needing to be picklable."""
    cfg = BenchmarkConfig.from_json(config_path)
    cfg.validate()
    if cfg.results_root and not os.path.isabs(results_dir):
        eval_dir = os.path.join(cfg.results_root, results_dir)
    else:
        eval_dir = results_dir
    cfg.evaluation.output_dir = eval_dir

    pred_dir = os.path.join(cfg.evaluation.output_dir, "predictions")
    pred_path = os.path.join(pred_dir, f"{scan_id}{cfg.data.file_extension}")
    if not os.path.exists(pred_path):
        # Scored set = whatever predictions exist. A partial run is still worth
        # summarising, as long as the report says how many scans it covers.
        print(f"    [warn] prediction not found: {pred_path} — skipping scan")
        return scan_id, None, None
    pred_orig, ref_img = bio.load_mask(pred_path)
    gt_orig, gt_centerline_pts = _load_gt(cfg, scan_id, ref_img)
    orig_spacing = ref_img.GetSpacing()

    standard_names  = [m for m in cfg.evaluation.metrics if m not in CENTERLINE_METRICS]
    centerline_names = [m for m in cfg.evaluation.metrics if m in CENTERLINE_METRICS]

    metrics = compute_metrics(pred_orig, gt_orig, standard_names, orig_spacing)

    cl = compute_centerline_metrics(pred_orig, ref_img, gt_centerline_pts)
    metrics.update({k: cl[k] for k in centerline_names})

    points = _evaluate_points(cfg, scan_id, pred_orig, gt_orig, ref_img)
    if points is not None:
        # Rides along in `metrics` so it appears in the ordinary summary and in
        # every descriptor subgroup for free.
        d = points["local_dice"]
        metrics["local_dice"] = float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")

    return scan_id, metrics, points


def evaluate(config_path: str, results_dir: str, num_workers: int = 8):
    seed_everything()
    cfg = BenchmarkConfig.from_json(config_path)
    cfg.validate()

    if cfg.results_root and not os.path.isabs(results_dir):
        eval_dir = os.path.join(cfg.results_root, results_dir)
    else:
        eval_dir = results_dir
    cfg.evaluation.output_dir = eval_dir

    print(f"[evaluate] method={cfg.method_name}")
    print(f"[evaluate] results_dir={cfg.evaluation.output_dir}")

    scan_ids = cfg.data.test_ids
    pred_dir = os.path.join(cfg.evaluation.output_dir, "predictions")
    all_results = {}
    all_points = {}

    missing_ids = []

    def _collect(scan_id, metrics, points):
        if metrics is None:  # no prediction on disk for this scan
            missing_ids.append(scan_id)
            return
        all_results[scan_id] = metrics
        if points is not None:
            all_points[scan_id] = points
        print(f"  {scan_id}: {metrics}")

    if num_workers <= 1:
        for scan_id in tqdm(scan_ids):
            _collect(*_evaluate_scan(config_path, results_dir, scan_id))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_evaluate_scan, config_path, results_dir, scan_id)
                       for scan_id in scan_ids]
            for future in tqdm(as_completed(futures), total=len(futures)):
                _collect(*future.result())

    n_total, n_valid = len(scan_ids), len(all_results)
    if n_valid == 0:
        raise FileNotFoundError(
            f"No predictions found in {pred_dir} for any of the {n_total} test scans.\n"
            f"Run `python -m inference -c {config_path} -r {results_dir}` first "
            f"to generate predictions (or place/symlink externally-generated "
            f"predictions there) before evaluating."
        )

    summary = _summarise(all_results)
    payload = {
        "per_scan": all_results,
        "summary": summary,
        "coverage": {
            "n_evaluated": n_valid,
            "n_test_ids": n_total,
            "n_missing_predictions": len(missing_ids),
            "missing_prediction_ids": sorted(missing_ids),
        },
    }

    strata = _summarise_by_group(cfg, all_results)
    if strata:
        payload["summary_by_group"] = strata

    point_summary = _summarise_points(cfg, all_points)
    if point_summary:
        payload["summary_by_point_group"] = point_summary
        csv_path = os.path.join(cfg.evaluation.output_dir, "point_metrics.csv")
        _write_point_csv(all_points, csv_path)
        print(f"Per-point metrics saved to {csv_path}")

    out_path = os.path.join(cfg.evaluation.output_dir, f"{cfg.method_name}_results.json")
    bio.save_results(payload, out_path)
    print(f"\nResults saved to {out_path}")
    print(f"Evaluated {n_valid}/{n_total} test scans with a prediction"
          + (f" ({len(missing_ids)} skipped, no prediction found)" if missing_ids else ""))
    if missing_ids:
        shown = ", ".join(sorted(missing_ids)[:10])
        print(f"  missing: {shown}" + (" ..." if len(missing_ids) > 10 else ""))
    print("Summary:", summary)
    _print_group_summary(strata, cfg.evaluation.metrics)
    _print_point_summary(point_summary)


# ---------------------------------------------------------------- multi-class --
# Coronary segment labels; 0 is background. Predictions for this mode live beside
# the binary ones under a distinct suffix, so both can be scored from one run dir.
MULTI_CLASS_LABELS = tuple(range(1, 15))
MULTI_CLASS_SUFFIX = ".multi_class.nii.gz"
# Label -> segment name, per the dataset's labelling scheme (see README).
SEGMENT_NAMES = {
    1: "LM", 2: "LAD", 3: "LCx", 4: "D1", 5: "D2", 6: "OM1", 7: "OM2",
    8: "IM", 9: "RCA", 10: "R-PDA", 11: "R-PLA", 12: "L-PDA", 13: "L-PLA",
    14: "Other",
}
# Every registry metric, since the point of this mode is the full per-segment table.
# Centerline metrics are excluded by construction: the GT centerline VTKs are not
# segment-labelled, so they cannot be scored per class.
MULTI_CLASS_METRICS = tuple(METRIC_REGISTRY)
# Scored from the two observers' centerline VTKs rather than from the masks, so they
# sit outside METRIC_REGISTRY and are computed separately below.
MULTI_CLASS_CENTERLINE_METRICS = ("centerline_md", "centerline_hd95")
MULTI_CLASS_REPORT_METRICS = MULTI_CLASS_METRICS + MULTI_CLASS_CENTERLINE_METRICS
# Point array naming the coronary segment each centerline vertex belongs to.
CENTERLINE_SEGMENT_ARRAY = "segment_label"


def _load_centerline_by_segment(directory: str, scan_id: str, source: str) -> dict:
    """{segment label: (N, 3) LPS mm points} from a scan's left and right VTKs.

    Splitting on the files' own `segment_label` array is what makes a per-segment
    centerline comparison possible at all. A file lacking that array is skipped with
    a warning rather than having all of its points folded into every segment, which
    would silently produce plausible-looking distances that mean nothing.
    """
    by_label: dict[int, list] = {}
    for side in CENTERLINE_SIDES:
        path = os.path.join(directory, f"{scan_id}.coronary_{side}_centerline.vtk")
        if not os.path.exists(path):
            continue
        try:
            pts, arrays = bio.load_vtk_centerline(path)
        except Exception as e:
            print(f"    [warn] failed to load {source} centerline {path}: {e}")
            continue
        if len(pts) == 0:
            continue
        labels = arrays.get(CENTERLINE_SEGMENT_ARRAY)
        if labels is None:
            print(f"    [warn] {path} has no '{CENTERLINE_SEGMENT_ARRAY}' array "
                  f"(found: {sorted(arrays)}) — its points are excluded from the "
                  f"per-segment centerline metrics")
            continue
        labels = np.asarray(labels).ravel().astype(int)
        for label in np.unique(labels):
            by_label.setdefault(int(label), []).append(pts[labels == label])
    return {k: np.concatenate(v, axis=0) for k, v in by_label.items()}


def _crop_to_union(pred: np.ndarray, gt: np.ndarray, pad: int = 2) -> tuple:
    """Both masks cropped to the bounding box of their union, padded by `pad` voxels.

    Every metric here is invariant to this crop: Dice/precision/recall/volumes are
    voxel counts, surface distances only ever reference voxels of the *other* mask
    (all of which are inside the box), and skeletons and Betti numbers are intrinsic
    to the mask provided it does not touch the array border — which the padding
    guarantees. One coronary segment fills ~0.005% of a CCTA grid, so this is the
    difference between minutes and milliseconds per label.
    """
    union = pred | gt
    if not union.any():
        return pred, gt
    box = []
    for axis in range(union.ndim):
        others = tuple(a for a in range(union.ndim) if a != axis)
        idx = np.where(union.any(axis=others))[0]
        box.append(slice(max(0, idx[0] - pad), min(union.shape[axis], idx[-1] + 1 + pad)))
    box = tuple(box)
    return pred[box], gt[box]


def _evaluate_scan_multi_class(config_path: str, results_dir: str, scan_id: str) -> tuple:
    """Per-segment metrics for one scan, prediction vs GT label for label.

    A segment is scored only where BOTH masks contain it. Absent from both, scoring
    it would count a shared omission as a perfect match and inflate mean Dice to 1.0;
    present in only one, Dice is 0 by construction and says nothing about delineation
    quality — that disagreement is already reported, in full, by `agreement_pct`.
    Scoring it in both places would count the same disagreement twice and drag the
    mean far below what the overlapping annotations actually show. It also keeps one
    denominator across the whole row: the surface and topology metrics are NaN
    against an empty mask, so they were only ever averaged over both-present scans.
    """
    cfg = BenchmarkConfig.from_json(config_path)
    cfg.validate()
    if cfg.results_root and not os.path.isabs(results_dir):
        eval_dir = os.path.join(cfg.results_root, results_dir)
    else:
        eval_dir = results_dir
    cfg.evaluation.output_dir = eval_dir

    pred_path = os.path.join(cfg.evaluation.output_dir, "predictions",
                             f"{scan_id}{MULTI_CLASS_SUFFIX}")
    if not os.path.exists(pred_path):
        print(f"    [warn] multi-class prediction not found: {pred_path} — skipping scan")
        return scan_id, None

    pred, ref_img = bio.load_mask(pred_path)
    gt = _load_gt_labels(cfg, scan_id, ref_img)
    spacing = ref_img.GetSpacing()

    # The observers' centerlines live beside their masks, under the same naming the
    # GT uses; both are already in LPS mm, so no resampling is involved.
    gt_cl = _load_centerline_by_segment(_centerline_dir(cfg), scan_id, "GT")
    pred_cl = _load_centerline_by_segment(os.path.dirname(pred_path), scan_id,
                                          "prediction")

    per_label = {}
    for label in MULTI_CLASS_LABELS:
        pred_l, gt_l = (pred == label), (gt == label)
        entry = {"segment_name": SEGMENT_NAMES[label],
                 "gt_present": bool(gt_l.any()), "pred_present": bool(pred_l.any())}
        if entry["gt_present"] and entry["pred_present"]:
            pred_l, gt_l = _crop_to_union(pred_l, gt_l)
            entry.update(compute_metrics(pred_l, gt_l, list(MULTI_CLASS_METRICS),
                                         spacing, warn=False))

        # Centerline presence is judged from the VTKs, not the masks, so a segment
        # one observer traced but did not label in the mask (or vice versa) is
        # handled on its own terms. Same both-present rule as above.
        g_pts, p_pts = gt_cl.get(label), pred_cl.get(label)
        entry["gt_centerline_present"] = g_pts is not None and len(g_pts) > 0
        entry["pred_centerline_present"] = p_pts is not None and len(p_pts) > 0
        if entry["gt_centerline_present"] and entry["pred_centerline_present"]:
            entry.update(compute_centerline_point_metrics(p_pts, g_pts))

        per_label[str(label)] = entry
    return scan_id, per_label


def _summarise_by_class(all_results: dict) -> dict:
    """Per label: mean±std of each metric over the scans where both masks contain the
    segment, plus how often each mask contains it and how often the two agree on that.

    `n_scored` is that both-present count, the denominator shared by every metric in
    the row. Presence disagreement is reported by `agreement_pct` alone and is
    deliberately kept out of the metric means; see `_evaluate_scan_multi_class`."""
    n_scans = len(all_results)
    out = {}
    for label in MULTI_CLASS_LABELS:
        key = str(label)
        values: dict[str, list] = {}
        gt_n = pred_n = both_n = agree_n = cl_both_n = 0
        for per_label in all_results.values():
            entry = per_label.get(key, {})
            in_gt, in_pred = bool(entry.get("gt_present")), bool(entry.get("pred_present"))
            gt_n += in_gt
            pred_n += in_pred
            both_n += in_gt and in_pred
            agree_n += in_gt == in_pred
            cl_both_n += (bool(entry.get("gt_centerline_present"))
                          and bool(entry.get("pred_centerline_present")))
            for name, value in entry.items():
                if isinstance(value, float):
                    values.setdefault(name, []).append(value)
        pct = lambda n: 100.0 * n / n_scans if n_scans else float("nan")  # noqa: E731
        out[key] = {
            "segment_name": SEGMENT_NAMES[label],
            "metrics": {name: _stats(v) for name, v in values.items()},
            "presence": {
                "n_scans": n_scans,
                "gt_present_pct": pct(gt_n),
                "pred_present_pct": pct(pred_n),
                "both_present_pct": pct(both_n),
                # Presence/absence agreement: 1 per scan where both masks either
                # contain the segment or both omit it, averaged over scans.
                "agreement_pct": pct(agree_n),
                # Denominator for the mask metrics, and for the centerline ones,
                # which are gated on the VTKs and so can differ.
                "n_scored": both_n,
                "centerline_n_scored": cl_both_n,
            },
        }
    return out


def _print_class_summary(by_class: dict):
    metric_names = [m for m in MULTI_CLASS_REPORT_METRICS
                    if any(m in c["metrics"] for c in by_class.values())]
    header = (f"\n{'seg':>4} {'name':<6} {'gt%':>6} {'pred%':>6} {'agree%':>7} {'n':>5} "
              f"{'n_cl':>5}  " + "  ".join(f"{m:>18}" for m in metric_names))
    print(header)
    for key, block in by_class.items():
        p = block["presence"]
        cells = []
        for m in metric_names:
            s = block["metrics"].get(m)
            cell = "—" if s is None or not s["n_valid"] else f"{s['mean']:.4g}±{s['std']:.3g}"
            cells.append(f"{cell:>18}")
        print(f"{key:>4} {block['segment_name']:<6} "
              f"{p['gt_present_pct']:>6.1f} {p['pred_present_pct']:>6.1f} "
              f"{p['agreement_pct']:>7.1f} {p['n_scored']:>5} "
              f"{p['centerline_n_scored']:>5}  " + "  ".join(cells))


def evaluate_multi_class(config_path: str, results_dir: str, num_workers: int = 8):
    """Score <run_dir>/predictions/<scan_id>.multi_class.nii.gz per coronary segment.

    Deliberately narrower than `evaluate`: no descriptor stratification and no
    centerline/point metrics, just the per-segment table.
    """
    seed_everything()
    cfg = BenchmarkConfig.from_json(config_path)
    cfg.validate()

    if cfg.results_root and not os.path.isabs(results_dir):
        eval_dir = os.path.join(cfg.results_root, results_dir)
    else:
        eval_dir = results_dir
    cfg.evaluation.output_dir = eval_dir

    print(f"[evaluate:multi-class] method={cfg.method_name}")
    print(f"[evaluate:multi-class] results_dir={cfg.evaluation.output_dir}")

    scan_ids = cfg.data.test_ids
    pred_dir = os.path.join(cfg.evaluation.output_dir, "predictions")
    all_results, missing_ids = {}, []

    def _collect(scan_id, per_label):
        if per_label is None:
            missing_ids.append(scan_id)
            return
        all_results[scan_id] = per_label
        scored = sum(1 for e in per_label.values() if e["gt_present"] and e["pred_present"])
        print(f"  {scan_id}: {scored}/{len(MULTI_CLASS_LABELS)} segments scored")

    if num_workers <= 1:
        for scan_id in tqdm(scan_ids):
            _collect(*_evaluate_scan_multi_class(config_path, results_dir, scan_id))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_evaluate_scan_multi_class, config_path, results_dir, s)
                       for s in scan_ids]
            for future in tqdm(as_completed(futures), total=len(futures)):
                _collect(*future.result())

    n_total, n_valid = len(scan_ids), len(all_results)
    if n_valid == 0:
        raise FileNotFoundError(
            f"No '*{MULTI_CLASS_SUFFIX}' predictions found in {pred_dir} for any of the "
            f"{n_total} test scans."
        )

    by_class = _summarise_by_class(all_results)
    payload = {
        "per_scan": all_results,
        "summary_by_class": by_class,
        "coverage": {
            "n_evaluated": n_valid,
            "n_test_ids": n_total,
            "n_missing_predictions": len(missing_ids),
            "missing_prediction_ids": sorted(missing_ids),
        },
    }

    out_path = os.path.join(cfg.evaluation.output_dir,
                            f"{cfg.method_name}_multi_class_results.json")
    bio.save_results(payload, out_path)
    print(f"\nResults saved to {out_path}")
    print(f"Evaluated {n_valid}/{n_total} test scans with a multi-class prediction"
          + (f" ({len(missing_ids)} skipped, none found)" if missing_ids else ""))
    _print_class_summary(by_class)


def _group_sort_key(value: str):
    """Numeric-looking group values sort numerically, the rest lexically."""
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def _summarise_by_group(cfg: BenchmarkConfig, results: dict) -> dict:
    """Re-summarise per-scan results within each scan-level descriptor subgroup.

    Subgroup values come from the descriptor file rather than a hardcoded list, so an
    unexpected label shows up as its own group instead of being silently merged.
    Returns {} with a warning when the table is missing — that must not fail an
    otherwise complete evaluation.
    """
    rel = (cfg.evaluation.descriptors_file or "").strip()
    columns = list(cfg.evaluation.stratify_by or [])
    if not rel or not columns:
        return {}

    path = rel if os.path.isabs(rel) else os.path.join(cfg.data.data_root, rel)
    if not os.path.exists(path):
        print(f"[warn] descriptor file not found: {path} — skipping subgroup summaries")
        return {}
    try:
        table = bio.load_descriptors(path, id_column=cfg.evaluation.descriptors_id_column)
    except Exception as e:
        print(f"[warn] could not read descriptor file {path}: {e} — skipping subgroup summaries")
        return {}

    missing = [s for s in results if str(s) not in table]
    if missing:
        print(f"[warn] {len(missing)}/{len(results)} evaluated scans absent from {os.path.basename(path)} "
              f"(e.g. {missing[:5]}) — excluded from subgroup summaries")

    strata = {}
    for column in columns:
        if not any(column in row for row in table.values()):
            print(f"[warn] descriptor column '{column}' not found in {os.path.basename(path)} — skipping")
            continue

        groups: dict[str, dict] = {}
        for scan_id, metrics in results.items():
            value = table.get(str(scan_id), {}).get(column, "")
            if not value:
                continue
            groups.setdefault(value, {})[scan_id] = metrics

        if groups:
            strata[column] = {
                value: _summarise(groups[value])
                for value in sorted(groups, key=_group_sort_key)
            }
    return strata


def _print_group_summary(strata: dict, metric_names: list):
    """One line per subgroup: mean±std per metric, plus the group's n."""
    if not strata:
        return
    for column, groups in strata.items():
        print(f"\nBy {column}:")
        for value, summary in groups.items():
            n = max((summary[m]["n_total"] for m in summary), default=0)
            # Config order first, then anything else the summary holds — local_dice
            # comes from the point pass rather than evaluation.metrics.
            ordered = [m for m in metric_names if m in summary]
            ordered += [m for m in summary if m not in ordered]
            parts = [f"{m} {summary[m]['mean']:.4g}±{summary[m]['std']:.3g}" for m in ordered]
            print(f"  {value:<10} n={n:<4} " + "  ".join(parts))


def _bin_labels(edges: list) -> list:
    """['<25', '25-50', ..., '>150'] for cut points [25, 50, ..., 150]. The config
    lists CUT POINTS, so N of them give N+1 bins and no sample is ever dropped for
    falling outside the range. Bins are lower-inclusive."""
    labels = [f"<{edges[0]:g}"]
    labels += [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    return labels + [f">{edges[-1]:g}"]


def _summarise_points(cfg: BenchmarkConfig, all_points: dict) -> dict:
    """Group per-point local Dice by segment, HU, geodesic distance and diameter.

    Aggregation is MACRO: one mean per scan per bin, then statistics over scans.
    Pooling raw points would let a long well-opacified tree outweigh a short one and
    make the std meaningless as between-patient variability.

    Every bin also reports its mean diameter_mm, the confounder shared by the other
    groupings — distal vessels are both thinner and dimmer — so a drop in local Dice
    can be read against the calibre change accompanying it.
    """
    if not all_points:
        return {}
    bins_cfg = cfg.evaluation.point_metrics.get("bins", {}) or {}

    def _block(scan_masks: dict) -> dict:
        """scan_id -> boolean mask of that scan's points falling in this bin."""
        per_scan_dice, dice_all, cov_sums, n_points = [], [], {}, 0
        for scan_id, mask in scan_masks.items():
            if not mask.any():
                continue
            cols = all_points[scan_id]
            d = cols["local_dice"][mask]
            n_points += int(mask.sum())
            dice_all.append(d)
            if np.isfinite(d).any():
                per_scan_dice.append(float(np.nanmean(d)))
            for name in POINT_CONTINUOUS:
                if name in cols:
                    cov_sums.setdefault(name, []).append(cols[name][mask].astype(float))
        if n_points == 0:
            return None
        return {
            "local_dice": _stats(per_scan_dice),
            "n_points": n_points,
            "covariates": {k: float(np.nanmean(np.concatenate(v))) for k, v in cov_sums.items()},
        }

    out: dict = {}

    # Coronary segment: categorical, not binned. Prefers the VTK's own
    # `segment_name` ('LAD', 'RCA', ...) over the numeric label, since a results
    # table reading "LAD" beats one reading "2".
    seg_key = next((k for k in POINT_SEGMENT_KEYS
                    if any(k in cols for cols in all_points.values())), None)
    if seg_key:
        values = sorted({str(v) for cols in all_points.values()
                         for v in np.unique(cols[seg_key])})
        seg_out = {}
        for label in values:
            block = _block({s: (cols[seg_key].astype(str) == label)
                            for s, cols in all_points.items() if seg_key in cols})
            if block:
                seg_out[label] = block
        if seg_out:
            out[seg_key] = seg_out

    for name, edges in bins_cfg.items():
        if not edges:
            continue
        edges = [float(e) for e in edges]
        labels = _bin_labels(edges)
        if not any(name in cols for cols in all_points.values()):
            print(f"[warn] point covariate '{name}' missing from the sample files — skipping "
                  f"its breakdown (regenerate with utils.precompute_centerline_samples)")
            continue
        # searchsorted maps a value to 0..len(edges), indexing the N+1 labels
        # directly. Non-finite values would sort to the top bin, so they get -1.
        idx_per_scan = {}
        for s, cols in all_points.items():
            if name not in cols:
                continue
            vals = cols[name].astype(float)
            idx = np.searchsorted(edges, vals, side="right")
            idx[~np.isfinite(vals)] = -1
            idx_per_scan[s] = idx
        grouped = {}
        for b, label in enumerate(labels):
            block = _block({s: (idx == b) for s, idx in idx_per_scan.items()})
            if block:
                grouped[label] = block
        if grouped:
            out[name] = grouped
    return out


def _write_point_csv(all_points: dict, path: str):
    """One row per sampled point — the tidy table for analysis outside this script."""
    import csv

    names = [c for c in POINT_COVARIATES
             if any(c in cols for cols in all_points.values())]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan_id"] + names + ["local_dice"])
        for scan_id in sorted(all_points):
            cols = all_points[scan_id]
            for i in range(len(cols["local_dice"])):
                w.writerow([scan_id] + [cols[c][i] for c in names] + [cols["local_dice"][i]])


def _print_point_summary(point_summary: dict):
    if not point_summary:
        return
    for name, groups in point_summary.items():
        print(f"\nLocal Dice by {name}:")
        print(f"  {'bin':<12} {'n_pts':>7} {'n_scans':>8} {'diam':>7} {'hu':>7} "
              f"{'dist':>7}  local_dice")
        for label, block in groups.items():
            c = block["covariates"]
            print(f"  {label:<12} {block['n_points']:>7} "
                  f"{block['local_dice']['n_total']:>8} "
                  f"{c.get('diameter_mm', float('nan')):>7.2f} "
                  f"{c.get('hu', float('nan')):>7.0f} "
                  f"{c.get('dist_mm', float('nan')):>7.1f}  "
                  f"{block['local_dice']['mean']:.4f}±{block['local_dice']['std']:.3f}")


def _stats(values: list) -> dict:
    arr = np.asarray(values, dtype=float)
    n_valid = int(np.count_nonzero(~np.isnan(arr)))
    return {
        "mean":    float(np.nanmean(arr)) if n_valid else float("nan"),
        "std":     float(np.nanstd(arr))  if n_valid else float("nan"),
        "n_valid": n_valid,
        "n_total": int(arr.size),
    }


def _summarise(results: dict) -> dict:
    """mean/std over scans of every scalar metric. Non-scalar entries aggregate over
    points instead, see `_summarise_points`."""
    all_metrics: dict[str, list] = {}
    for scan_metrics in results.values():
        for k, v in scan_metrics.items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, []).append(v)
    return {k: _stats(v) for k, v in all_metrics.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to method config JSON")
    parser.add_argument("-r", "--results-dir", required=True,
                        help="Run folder to evaluate, as printed by train "
                             "(e.g. cas_net_2026-07-05-14-30-22-123456). Relative to "
                             "results_root in the config, or an absolute path.")
    parser.add_argument("-j", "--workers", type=int, default=8,
                        help="Number of scans to evaluate in parallel (separate "
                             "processes). Set to 1 to disable parallelism. Default: 8.")
    parser.add_argument("--multi_class", "--multi-class", dest="multi_class",
                        action="store_true",
                        help="Score predictions/<scan_id>.multi_class.nii.gz per "
                             "coronary segment (labels 1-14) instead of the binary "
                             "lumen evaluation.")
    args = parser.parse_args()
    if args.multi_class:
        evaluate_multi_class(args.config, args.results_dir, num_workers=args.workers)
    else:
        evaluate(args.config, args.results_dir, num_workers=args.workers)
