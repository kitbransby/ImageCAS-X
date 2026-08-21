import numpy as np


def _betti_numbers(mask: np.ndarray) -> tuple:
    """b0 (components) and b1 (loops), 26-connectivity. b1 is approximated as
    max(0, b0 - chi) assuming no enclosed voids, which holds for tubular structures."""
    from scipy.ndimage import label
    from skimage.measure import euler_number

    mask = mask.astype(bool)
    if not mask.any():
        return 0, 0

    b0 = label(mask, structure=np.ones((3, 3, 3)))[1]
    chi = euler_number(mask, connectivity=3)
    b1 = max(0, int(b0 - chi))
    return b0, b1


def betti_error_1(pred: np.ndarray, gt: np.ndarray, **_) -> float:
    """Absolute error in b0 (connected components)."""
    return float(abs(_betti_numbers(pred)[0] - _betti_numbers(gt)[0]))


def betti_error_2(pred: np.ndarray, gt: np.ndarray, **_) -> float:
    """Absolute error in b1 (loops / tunnels)."""
    return float(abs(_betti_numbers(pred)[1] - _betti_numbers(gt)[1]))


def betti_error_1_and_2(pred: np.ndarray, gt: np.ndarray, **_) -> float:
    """Sum of absolute errors in b0 and b1."""
    b0_p, b1_p = _betti_numbers(pred)
    b0_g, b1_g = _betti_numbers(gt)
    return float(abs(b0_p - b0_g) + abs(b1_p - b1_g))


def dice(pred: np.ndarray, gt: np.ndarray, **_) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0
    return 2 * intersection / union


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0), **_) -> float:
    """95th-percentile symmetric Hausdorff distance in mm."""
    from scipy.ndimage import distance_transform_edt, binary_erosion

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any():
        raise ValueError("hausdorff_95: predicted mask is empty — cannot compute surface distance.")
    if not gt.any():
        raise ValueError("hausdorff_95: ground-truth mask is empty — cannot compute surface distance.")

    dist_pred = distance_transform_edt(~pred, sampling=spacing)
    dist_gt = distance_transform_edt(~gt, sampling=spacing)

    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)

    forward = dist_pred[gt_surface]
    backward = dist_gt[pred_surface]

    return float(max(np.percentile(forward, 95), np.percentile(backward, 95)))


def cl_dice(pred: np.ndarray, gt: np.ndarray, **_) -> float:
    """Centreline Dice for tubular structures (Shit et al. 2021)."""
    from skimage.morphology import skeletonize

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any():
        raise ValueError("cl_dice: predicted mask is empty.")
    if not gt.any():
        raise ValueError("cl_dice: ground-truth mask is empty.")

    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)

    t_prec = (skel_pred & gt).sum() / (skel_pred.sum() + 1e-8)
    t_sen = (skel_gt & pred).sum() / (skel_gt.sum() + 1e-8)

    if t_prec + t_sen == 0:
        return 0.0
    return float(2 * t_prec * t_sen / (t_prec + t_sen))


def volume_pred_ml(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0), **_) -> float:
    """Predicted lumen volume in ml."""
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return float(pred.astype(bool).sum() * voxel_vol / 1000.0)


def volume_gt_ml(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0), **_) -> float:
    """GT lumen volume in ml."""
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return float(gt.astype(bool).sum() * voxel_vol / 1000.0)


def volume_mad_ml(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0), **_) -> float:
    """Absolute volume difference |vol_pred - vol_gt| in ml."""
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return float(abs(pred.astype(bool).sum() - gt.astype(bool).sum()) * voxel_vol / 1000.0)


def volume_bias_ml(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0), **_) -> float:
    """Signed volume difference vol_pred - vol_gt in ml; positive = overestimate."""
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return float((pred.astype(bool).sum() - gt.astype(bool).sum()) * voxel_vol / 1000.0)


def precision(pred: np.ndarray, gt: np.ndarray, **_) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    if tp + fp == 0:
        raise ValueError("precision: predicted mask is empty.")
    return float(tp / (tp + fp))


def recall(pred: np.ndarray, gt: np.ndarray, **_) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    if tp + fn == 0:
        raise ValueError("recall: ground-truth mask is empty.")
    return float(tp / (tp + fn))


def _voxel_indices_to_mm(indices: np.ndarray, ref_img) -> np.ndarray:
    """Voxel indices -> LPS mm."""
    origin = np.array(ref_img.GetOrigin())
    spacing = np.array(ref_img.GetSpacing())
    direction = np.array(ref_img.GetDirection()).reshape(3, 3)
    return origin + (direction @ (indices * spacing).T).T


def compute_centerline_metrics(pred_mask: np.ndarray, ref_img,
                               gt_centerline_pts: np.ndarray) -> dict:
    """Symmetric distances between the predicted skeleton and the GT centerline: MD
    is the average symmetric distance, HD95 the 95th-percentile Hausdorff."""
    from skimage.morphology import skeletonize
    from scipy.spatial import KDTree

    pred = pred_mask.astype(bool)
    if not pred.any():
        raise ValueError("predicted mask is empty.")
    if len(gt_centerline_pts) == 0:
        raise ValueError("GT centerline has no points.")

    skel = skeletonize(pred)
    if not skel.any():
        raise ValueError("predicted skeleton is empty after skeletonization.")

    pred_pts = _voxel_indices_to_mm(np.argwhere(skel).astype(np.float64), ref_img)

    tree_pred = KDTree(pred_pts)
    d_g2p, _ = tree_pred.query(gt_centerline_pts)
    md_g2p    = float(np.mean(d_g2p))
    hd95_g2p  = float(np.percentile(d_g2p, 95))

    tree_gt = KDTree(gt_centerline_pts)
    d_p2g, _ = tree_gt.query(pred_pts)
    md_p2g   = float(np.mean(d_p2g))
    hd95_p2g = float(np.percentile(d_p2g, 95))

    return {
        "centerline_md":   (md_g2p + md_p2g) / 2.0,
        "centerline_hd95": max(hd95_g2p, hd95_p2g),
    }


def compute_centerline_point_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray) -> dict:
    """Symmetric distances between two centerline point sets, both in LPS mm.

    `compute_centerline_metrics` skeletonises a predicted *mask* to obtain its
    centerline; here both curves are supplied directly, so that step — and the voxel
    quantisation it imposes — drops out. The aggregation is identical to that
    function's, so the two are directly comparable.
    """
    from scipy.spatial import KDTree

    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    if len(pred_pts) == 0:
        raise ValueError("predicted centerline has no points.")
    if len(gt_pts) == 0:
        raise ValueError("GT centerline has no points.")

    d_g2p, _ = KDTree(pred_pts).query(gt_pts)
    d_p2g, _ = KDTree(gt_pts).query(pred_pts)
    return {
        "centerline_md": float((np.mean(d_g2p) + np.mean(d_p2g)) / 2.0),
        "centerline_hd95": float(max(np.percentile(d_g2p, 95),
                                     np.percentile(d_p2g, 95))),
    }


def _mm_to_voxel_indices(pts_mm: np.ndarray, ref_img) -> np.ndarray:
    """LPS mm -> rounded voxel indices."""
    origin = np.array(ref_img.GetOrigin())
    spacing = np.array(ref_img.GetSpacing())
    direction = np.array(ref_img.GetDirection()).reshape(3, 3)
    return np.round(np.linalg.solve(direction, (pts_mm - origin).T).T / spacing).astype(int)


def compute_local_dice(pred: np.ndarray, gt: np.ndarray, pts_mm: np.ndarray,
                       ref_img, roi_size_mm: float) -> np.ndarray:
    """Dice inside a cubic ROI centred on each point — the per-point outcome the
    stratified results are built from, and the only part depending on a prediction.

    The ROI is a fixed PHYSICAL size, not a voxel count: original spacing varies
    across the dataset, so a fixed 16^3 box would cover 8mm in one scan and 5mm in
    another and bins pooled across scans would silently mix the two.

    NaN where the ROI contains no GT, the point having strayed off the annotated
    lumen, so local Dice would score background against background.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    shape = np.array(gt.shape)

    spacing = np.array(ref_img.GetSpacing())
    half = np.maximum(np.round((roi_size_mm / 2.0) / spacing).astype(int), 1)
    idx = _mm_to_voxel_indices(np.atleast_2d(pts_mm), ref_img)

    out = np.full(len(idx), np.nan, dtype=np.float64)
    for i, centre in enumerate(idx):
        lo = np.maximum(centre - half, 0)
        hi = np.minimum(centre + half + 1, shape)
        if np.any(lo >= hi):  # point lies outside the volume entirely
            continue
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        g = gt[sl]
        if not g.any():
            continue
        p = pred[sl]
        out[i] = 2.0 * (p & g).sum() / (p.sum() + g.sum())
    return out


CENTERLINE_METRICS = frozenset({"centerline_md", "centerline_hd95"})

METRIC_REGISTRY = {
    "dice": dice,
    "hd95": hausdorff_95,
    "cl_dice": cl_dice,
    "precision": precision,
    "recall": recall,
    "betti_error_1": betti_error_1,
    "betti_error_2": betti_error_2,
    "betti_error_1_and_2": betti_error_1_and_2,
    "volume_pred_ml": volume_pred_ml,
    "volume_gt_ml": volume_gt_ml,
    "volume_mad_ml": volume_mad_ml,
    "volume_bias_ml": volume_bias_ml,
}


def evaluate(pred: np.ndarray, gt: np.ndarray, metric_names: list,
             spacing: tuple = (1.0, 1.0, 1.0), warn: bool = True) -> dict:
    """One that cannot be computed (an empty prediction for a surface metric)
    returns NaN with a warning, so a single bad scan does not abort the run.

    `warn=False` silences those messages, for callers that expect empty masks in
    bulk (per-class scoring, where most segments are absent from most scans)."""
    unknown = [n for n in metric_names if n not in METRIC_REGISTRY and n not in CENTERLINE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. "
                         f"Available registry: {list(METRIC_REGISTRY)}; "
                         f"centerline (handled separately): {list(CENTERLINE_METRICS)}")
    metric_names = [n for n in metric_names if n not in CENTERLINE_METRICS]
    results = {}
    for name in metric_names:
        try:
            results[name] = METRIC_REGISTRY[name](pred, gt, spacing=spacing)
        except ValueError as e:
            if warn:
                print(f"    [warn] metric '{name}' -> NaN ({e})")
            results[name] = float("nan")
    return results
