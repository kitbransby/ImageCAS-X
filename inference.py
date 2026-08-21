"""Full-volume inference: the tiling/stitching library, and the script that loads a
checkpoint and writes predictions for evaluate.py to score.

    python -m inference -c configs/<method>.json -r <run_dir>

    python -m inference -c configs/<method>.json -r <run_dir> --computational_analysis

Writes <run_dir>/predictions/<scan_id><ext> plus computational_cost.json;
--computational_analysis writes the latter only, over a fixed slice of the split.
evaluate.py
never loads a model, so externally-produced predictions can be dropped into
predictions/ and scored without running this.

`run_inference` is the single entry point, and its contract never changes: the full
preprocessed volume for one scan in, full-volume logits on CPU out. How that is met
depends on cfg.input_type:

  - patch/random_crop/ade_htl_crop: tile into overlapping windows and stitch the
    logits back (`patch_stitch_inference`).
  - patch with data.params.centers_dir set (ImageCAS Stage 3): predict only at
    precomputed skeleton centres. Dense tiling would run these nets far outside the
    regime they trained on.
  - anything else: one forward pass. Covers imagecas_baseline, whose own forward()
    runs its coarse->patch->assemble pipeline internally.

Mirror TTA wraps every branch, averaging over the 4 flip combinations of axes X and
Y. Z is left unflipped to match the training-time random_flip axes -- the network
never saw a craniocaudally-flipped volume, and that flip is not anatomically valid.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

from utils.config import BenchmarkConfig
from utils.seeding import seed_everything
from utils.tiling import axis_starts
from utils import io as bio
from dataloading.factory import build_eval_dataloader
from models.registry import build_model, is_staged_model
from models.imagecas_baseline import _extract_patch, _assemble_predictions
from preprocessing.pipeline import build_preprocessing
from postprocessing.pipeline import build_postprocessing


# --computational_analysis timing window: scans [1:11] of the split, i.e. one untimed
# warm-up followed by 10 measured scans. Fixed rather than a CLI number so every method
# in the efficiency table is measured over the same scans.
_COST_WARMUP_SCANS = 1
_COST_TIMED_SCANS = 10


def _gaussian_kernel(patch_size, sigma_scale: float) -> torch.Tensor:
    """nnU-Net's sliding-window weighting: per-axis sigma = patch_size * sigma_scale,
    so tile centres dominate over weaker-context edges. Unnormalised, since
    accumulation divides by the weight sum."""
    px, py, pz = patch_size
    sx, sy, sz = (max(p * sigma_scale, 1e-8) for p in (px, py, pz))
    coords = [torch.arange(p, dtype=torch.float32) - (p - 1) / 2.0 for p in (px, py, pz)]
    dx, dy, dz = torch.meshgrid(*coords, indexing="ij")
    dist_sq = (dx / sx) ** 2 + (dy / sy) ** 2 + (dz / sz) ** 2
    return torch.exp(-dist_sq / 2)


def patch_stitch_inference(model, volume: torch.Tensor, patch_size, device,
                           batch_size: int = 4, overlap: float = 0.5,
                           weighting: str = "gaussian", sigma_scale: float = 0.125) -> torch.Tensor:
    """Tile `volume`, run `model` per tile, stitch the overlapping logits back into
    one (1, C_out, X, Y, Z) tensor. Tiles past the volume edge are zero-padded and
    cropped back out."""
    assert volume.shape[0] == 1, "patch_stitch_inference expects one full volume at a time"
    _, c_in, X, Y, Z = volume.shape
    px, py, pz = (int(s) for s in patch_size)
    stride = tuple(max(1, int(round(p * (1 - overlap)))) for p in (px, py, pz))

    starts = [
        (x0, y0, z0)
        for x0 in axis_starts(X, px, stride[0])
        for y0 in axis_starts(Y, py, stride[1])
        for z0 in axis_starts(Z, pz, stride[2])
    ]

    if weighting == "gaussian":
        weight = _gaussian_kernel((px, py, pz), sigma_scale)
    else:
        weight = torch.ones(px, py, pz, dtype=torch.float32)

    accum = None
    count = torch.zeros(1, X, Y, Z, dtype=torch.float32)

    for b0 in range(0, len(starts), batch_size):
        batch_starts = starts[b0:b0 + batch_size]
        patches = []
        for x0, y0, z0 in batch_starts:
            patch = torch.zeros(1, c_in, px, py, pz, dtype=volume.dtype)
            x1, y1, z1 = min(x0 + px, X), min(y0 + py, Y), min(z0 + pz, Z)
            patch[:, :, :x1 - x0, :y1 - y0, :z1 - z0] = volume[:, :, x0:x1, y0:y1, z0:z1]
            patches.append(patch)
        batch = torch.cat(patches, dim=0).to(device)

        with torch.no_grad():
            logits = model(batch)["logits"]
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = logits.cpu()

        if accum is None:
            accum = torch.zeros(logits.shape[1], X, Y, Z, dtype=torch.float32)

        for i, (x0, y0, z0) in enumerate(batch_starts):
            x1, y1, z1 = min(x0 + px, X), min(y0 + py, Y), min(z0 + pz, Z)
            w = weight[:x1 - x0, :y1 - y0, :z1 - z0]
            accum[:, x0:x1, y0:y1, z0:z1] += logits[i, :, :x1 - x0, :y1 - y0, :z1 - z0] * w
            count[:, x0:x1, y0:y1, z0:z1] += w

    accum /= count.clamp(min=1e-8)
    return accum.unsqueeze(0)


# Tensor dims of (1, C, X, Y, Z) that mirror TTA flips: X and Y only, see module
# docstring.
_MIRROR_AXES = (2, 3)


def _mirror_axis_combos() -> list:
    """All subsets of the mirrored axes, including the empty (no-flip) one."""
    combos = [()]
    for axis in _MIRROR_AXES:
        combos = combos + [c + (axis,) for c in combos]
    return combos


def _forward_full_volume(model, volume: torch.Tensor, cfg: BenchmarkConfig, device) -> torch.Tensor:
    """One full-volume logits pass in `volume`'s current orientation: a single
    forward, or sliding-window tiling for patch-trained methods."""
    if cfg.input_type in ("patch", "random_crop", "ade_htl_crop"):
        p = cfg.data.params
        default_ps = [128, 160, 160] if cfg.input_type == "random_crop" else 64
        raw_ps = p.get("patch_size", default_ps)
        patch_size = raw_ps if isinstance(raw_ps, (list, tuple)) else [int(raw_ps)] * 3
        overlap = p.get("inference_overlap", 0.5)
        batch_size = p.get("inference_batch_size", 6)
        weighting = p.get("inference_weighting", "gaussian")
        sigma_scale = p.get("inference_gaussian_sigma_scale", 0.125)
        return patch_stitch_inference(model, volume, patch_size, device,
                                      batch_size=batch_size, overlap=overlap,
                                      weighting=weighting, sigma_scale=sigma_scale)

    with torch.no_grad():
        out = model(volume.to(device))
    logits = out["logits"]
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits.cpu()


def run_skeleton_patch_inference(model, volume: torch.Tensor, cfg: BenchmarkConfig,
                                 device, scan_id: str, axes: tuple = ()) -> torch.Tensor:
    """Standalone Stage 3 inference for one ImageCAS patch net: predict only at the
    precomputed skeleton centres and assemble the overlapping patch probabilities.
    Reuses `_extract_patch`/`_assemble_predictions` so this matches
    ImageCASBaseline._run_patch_stage exactly.

    `axes` are the axes the mirror-TTA wrapper already flipped on `volume`; centres
    are precomputed in the unflipped grid, so they are flipped here to match.

    Returns an inverse-sigmoid of the assembled probabilities, so predict()'s
    sigmoid -> resample -> threshold path recovers them exactly.
    """
    assert volume.shape[0] == 1, "run_skeleton_patch_inference expects one full volume at a time"

    centers_dir = cfg.data.params.get("centers_dir")
    if not centers_dir:
        raise ValueError(
            f"{cfg.method_name}: input_type is 'patch' but data.params.centers_dir "
            f"is not set -- required to know where to read precomputed skeleton "
            f"centres from (see models/imagecas_baseline/stage3_generate_centers.py)."
        )
    centers_path = os.path.join(centers_dir, f"{scan_id}.npy")
    if not os.path.exists(centers_path):
        raise FileNotFoundError(
            f"No precomputed skeleton centres for scan '{scan_id}' at {centers_path}. "
            f"Generate them first (including the test split): python -m "
            f"models.imagecas_baseline.stage3_generate_centers -c configs/imagecas_stage2_coarse_dilated.json "
            f"--coarse-checkpoint <stage2_best.pt> --out-dir {centers_dir} --split test"
        )
    centers = np.load(centers_path)

    vol_np = volume[0, 0].cpu().numpy()
    vol_shape = vol_np.shape

    if len(centers) == 0:
        prob = np.zeros(vol_shape, dtype=np.float32)
    else:
        for axis in axes:
            col = axis - 2  # tensor dim 2 (X) -> centres col 0, dim 3 (Y) -> col 1
            centers[:, col] = (vol_shape[col] - 1) - centers[:, col]

        patch_size = cfg.data.params["patch_size"]
        batch_size = cfg.data.params.get("inference_batch_size", 64)

        model.eval()
        preds = []
        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start:start + batch_size]
            patches = np.stack([_extract_patch(vol_np, c, patch_size) for c in batch_centers])
            t = torch.from_numpy(patches).float().unsqueeze(1).to(device)
            with torch.no_grad():
                out = model(t)
            logits = out["logits"]
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = torch.sigmoid(logits).cpu().numpy()
            if pred.ndim == 5:
                pred = pred[:, 0]  # (B, P, P, P)
            preds.extend([pred[i] for i in range(len(batch_centers))])

        prob = _assemble_predictions(preds, centers, patch_size, vol_shape, threshold=None)

    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(prob / (1 - prob)).astype(np.float32)
    return torch.from_numpy(logit)[None, None]


def _mirror_tta(forward_fn, volume: torch.Tensor, combos: list, model=None) -> torch.Tensor:
    """For each axis combo, flip `volume`, forward it, flip the logits back, average.

    A model whose output channels carry a DIRECTION (ADE-HTL's connectivity) must
    also be un-flipped along the channel axis, since mirroring the volume mirrors
    every offset. Such a model exposes `mirror_channel_permutation`; without it, TTA
    would average channels describing different directions.
    """
    permute = getattr(model, "mirror_channel_permutation", None)
    accum = None
    for axes in combos:
        flipped = torch.flip(volume, dims=axes) if axes else volume
        logits = forward_fn(flipped, axes)
        if axes:
            logits = torch.flip(logits, dims=axes)
            if permute is not None:
                # axes are tensor dims of (1, C, X, Y, Z); the model reasons in
                # spatial axes 0/1/2.
                logits = logits[:, permute(tuple(a - 2 for a in axes))]
        accum = logits if accum is None else accum + logits
    return accum / len(combos)


def run_inference(model, volume: torch.Tensor, cfg: BenchmarkConfig, device,
                  scan_id: str | None = None) -> dict:
    """Full-volume logits for one scan, averaged over mirror TTA. `scan_id` is only
    used by the skeleton-centred branch, to find that scan's precomputed centres."""
    model.eval()

    combos = _mirror_axis_combos() if cfg.data.params.get("inference_mirror_tta", True) else [()]

    if cfg.input_type == "patch":
        # Fail loudly rather than fall back to the dense tiler, which would run a
        # patch-scale net over the ENTIRE volume and produce false positives far
        # outside the coronary tree.
        if not cfg.data.params.get("centers_dir"):
            raise ValueError(
                f"{cfg.method_name}: input_type is 'patch' but data.params.centers_dir is "
                f"not set (or empty). Generate skeleton centres first -- including the TEST "
                f"split, which build_eval_dataloader always uses regardless of training "
                f"input_type: python -m models.imagecas_baseline.stage3_generate_centers "
                f"-c configs/imagecas_stage2_coarse_dilated.json --coarse-checkpoint "
                f"<stage2_best.pt> --out-dir <centers_dir> --split test  -- then set "
                f"data.params.centers_dir to <centers_dir> in this config."
            )
        forward_fn = lambda v, axes: run_skeleton_patch_inference(model, v, cfg, device, scan_id, axes)
    else:
        forward_fn = lambda v, axes: _forward_full_volume(model, v, cfg, device)

    return {"logits": _mirror_tta(forward_fn, volume, combos, model=model)}


def _load_reference_volume(cfg: BenchmarkConfig, scan_id: str) -> sitk.Image:
    """The original CT volume's sitk geometry. Predictions are produced in a frame
    derived from this header, not the GT mask file's, and evaluate.py later resamples
    GT onto the saved prediction's header rather than re-deriving it."""
    vol_base = cfg.data.volume_dir
    if vol_base and not os.path.isabs(vol_base):
        vol_base = os.path.join(cfg.data.data_root, vol_base)
    vol_suffix = cfg.data.volume_suffix or cfg.data.file_extension
    vol_path = bio.resolve_scan_path(vol_base, scan_id, vol_suffix)
    return sitk.ReadImage(vol_path)


def _resample_probs_to_original_space(prob: np.ndarray, processed_spacing, ref_img: sitk.Image) -> np.ndarray:
    """Resample per-channel probabilities onto the original grid with linear
    interpolation, so binarisation happens at full original resolution rather than on
    an already-thresholded mask, which would alias fine vessel boundaries.

    Preprocessing preserves origin and direction and changes only spacing, so `prob`
    shares those with `ref_img`. Crop-based pipelines must be re-embedded first
    (`_embed_crop`), their crop metadata being in processed-grid coordinates.
    """
    channels = []
    for c in range(prob.shape[0]):
        p = sitk.GetImageFromArray(prob[c].transpose(2, 1, 0).astype(np.float32))
        p.SetSpacing(tuple(float(s) for s in processed_spacing))
        p.SetOrigin(ref_img.GetOrigin())
        p.SetDirection(ref_img.GetDirection())
        resampled = sitk.Resample(p, ref_img, sitk.Transform(),
                                  sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        channels.append(sitk.GetArrayFromImage(resampled).transpose(2, 1, 0))
    return np.stack(channels, axis=0).astype(np.float32)


def _embed_crop(prob: np.ndarray, batch: dict, i: int) -> np.ndarray:
    """Place a sub-volume prediction back into the full processed grid, using the
    `crop_origin`/`pre_crop_shape` ADEHTLDataset emits. Returns `prob` unchanged for
    every other method. Voxels outside the crop stay at 0 — the prediction is simply
    not defined there."""
    if "crop_origin" not in batch or "pre_crop_shape" not in batch:
        return prob
    origin = tuple(int(v[i]) for v in batch["crop_origin"])
    full_shape = tuple(int(v[i]) for v in batch["pre_crop_shape"])
    out = np.zeros((prob.shape[0], *full_shape), dtype=prob.dtype)
    sl = tuple(slice(o, o + s) for o, s in zip(origin, prob.shape[1:]))
    out[(slice(None), *sl)] = prob
    return out


def _fuse_connectivity(prob: np.ndarray, cfg: BenchmarkConfig, keep_self: bool) -> np.ndarray:
    """Collapse ADE-HTL's 27 connectivity channels into one binary lumen indicator,
    on the processed grid.

    Returns channel 0 as Fig. 3's fused mask, plus, when `keep_self`, the self-offset
    channel — the only real per-voxel probability in the stack, which is what
    --save-probs needs.

    BOTH halves of Fig. 3's rule run here. The vote must, because it reads +-1-voxel
    neighbours and is only meaningful at the spacing the network predicted at. The
    cut must follow it here because the vote is a COUNT in [0, 27], not a
    probability: interpolating a 0-to-27 count and cutting it at 2 puts the
    iso-surface `1 - 2/v` of the way out from an interior voxel of vote v, so every
    lumen boundary moves outward and thin vessels inflate proportionally more.
    Measured with a perfect connectivity prediction, cutting after the resample gave
    1.50x the true volume at 0.75mm radius against 1.14x at 3mm; the ImageCAS test
    set showed the same signature at mean 1.25x.
    """
    from models.ade_htl import connectivity_votes, SELF_CHANNEL

    params = cfg.postprocessing.params.get("connectivity_fuse", {})
    threshold = params.get("threshold", 0.5)
    min_connections = params.get("min_connections", 2)
    votes = connectivity_votes(prob, threshold)
    out = (votes >= min_connections)[None].astype(np.float32)
    if keep_self:
        out = np.concatenate([out, prob[SELF_CHANNEL:SELF_CHANNEL + 1].astype(np.float32)], axis=0)
    return out


def predict(config_path: str, results_dir: str, save_probs: bool = False,
            split: str = "test", computational_analysis: bool = False,
            overwrite: bool = False):
    """Predict `split` with the checkpoint in `results_dir` and write masks to disk.

    overwrite re-runs every scan and replaces any prediction already on disk, instead of
    the default resume behaviour of skipping scans that already have one.

    computational_analysis measures instead of predicting: scans [1:11] of the split
    are timed (scan 0 runs first as an untimed warm-up, absorbing CUDA/cuDNN autotuning
    that would otherwise inflate the mean), the cache is ignored so every one of them is
    genuinely forwarded, and nothing but computational_cost.json is written."""
    seed_everything()
    cfg = BenchmarkConfig.from_json(config_path)
    cfg.validate()

    if cfg.results_root and not os.path.isabs(results_dir):
        eval_dir = os.path.join(cfg.results_root, results_dir)
    else:
        eval_dir = results_dir
    cfg.evaluation.output_dir = eval_dir

    if is_staged_model(cfg.model.name):
        # Staged models load each stage's checkpoint independently — model.checkpoint
        # and the -r run dir don't apply to them.
        missing = [f"model.{field}" for field, path in
                  (("coarse_checkpoint", cfg.model.coarse_checkpoint),
                   ("dilated_checkpoint", cfg.model.dilated_checkpoint),
                   ("patch_checkpoint_16", cfg.model.patch_checkpoint_16),
                   ("patch_checkpoint_32", cfg.model.patch_checkpoint_32),
                   ("patch_checkpoint_64", cfg.model.patch_checkpoint_64))
                  if not path or not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"{cfg.method_name} is a staged model — set {', '.join(missing)} in the "
                f"config to each stage's trained checkpoint path before running inference."
            )
    else:
        if not cfg.model.checkpoint:
            cfg.model.checkpoint = os.path.join(eval_dir, f"{cfg.method_name}_best.pt")
        if not os.path.exists(cfg.model.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {cfg.model.checkpoint}\n"
                f"-r/--results-dir must point at the run folder printed by train "
                f"(e.g. {cfg.method_name}_2026-07-05-14-30-22-123456), or set model.checkpoint "
                f"explicitly in the config."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] method={cfg.method_name}  device={device}  split={split}")
    print(f"[inference] results_dir={cfg.evaluation.output_dir}")
    if is_staged_model(cfg.model.name):
        print(f"[inference] coarse_checkpoint={cfg.model.coarse_checkpoint}")
        print(f"[inference] dilated_checkpoint={cfg.model.dilated_checkpoint}")
        print(f"[inference] patch_checkpoint_16={cfg.model.patch_checkpoint_16}")
        print(f"[inference] patch_checkpoint_32={cfg.model.patch_checkpoint_32}")
        print(f"[inference] patch_checkpoint_64={cfg.model.patch_checkpoint_64}")
    else:
        print(f"[inference] checkpoint={cfg.model.checkpoint}")

    preprocessing = build_preprocessing(cfg)
    postprocessing = build_postprocessing(cfg)
    test_loader = build_eval_dataloader(cfg, preprocessing=preprocessing, split=split)

    model = build_model(cfg).to(device)
    model.eval()

    # Only test writes to predictions/, the one directory evaluate.py scores. Other
    # splits are for cascades needing an earlier stage's output over their own
    # training scans, and land in a suffixed directory so they cannot overwrite it.
    pred_dir = os.path.join(cfg.evaluation.output_dir,
                            "predictions" if split == "test" else f"predictions_{split}")
    if computational_analysis:
        # Nothing is written under pred_dir in this mode, so don't create it — a cost
        # run must not leave an empty predictions/ that looks like a started run.
        os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
        print(f"[cost] computational analysis: {_COST_WARMUP_SCANS} warm-up scan then "
              f"{_COST_TIMED_SCANS} timed scans; no predictions saved, existing ones "
              f"ignored rather than skipped.")
    else:
        os.makedirs(pred_dir, exist_ok=True)
        if overwrite:
            print(f"[inference] --overwrite: existing predictions in {pred_dir} will be "
                  f"recomputed and replaced.")

    # The timed region is run_inference() only: the forward passes for one scan,
    # including tiling and TTA, but not the method-independent I/O around them. Peak
    # memory is reset here so the model's own weights count towards it.
    scan_times = []
    n_seen = 0  # scans forwarded under --computational_analysis, warm-up included
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in tqdm(test_loader):
        scan_ids = batch["scan_id"]
        volumes = batch["volume"]  # kept on CPU; run_inference moves it per strategy
        spacings = batch["spacing"]

        # The eval loader is batch_size=1, so this skip is an unambiguous per-scan
        # decision: a prediction already on disk is not recomputed.
        pred_paths = [os.path.join(pred_dir, f"{sid}{cfg.data.file_extension}") for sid in scan_ids]
        prob_paths = [os.path.join(pred_dir, f"{sid}_prob{cfg.data.file_extension}") for sid in scan_ids]
        cached = [os.path.exists(p) and (not save_probs or os.path.exists(pp))
                 for p, pp in zip(pred_paths, prob_paths)]
        if overwrite:
            cached = [False] * len(scan_ids)
        if computational_analysis:
            cached = [False] * len(scan_ids)
        elif all(cached):
            continue

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        output = run_inference(model, volumes, cfg, device, scan_id=scan_ids[0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        if computational_analysis:
            # The timed region is over; everything below is saving, which this mode
            # exists to avoid.
            n_seen += 1
            warmup = n_seen <= _COST_WARMUP_SCANS
            if not warmup:
                scan_times.append(elapsed)
            print(f"  {scan_ids[0]}: {elapsed:.2f} s" + (" (warm-up, not counted)" if warmup else ""))
            if n_seen >= _COST_WARMUP_SCANS + _COST_TIMED_SCANS:
                break
            continue

        scan_times.append(elapsed)

        for i, scan_id in enumerate(scan_ids):
            if cached[i]:
                continue

            # Activate BEFORE resampling, so the continuous prediction survives the
            # resolution change instead of an already-binarised mask.
            logits = output["logits"][i]
            prob = torch.sigmoid(logits).numpy()
            processed_spacing = tuple(s[i].item() for s in spacings)

            # ADE-HTL's channels collapse HERE, before the embed+resample: a channel
            # means "this voxel and its neighbour one voxel away ON THIS GRID", and
            # after a resample the +-1 shift would test the wrong neighbourhood while
            # interpolation made "both sides agree" near-trivially true.
            save_channel = 0
            if cfg.output_type == "mask_connectivity_27ch":
                prob = _fuse_connectivity(prob, cfg, keep_self=save_probs)
                save_channel = 1 if save_probs else 0

            # Re-embed BEFORE the resample: crop_origin/pre_crop_shape are
            # processed-grid coordinates. Everything outside the ROI stays zero, which
            # is also how ADE's background suppression takes effect.
            prob = _embed_crop(prob, batch, i)

            # Map to the ORIGINAL scan geometry so evaluate.py's metrics are in true
            # mm and identical for every method regardless of internal resampling.
            ref_img = _load_reference_volume(cfg, scan_id)
            prob_orig = _resample_probs_to_original_space(prob, processed_spacing, ref_img)

            if save_probs:
                # NOT correct for multi-channel outputs like CAS-Net's, where no
                # single channel is simply "lumen". Used for post-hoc soft-vote
                # ensembling across predictions/ directories, which works because they
                # are all on the same original-scan geometry.
                bio.save_prob(prob_orig[save_channel], ref_img, prob_paths[i])

            sample = {"pred": prob_orig, "scan_id": scan_id}
            sample = postprocessing(sample)
            pred_orig = sample["mask"]

            bio.save_mask(pred_orig, ref_img, pred_paths[i])
            print(f"  {scan_id}: saved -> {pred_paths[i]}")

    if not computational_analysis:
        print(f"\nPredictions saved to {pred_dir}")
    _report_computational_cost(cfg, device, scan_times, model)


def _report_computational_cost(cfg: BenchmarkConfig, device, scan_times: list, model):
    """Parameter count, per-scan inference time and peak GPU memory, to
    computational_cost.json. Cached scans are not timed, so a fully-resumed run reports
    nothing rather than a misleading n=0 average."""
    # Counted off the model actually used for this run, so a composite/staged module
    # reports every stage it holds.
    n_params = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cost] #params: {n_params / 1e6:.2f} M ({n_params:,})")

    if not scan_times:
        print("[cost] no scans were run this session (all predictions cached) "
              "-- timing/memory not recorded.")
        return

    times = np.asarray(scan_times, dtype=np.float64)
    cost = {
        "method_name": cfg.method_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "n_params": int(n_params),
        "n_params_m": n_params / 1e6,
        "n_params_trainable": int(n_params_trainable),
        "n_scans": int(times.size),
        "mirror_tta": bool(cfg.data.params.get("inference_mirror_tta", True)),
        "inference_s_mean": float(times.mean()),
        "inference_s_std": float(times.std(ddof=1)) if times.size > 1 else 0.0,
        "inference_s_median": float(np.median(times)),
        "inference_s_min": float(times.min()),
        "inference_s_max": float(times.max()),
        # The first scan absorbs CUDA warmup; reported separately rather than
        # dropped, so the mean stays over the full split.
        "inference_s_mean_excl_first": float(times[1:].mean()) if times.size > 1 else None,
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 1024 ** 3
                           if device.type == "cuda" else None),
        "peak_memory_reserved_gb": (torch.cuda.max_memory_reserved(device) / 1024 ** 3
                                    if device.type == "cuda" else None),
    }

    out_path = os.path.join(cfg.evaluation.output_dir, "computational_cost.json")
    with open(out_path, "w") as f:
        json.dump(cost, f, indent=2)

    mem = f"{cost['peak_memory_gb']:.2f} GB" if cost["peak_memory_gb"] is not None else "n/a (CPU)"
    print(f"[cost] inference: {cost['inference_s_mean']:.2f} +/- {cost['inference_s_std']:.2f} s/scan "
          f"over {cost['n_scans']} scans (median {cost['inference_s_median']:.2f} s)")
    print(f"[cost] peak GPU memory: {mem}   -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to method config JSON")
    parser.add_argument("-r", "--results-dir", required=True,
                        help="Run folder to predict on, as printed by train "
                             "(e.g. cas_net_2026-07-05-14-30-22-123456). Relative to "
                             "results_root in the config, or an absolute path.")
    parser.add_argument("--save-probs", action="store_true",
                        help="Also save each scan's resampled lumen probability map as "
                             "<scan_id>_prob<file_extension> under predictions/, for post-hoc "
                             "soft-vote ensembling (see models/imagecas_baseline/ensemble_vote.py).")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Which split to predict on. Default 'test' -> predictions/, the "
                             "only directory evaluate.py reads. 'train'/'val' write to "
                             "predictions_<split>/ instead, for cascades whose next stage "
                             "consumes this stage's output over its own training scans "
                             "(see models/ade_htl/precompute_ade.py).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute and replace predictions that already exist under "
                             "predictions/ (or predictions_<split>/). Default is to resume: "
                             "scans with a prediction on disk are skipped.")
    parser.add_argument("--computational_analysis", action="store_true",
                        help=f"Measure instead of predict: time scans "
                             f"[{_COST_WARMUP_SCANS}:{_COST_WARMUP_SCANS + _COST_TIMED_SCANS}] "
                             f"of the split ({_COST_TIMED_SCANS} scans, after one untimed "
                             f"warm-up), ignoring any cached predictions, and write only "
                             f"computational_cost.json -- no masks. Use this for the "
                             f"efficiency table (utils/computational_cost_analysis.py), "
                             f"which then needs no fresh run dir.")
    args = parser.parse_args()
    predict(args.config, args.results_dir, save_probs=args.save_probs, split=args.split,
            computational_analysis=args.computational_analysis, overwrite=args.overwrite)
