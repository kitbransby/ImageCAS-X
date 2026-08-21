"""Generate ImageCAS patch-centre files from the trained Stage 2 (dilated) net.

Runs the trained Stage 2 net over each volume and saves the skeleton of its
further-dilated prediction as per-scan {scan_id}.npy patch centres, which
PatchDataset consumes to train the Stage 3 nets.

Must point at the Stage 2 (dilated) checkpoint, not Stage 1's, which is trained on
the undilated mask and feeds the final majority vote instead.

Recipe: sigmoid > 0.5 -> dilate again ("at test the output is further dilated") ->
drop small components -> skeletonise -> argwhere. DEVIATION: the paper keeps the 2
largest components at that step; this repo drops components below a size floor
instead, since keeping exactly 2 discards genuine distal branches whenever the coarse
prediction leaves one disconnected.

The Stage 2 config works at 128x128x64, so the skeleton is extracted in THAT grid,
not the 0.5mm one Stage 3 crops from. Centres are rescaled before saving: the resize
preserves origin and direction and changes only spacing, so
`idx_target = idx_128 * spacing_128 / target_spacing` maps a voxel index exactly.

Usage:
    python -m models.imagecas_baseline.stage3_generate_centers \
        -c configs/imagecas_stage2_coarse_dilated.json \
        --coarse-checkpoint results/imagecas_stage2_coarse_dilated/imagecas_stage2_coarse_dilated_best.pt \
        --out-dir /path/to/centers \
        --split train        # train | val | test | all  (default: train)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.config import BenchmarkConfig
from utils import io as bio
from preprocessing.pipeline import build_preprocessing
from models.imagecas_baseline import (
    UNet3DCoarse, _dilate_mask, _remove_small_components, _extract_skeleton,
)


def _split_ids(cfg: BenchmarkConfig, split: str):
    mapping = {
        "train": cfg.data.train_ids,
        "val": cfg.data.val_ids,
        "test": cfg.data.test_ids,
    }
    if split == "all":
        return list(cfg.data.train_ids) + list(cfg.data.val_ids) + list(cfg.data.test_ids)
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}'. Options: train | val | test | all.")
    return list(mapping[split])


def _resolve_dir(cfg: BenchmarkConfig, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(cfg.data.data_root, rel_or_abs)


def _volume_path(cfg: BenchmarkConfig, scan_id: str) -> str:
    base = _resolve_dir(cfg, cfg.data.volume_dir)
    suffix = cfg.data.volume_suffix or cfg.data.file_extension
    return bio.resolve_scan_path(base, scan_id, suffix)


def _build_coarse_net(cfg: BenchmarkConfig, coarse_ckpt: str, device):
    p = cfg.model.params
    in_ch = p.get("in_channels", 1)
    base_ch = p.get("base_ch_coarse", p.get("base_ch", 32))
    net = UNet3DCoarse(in_channels=in_ch, base_ch=base_ch).to(device)
    net.load_weights(coarse_ckpt)
    net.eval()
    return net


def generate(config_path: str, coarse_ckpt: str, out_dir: str,
             split: str = "train", overwrite: bool = False, limit: int | None = None,
             target_spacing: float = 0.5):
    cfg = BenchmarkConfig.from_json(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen_centers] method={cfg.method_name}  split={split}  device={device}")

    # Resolve the checkpoint (CLI overrides config). This should be the Stage 2
    # (dilated) checkpoint — see module docstring — not Stage 1's plain coarse one.
    coarse_ckpt = coarse_ckpt or cfg.model.coarse_checkpoint or cfg.model.checkpoint
    if not coarse_ckpt:
        raise ValueError(
            "No checkpoint given. Pass --coarse-checkpoint or set "
            "model.coarse_checkpoint in the config. Train Stage 2 first "
            "(configs/imagecas_stage2_coarse_dilated.json, model.name='imagecas_coarse')."
        )

    # Resolve the output directory (CLI overrides config's centers_dir/skeleton_dir).
    out_dir = out_dir or cfg.data.params.get("centers_dir") or cfg.data.params.get("skeleton_dir")
    if not out_dir:
        raise ValueError(
            "No output directory. Pass --out-dir, or set data.params.centers_dir "
            "in the config (this is what PatchDataset reads)."
        )
    os.makedirs(out_dir, exist_ok=True)

    preprocessing = build_preprocessing(cfg)
    net = _build_coarse_net(cfg, coarse_ckpt, device)

    coarse_shape = tuple(cfg.model.params.get("coarse_input_shape", [128, 128, 64]))
    use_dilation = cfg.model.params.get("use_dilation", True)
    dilation_radius = cfg.model.params.get("dilation_radius", 5)
    min_component_size = cfg.model.params.get("min_component_size", 100)

    ids = _split_ids(cfg, split)
    if limit is not None:
        ids = ids[:limit]

    n_written = 0
    for scan_id in ids:
        out_path = os.path.join(out_dir, f"{scan_id}.npy")
        if os.path.exists(out_path) and not overwrite:
            print(f"  skip (exists): {scan_id}")
            continue

        # Preprocess the volume exactly as Stage 2 training does (128-grid) --
        # centres get rescaled into PatchDataset's actual grid further below.
        volume, sitk_img = bio.load_volume(_volume_path(cfg, scan_id))
        sample = {
            "volume": volume.astype(np.float32),
            "scan_id": scan_id,
            "spacing": sitk_img.GetSpacing(),
            "sitk_img": sitk_img,
        }
        sample = preprocessing(sample)
        vol_proc = sample["volume"]
        # Per-axis spacing of the 128-grid ResampleToShape just resized onto
        # (anisotropic, scan-specific) -- needed to rescale centres into the
        # grid PatchDataset actually crops from; see module docstring.
        spacing_128 = np.asarray(sample["spacing"], dtype=np.float64)

        # Stage 2 (dilated) net inference: resize -> net -> upsample back.
        with torch.no_grad():
            x = torch.from_numpy(vol_proc).float()[None, None].to(device)
            x_small = F.interpolate(x, size=coarse_shape, mode="trilinear", align_corners=False)
            logits_small = net(x_small)["logits"]
            logits = F.interpolate(logits_small, size=vol_proc.shape,
                                   mode="trilinear", align_corners=False)
            mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

        # Dilate -> drop sub-min_component_size CCs -> skeletonise (paper order,
        # except the paper's keep-top-2-components step is replaced by this
        # repo's single component rule -- see _remove_small_components).
        src = _dilate_mask(mask, dilation_radius) if use_dilation else mask
        src = _remove_small_components(src, min_component_size)
        skeleton = _extract_skeleton(src)
        centers_128 = np.argwhere(skeleton).astype(np.float64)  # (N, 3) voxel coords, 128-grid

        # Rescale 128-grid voxel indices -> target (0.5mm-grid) voxel indices.
        # Same origin/direction on both grids (ResampleToShape/Resample both
        # preserve the original sitk_img's), so this is an exact per-axis
        # scale by the two grids' spacing ratio -- see module docstring.
        centers = np.round(centers_128 * (spacing_128 / target_spacing))
        centers = np.maximum(centers, 0).astype(np.int32)

        if len(centers) == 0:
            print(f"  [warn] {scan_id}: empty skeleton (coarse prediction too sparse?) — "
                  f"writing empty centre file.")
        np.save(out_path, centers)
        n_written += 1
        print(f"  {scan_id}: {len(centers)} centres -> {out_path}")

    print(f"[gen_centers] done — wrote {n_written} centre file(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ImageCAS patch-centre .npy files.")
    parser.add_argument("-c", "--config", required=True, help="Path to the ImageCAS config JSON")
    parser.add_argument("--coarse-checkpoint", default="",
                        help="Trained coarse-net checkpoint (overrides config)")
    parser.add_argument("--out-dir", default="",
                        help="Output dir for {scan_id}.npy (overrides data.params.centers_dir)")
    parser.add_argument("--split", default="train", help="train | val | test | all")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing files")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scans")
    parser.add_argument("--target-spacing", type=float, default=0.5,
                        help="Isotropic spacing (mm) of the grid Stage 3's PatchDataset actually "
                             "crops from -- must match pipeline.json's preprocessing.params.resample."
                             "target_spacing (default 0.5) unless a Stage 3 config overrides it.")
    args = parser.parse_args()

    generate(args.config, args.coarse_checkpoint, args.out_dir,
             split=args.split, overwrite=args.overwrite, limit=args.limit,
             target_spacing=args.target_spacing)
