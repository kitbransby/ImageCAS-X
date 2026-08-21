"""Post-hoc soft-vote ensemble for the ImageCAS baseline.

Combines the same 4 members ImageCASBaseline votes over, but averages each member's
saved probability map instead of hard-voting their thresholded masks. --coarse-dir is
optional; omitting it ensembles only the 3 patch scales.

DEVIATION from the paper's stated majority voting, not a bug fix. A hard vote on 4
masks has a silent tie-break — an exact 2-vs-2 split resolves to background with no
visibility into how often that happens — whereas averaging degrades gracefully. Treat
ImageCASBaseline's own forward() as the paper-faithful default and this as an
optional comparison. Dropping --coarse-dir is a second, independent deviation: 3
members give a well-defined majority with no possible tie, but Stage 1 is the only
member predicting over the whole volume independently of Stage 2's skeleton, so
losing it likely costs recall wherever that chain misses part of the tree.

Prerequisite: run inference.py --save-probs on each member config included --
    python -m inference -c configs/imagecas_stage1_coarse.json -r <run> --save-probs   # optional, see --coarse-dir
    python -m inference -c configs/imagecas_stage3_patch_16.json -r <run> --save-probs
    python -m inference -c configs/imagecas_stage3_patch_32.json -r <run> --save-probs
    python -m inference -c configs/imagecas_stage3_patch_64.json -r <run> --save-probs
The patch runs go through inference.py's skeleton-centred branch, which needs
centers_dir to already hold a {scan_id}.npy for every test scan.

Usage:
    python -m models.imagecas_baseline.ensemble_vote \
        [--coarse-dir results/imagecas_stage1_coarse_.../predictions] \
        --patch16-dir results/imagecas_stage3_patch_16_.../predictions \
        --patch32-dir results/imagecas_stage3_patch_32_.../predictions \
        --patch64-dir results/imagecas_stage3_patch_64_.../predictions \
        --out-dir results/imagecas_ensemble_softvote/predictions \
        [--threshold 0.5] [-c configs/imagecas_inference.json]

evaluate.py then scores the combined masks with any imagecas config:
    python -m evaluate -c configs/imagecas_inference.json -r results/imagecas_ensemble_softvote
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk

from utils.config import BenchmarkConfig

_PROB_SUFFIX = "_prob"


def _scan_ids(prob_dir: str, extension: str) -> set:
    suffix = f"{_PROB_SUFFIX}{extension}"
    ids = set()
    for path in sorted(glob.glob(os.path.join(prob_dir, f"*{suffix}"))):
        name = os.path.basename(path)
        ids.add(name[: -len(suffix)])
    return ids


def ensemble(patch16_dir: str, patch32_dir: str, patch64_dir: str, out_dir: str,
            coarse_dir: str | None = None, extension: str = ".nii.gz",
            threshold: float = 0.5):
    member_dirs = {
        "patch_16": patch16_dir,
        "patch_32": patch32_dir,
        "patch_64": patch64_dir,
    }
    if coarse_dir:
        member_dirs["coarse"] = coarse_dir
    n_members = len(member_dirs)

    ids_per_member = {name: _scan_ids(d, extension) for name, d in member_dirs.items()}
    for name, ids in ids_per_member.items():
        print(f"  {name}: {len(ids)} '*{_PROB_SUFFIX}{extension}' file(s) found in {member_dirs[name]}")

    common_ids = set.intersection(*ids_per_member.values())
    for name, ids in ids_per_member.items():
        missing = ids - common_ids  # present here but not in every other member
        if missing:
            print(f"  [warn] {name}: {len(missing)} scan(s) not shared across all "
                  f"{n_members} members, skipped from ensemble: {sorted(missing)}")

    if not common_ids:
        empty = [name for name, ids in ids_per_member.items() if not ids]
        if empty:
            raise RuntimeError(
                f"{', '.join(empty)} -- 0 probability files found. Check the path(s) "
                f"point at the run's predictions/ directory (not the run dir itself), "
                f"and that inference.py was actually invoked with --save-probs for "
                f"{'that' if len(empty) == 1 else 'those'} run(s)."
            )
        raise RuntimeError(
            f"Every member directory has probability files, but no single scan_id is "
            f"present in all {n_members} of them -- see the [warn] lines above for which "
            f"scans are missing from which member. This usually means the runs used "
            f"different test splits/exclude lists, or inference.py --save-probs was run "
            f"on top of a partially-completed (resumed) prediction run."
        )

    os.makedirs(out_dir, exist_ok=True)
    for scan_id in sorted(common_ids):
        probs = []
        ref_img = None
        for name, d in member_dirs.items():
            path = os.path.join(d, f"{scan_id}{_PROB_SUFFIX}{extension}")
            img = sitk.ReadImage(path)
            if ref_img is None:
                ref_img = img
            probs.append(sitk.GetArrayFromImage(img).astype(np.float32))  # [z,y,x]

        avg = np.mean(probs, axis=0)
        mask = (avg > threshold).astype(np.uint8)

        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(ref_img)
        out_path = os.path.join(out_dir, f"{scan_id}{extension}")
        sitk.WriteImage(out, out_path)
        print(f"  {scan_id}: averaged {len(probs)} members -> {out_path}")

    print(f"[ensemble] done -- wrote {len(common_ids)} scan(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coarse-dir", default=None,
                        help="predictions/ dir from the imagecas_stage1_coarse.json run. "
                             "Optional -- omit to ensemble only the 3 Stage 3 patch scales "
                             "(see the module docstring for the tradeoff).")
    parser.add_argument("--patch16-dir", required=True,
                        help="predictions/ dir from the imagecas_stage3_patch_16.json run")
    parser.add_argument("--patch32-dir", required=True,
                        help="predictions/ dir from the imagecas_stage3_patch_32.json run")
    parser.add_argument("--patch64-dir", required=True,
                        help="predictions/ dir from the imagecas_stage3_patch_64.json run")
    parser.add_argument("--out-dir", required=True,
                        help="Output predictions/ dir for the ensembled masks")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold applied to the averaged probability (default 0.5)")
    parser.add_argument("-c", "--config", default="configs/imagecas_inference.json",
                        help="Any imagecas config, used only to read data.file_extension "
                             "(default configs/imagecas_inference.json)")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)
    ensemble(args.patch16_dir, args.patch32_dir, args.patch64_dir, args.out_dir,
            coarse_dir=args.coarse_dir, extension=cfg.data.file_extension,
            threshold=args.threshold)
