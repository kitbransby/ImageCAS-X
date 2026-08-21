"""Convert the ImageCAS dataset (this benchmark's on-disk layout, driven by
`configs/pipeline.json`) into nnU-Net v2's raw dataset format:
`nnUNet_raw/DatasetXXX_Name/{imagesTr,labelsTr,imagesTs,dataset.json}`.

nnU-Net does its own resampling and normalisation, so volumes are copied as-is. GT
masks are binarised to the same lumen-vs-background scheme every other method trains
on, keeping nnU-Net comparable, and are resampled onto the volume's own grid first —
raw volume/mask headers are not always identical in this dataset, and silently
misaligning image/label pairs would poison training.

train_ids + val_ids go to imagesTr/labelsTr, since nnU-Net manages its own held-out
folds; test_ids to imagesTs.

Usage (only the shared `data` block is read, so any method config works):
    python -m models.nnunet.data_prep -c configs/cas_net.json \
        --nnunet-data-folder /path/to/nnunet_data [--dataset-id 1] [--dataset-name ImageCAS]

Then, outside this script:
    export nnUNet_raw=/path/to/nnunet_data/nnUNet_raw
    export nnUNet_preprocessed=/path/to/nnunet_data/nnUNet_preprocessed
    export nnUNet_results=/path/to/nnunet_data/nnUNet_results
    nnUNetv2_plan_and_preprocess -d <dataset-id> -pl nnUNetPlannerResEncM
"""
import argparse
import json
import os
import shutil

import numpy as np
import SimpleITK as sitk

from utils.config import BenchmarkConfig
from utils import io as bio


def _resolve(data_root: str, rel_or_abs: str) -> str:
    if not rel_or_abs or os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(data_root, rel_or_abs)


def _write_label(mask_path: str, vol_path: str, dst_path: str,
                  background_label: int):
    """Binarise and resample onto the volume's own grid, so the label file aligns
    with the copied image voxel-for-voxel."""
    ref = sitk.ReadImage(vol_path)
    mask_img = sitk.ReadImage(mask_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    mask_img_r = resampler.Execute(mask_img)

    mask = sitk.GetArrayFromImage(mask_img_r).transpose(2, 1, 0).astype(np.uint8)
    mask_bin = bio.binarise_lumen(mask, background_label=background_label)
    bio.save_mask(mask_bin, ref, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                         help="Path to a benchmark config JSON")
    parser.add_argument("--nnunet-data-folder", required=True,
                         help="Output root; nnUNet_raw/nnUNet_preprocessed/nnUNet_results are created under it")
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="ImageCAS")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-copy cases whose outputs already exist")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_json(args.config)

    train_ids = sorted(set(cfg.data.train_ids) | set(cfg.data.val_ids))
    test_ids = sorted(set(cfg.data.test_ids))
    if not train_ids:
        raise ValueError(
            "No scan IDs found across train/val splits — check data.filelist_dir "
            f"in {args.config}."
        )

    vol_suffix = cfg.data.volume_suffix or cfg.data.file_extension
    mask_suffix = cfg.data.mask_suffix or cfg.data.file_extension
    vol_dir = _resolve(cfg.data.data_root, cfg.data.volume_dir)
    mask_dir = _resolve(cfg.data.data_root, cfg.data.gt_mask_dir)
    background_label = cfg.data.params.get("background_label", bio.LUMEN_BACKGROUND_LABEL)

    dataset_dir_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    nnunet_raw = os.path.join(args.nnunet_data_folder, "nnUNet_raw")
    dataset_dir = os.path.join(nnunet_raw, dataset_dir_name)
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")

    for d in (images_tr, labels_tr, images_ts):
        os.makedirs(d, exist_ok=True)
    # nnU-Net expects these as siblings even though only nnUNet_raw is populated here.
    os.makedirs(os.path.join(args.nnunet_data_folder, "nnUNet_preprocessed"), exist_ok=True)
    os.makedirs(os.path.join(args.nnunet_data_folder, "nnUNet_results"), exist_ok=True)

    print(f"[nnunet_data_prep] {len(train_ids)} train+val scans -> imagesTr/labelsTr, "
          f"{len(test_ids)} test scans -> imagesTs")
    print(f"[nnunet_data_prep] volumes: {vol_dir}  masks: {mask_dir}")
    print(f"[nnunet_data_prep] dataset dir: {dataset_dir}")

    n_copied, n_skipped = 0, 0
    for scan_id in train_ids:
        img_dst = os.path.join(images_tr, f"{scan_id}_0000.nii.gz")
        lbl_dst = os.path.join(labels_tr, f"{scan_id}.nii.gz")
        if not args.overwrite and os.path.exists(img_dst) and os.path.exists(lbl_dst):
            n_skipped += 1
            continue
        vol_path = bio.resolve_scan_path(vol_dir, scan_id, vol_suffix)
        mask_path = bio.resolve_scan_path(mask_dir, scan_id, mask_suffix)
        shutil.copy2(vol_path, img_dst)
        _write_label(mask_path, vol_path, lbl_dst, background_label)
        n_copied += 1

    n_ts_copied, n_ts_skipped = 0, 0
    for scan_id in test_ids:
        img_dst = os.path.join(images_ts, f"{scan_id}_0000.nii.gz")
        if not args.overwrite and os.path.exists(img_dst):
            n_ts_skipped += 1
            continue
        vol_path = bio.resolve_scan_path(vol_dir, scan_id, vol_suffix)
        shutil.copy2(vol_path, img_dst)
        n_ts_copied += 1

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "lumen": 1},
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[nnunet_data_prep] done. imagesTr/labelsTr: {n_copied} copied, {n_skipped} skipped "
          f"(already existed). imagesTs: {n_ts_copied} copied, {n_ts_skipped} skipped.")
    print(f"[nnunet_data_prep] export nnUNet_raw={nnunet_raw}")
    print(f"[nnunet_data_prep] export nnUNet_preprocessed={os.path.join(args.nnunet_data_folder, 'nnUNet_preprocessed')}")
    print(f"[nnunet_data_prep] export nnUNet_results={os.path.join(args.nnunet_data_folder, 'nnUNet_results')}")
    print(f"[nnunet_data_prep] then: nnUNetv2_plan_and_preprocess -d {args.dataset_id} -pl nnUNetPlannerResEncM")


if __name__ == "__main__":
    main()
