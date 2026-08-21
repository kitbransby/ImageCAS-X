"""nnU-Net: self-configuring medical image segmentation framework.

Source: https://github.com/MIC-DKFZ/nnUNet
Paper: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based
       biomedical image segmentation", Nature Methods 18 (2021) 203-211.

nnU-Net is a full training framework, not a single model class: it auto-configures
patch size, batch size, depth, augmentation and post-processing from dataset
statistics, so it cannot be wrapped as a BaseLumenModel here.

Setup:
    pip install nnunetv2
    export nnUNet_raw=/path/to/nnunet_raw
    export nnUNet_preprocessed=/path/to/nnunet_preprocessed
    export nnUNet_results=/path/to/nnunet_results

    # Convert dataset to nnU-Net format, then:
    nnUNetv2_plan_and_preprocess -d <DATASET_ID> -pl nnUNetPlannerResEncM
    nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD>  # for each fold 0-4

Inference:
    nnUNetv2_predict -i <input_dir> -o <output_dir> -d <DATASET_ID> -c 3d_fullres -f all

To score the output here, place the predicted masks under <run_dir>/predictions/ and
run `python -m evaluate -c configs/nnunet.json -r <run_dir>`. The NSDT Soft-clDice
variant is trained in https://github.com/kitbransby/nnUNet_NSDT_clDice and its masks
are likewise consumed as files.
"""
