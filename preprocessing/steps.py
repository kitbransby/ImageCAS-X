import numpy as np
from preprocessing.base import PreprocessingStep
from utils.morphology import dilate_mask


class Resample(PreprocessingStep):
    """Resample volume and mask to a fixed isotropic spacing. Linear for the volume,
    nearest-neighbour for the mask. Params: target_spacing (mm)."""

    def __call__(self, sample: dict) -> dict:
        import SimpleITK as sitk

        target = float(self.params.get("target_spacing", 1.0))
        sitk_img = sample.get("sitk_img")
        sitk_mask = sample.get("sitk_mask")

        if sitk_img is None:
            raise KeyError("Resample requires sample['sitk_img'] — pass sitk images through the pipeline.")

        orig_spacing = sitk_img.GetSpacing()
        orig_size = sitk_img.GetSize()
        new_size = [int(round(orig_size[i] * orig_spacing[i] / target)) for i in range(3)]

        img_resampler = sitk.ResampleImageFilter()
        img_resampler.SetOutputSpacing([target] * 3)
        img_resampler.SetSize(new_size)
        img_resampler.SetOutputDirection(sitk_img.GetDirection())
        img_resampler.SetOutputOrigin(sitk_img.GetOrigin())
        img_resampler.SetInterpolator(sitk.sitkLinear)
        img_resampler.SetDefaultPixelValue(0)
        sitk_img_r = img_resampler.Execute(sitk_img)

        sample["volume"] = sitk.GetArrayFromImage(sitk_img_r).transpose(2, 1, 0).astype(np.float32)
        sample["sitk_img"] = sitk_img_r
        sample["spacing"] = (target, target, target)

        if sitk_mask is not None:
            # Resample onto the volume's grid as reference, not independently from
            # the mask's own header: two independent resamples agree voxel-for-voxel
            # only if the headers match exactly.
            mask_resampler = sitk.ResampleImageFilter()
            mask_resampler.SetReferenceImage(sitk_img_r)
            mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_resampler.SetDefaultPixelValue(0)
            sitk_mask_r = mask_resampler.Execute(sitk_mask)
            sample["mask"] = sitk.GetArrayFromImage(sitk_mask_r).transpose(2, 1, 0).astype(np.uint8)
            sample["sitk_mask"] = sitk_mask_r

        return sample


class ResampleToShape(PreprocessingStep):
    """Resample to a fixed absolute voxel shape rather than a spacing, for the
    ImageCAS coarse stages. Params: target_shape ([X, Y, Z])."""

    def __call__(self, sample: dict) -> dict:
        import SimpleITK as sitk

        target_shape = self.params.get("target_shape")
        if target_shape is None:
            raise ValueError("ResampleToShape requires 'target_shape' in params, e.g. [128, 128, 64].")

        sitk_img = sample.get("sitk_img")
        if sitk_img is None:
            raise KeyError("ResampleToShape requires sample['sitk_img'].")

        orig_size = sitk_img.GetSize()
        orig_spacing = sitk_img.GetSpacing()
        tx, ty, tz = int(target_shape[0]), int(target_shape[1]), int(target_shape[2])
        new_spacing = [orig_size[i] * orig_spacing[i] / target_shape[i] for i in range(3)]

        img_resampler = sitk.ResampleImageFilter()
        img_resampler.SetOutputSpacing(new_spacing)
        img_resampler.SetSize([tx, ty, tz])
        img_resampler.SetOutputDirection(sitk_img.GetDirection())
        img_resampler.SetOutputOrigin(sitk_img.GetOrigin())
        img_resampler.SetInterpolator(sitk.sitkLinear)
        img_resampler.SetDefaultPixelValue(0)
        sitk_img_r = img_resampler.Execute(sitk_img)

        sample["volume"] = sitk.GetArrayFromImage(sitk_img_r).transpose(2, 1, 0).astype(np.float32)
        sample["sitk_img"] = sitk_img_r
        sample["spacing"] = tuple(new_spacing)

        sitk_mask = sample.get("sitk_mask")
        if sitk_mask is not None:
            mask_resampler = sitk.ResampleImageFilter()
            mask_resampler.SetReferenceImage(sitk_img_r)
            mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_resampler.SetDefaultPixelValue(0)
            sitk_mask_r = mask_resampler.Execute(sitk_mask)
            sample["mask"] = sitk.GetArrayFromImage(sitk_mask_r).transpose(2, 1, 0).astype(np.uint8)
            sample["sitk_mask"] = sitk_mask_r

        return sample


class DilateMask(PreprocessingStep):
    """Dilate the GT mask with a spherical structuring element, so a net trained on
    it predicts oversized vessels that stay connected through skeletonisation.

    Must run BEFORE the resample steps, since those operate on sample['sitk_mask']
    rather than sample['mask'] — this keeps both in sync. Radius is a physical
    distance, so dilating after a downsample would inflate it. No-op without a mask.
    """

    def __call__(self, sample: dict) -> dict:
        if "mask" not in sample:
            return sample

        import SimpleITK as sitk

        radius = int(self.params.get("radius", 5))
        mask = sample["mask"]
        dilated = dilate_mask(mask, radius).astype(mask.dtype)
        sample["mask"] = dilated

        sitk_mask = sample.get("sitk_mask")
        if sitk_mask is not None:
            dilated_img = sitk.GetImageFromArray(dilated.transpose(2, 1, 0))
            dilated_img.CopyInformation(sitk_mask)
            sample["sitk_mask"] = dilated_img

        return sample


class NormaliseToRange(PreprocessingStep):
    """Clip HU to [hu_min, hu_max] and scale linearly to [out_min, out_max]."""

    def __call__(self, sample: dict) -> dict:
        hu_min = self.params.get("hu_min", -200)
        hu_max = self.params.get("hu_max", 400)
        out_min = self.params.get("out_min", 0.0)
        out_max = self.params.get("out_max", 1.0)
        vol = sample["volume"].astype(np.float32)
        vol = np.clip(vol, hu_min, hu_max)
        vol = (vol - hu_min) / (hu_max - hu_min)
        sample["volume"] = vol * (out_max - out_min) + out_min
        return sample
