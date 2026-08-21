"""ImageCAS Baseline: 3-stage coarse-to-fine coronary artery segmentation.

Zeng et al., CMIG 109 (2023). Source:
https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT

5 trained checkpoints:
  Stage 1  coarse 3D U-Net on a 128x128x64 volume; votes.
  Stage 2  same grid but trained on a DILATED GT mask with the Eq. 2 loss. Supplies
           Stage 3's skeleton patch centres only; never votes.
  Stage 3  three independently trained 3D U-Net++ nets, one per patch scale.
Final mask is a majority vote over Stage 1 and the three patch scales.

Training is staged, under model names "imagecas_coarse" and "imagecas_patch";
"imagecas_baseline" is for full-pipeline inference with all 5 checkpoints loaded.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseLumenModel
from models.registry import register_model
from utils.morphology import dilate_mask as _dilate_mask


def _extract_skeleton(mask: np.ndarray) -> np.ndarray:
    """Binary mask -> binary skeleton via 3D surface thinning (Lee et al. 1994)."""
    from skimage.morphology import skeletonize
    return skeletonize(mask.astype(bool)).astype(np.uint8)


def _remove_small_components(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Drop foreground components smaller than `min_size` voxels.

    DEVIATION: the paper's skeleton recipe keeps the 2 largest components. Hardcoding
    how many to expect discards genuine distal branches whenever a prediction leaves
    one disconnected, and those are unrecoverable downstream — no centre is placed
    there, so no patch net ever looks at it.
    """
    from scipy.ndimage import label
    labeled, n_labels = label(mask)
    if n_labels == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    keep = np.zeros(counts.shape[0], dtype=bool)
    keep[1:] = counts[1:] >= min_size   # index 0 is background
    return keep[labeled].astype(mask.dtype)


def _extract_patch(volume: np.ndarray, center: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract a cubic patch of edge `patch_size` centred at `center`, zero-padded
    at boundaries."""
    half = patch_size // 2
    cx, cy, cz = int(round(center[0])), int(round(center[1])), int(round(center[2]))

    x0, x1 = cx - half, cx + half + 1
    y0, y1 = cy - half, cy + half + 1
    z0, z1 = cz - half, cz + half + 1

    pad = [
        (max(0, -x0), max(0, x1 - volume.shape[0])),
        (max(0, -y0), max(0, y1 - volume.shape[1])),
        (max(0, -z0), max(0, z1 - volume.shape[2])),
    ]
    vx0, vx1 = max(0, x0), min(volume.shape[0], x1)
    vy0, vy1 = max(0, y0), min(volume.shape[1], y1)
    vz0, vz1 = max(0, z0), min(volume.shape[2], z1)

    patch = volume[vx0:vx1, vy0:vy1, vz0:vz1]
    if any(p[0] > 0 or p[1] > 0 for p in pad):
        patch = np.pad(patch, pad, mode="constant", constant_values=0)
    return patch[:patch_size, :patch_size, :patch_size]


def _assemble_predictions(preds: list, centers: np.ndarray,
                          patch_size: int, vol_shape: tuple,
                          threshold: float | None = 0.5) -> np.ndarray:
    """Average overlapping patch predictions into a full-volume prediction.
    threshold=None returns raw probabilities, which is what inference.py's standalone
    Stage 3 branch needs to keep run_inference's logits contract meaningful."""
    accum = np.zeros(vol_shape, dtype=np.float32)
    count = np.zeros(vol_shape, dtype=np.float32)
    half = patch_size // 2

    for pred, center in zip(preds, centers):
        cx, cy, cz = int(round(center[0])), int(round(center[1])), int(round(center[2]))
        # Matches _extract_patch's retained footprint: it truncates its raw
        # patch_size+1 extraction, so [c-half, c+half+1) would be one voxel too wide.
        x0, x1 = cx - half, cx - half + patch_size
        y0, y1 = cy - half, cy - half + patch_size
        z0, z1 = cz - half, cz - half + patch_size

        px0 = max(0, -x0); vx0 = max(0, x0); vx1 = min(vol_shape[0], x1)
        py0 = max(0, -y0); vy0 = max(0, y0); vy1 = min(vol_shape[1], y1)
        pz0 = max(0, -z0); vz0 = max(0, z0); vz1 = min(vol_shape[2], z1)
        px1, py1, pz1 = px0 + (vx1 - vx0), py0 + (vy1 - vy0), pz0 + (vz1 - vz0)

        accum[vx0:vx1, vy0:vy1, vz0:vz1] += pred[px0:px1, py0:py1, pz0:pz1]
        count[vx0:vx1, vy0:vy1, vz0:vz1] += 1

    prob = accum / np.maximum(count, 1)
    if threshold is None:
        return prob.astype(np.float32)
    return (prob > threshold).astype(np.uint8)


def _majority_vote(masks: list) -> np.ndarray:
    """Majority vote across a list of binary masks."""
    stacked = np.stack(masks, axis=0).astype(np.int32)
    return (stacked.sum(axis=0) > len(masks) / 2).astype(np.uint8)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _upsample_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        nn.Conv3d(in_ch, out_ch, 1, bias=False),
    )


@register_model("imagecas_coarse")
class UNet3DCoarse(BaseLumenModel):
    """Coarse 3D U-Net, shared by Stage 1 (plain GT mask, Dice) and Stage 2 (dilated
    GT mask, weighted similarity loss). Also reused by ADE-HTL's coarse stage."""

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.pool = nn.MaxPool3d(2)

        self.enc0 = ConvBlock3D(in_channels, ch[0])
        self.enc1 = ConvBlock3D(ch[0], ch[1])
        self.enc2 = ConvBlock3D(ch[1], ch[2])
        self.bottleneck = ConvBlock3D(ch[2], ch[3])

        self.up2 = _upsample_conv(ch[3], ch[2])
        self.dec2 = ConvBlock3D(ch[2] * 2, ch[2])
        self.up1 = _upsample_conv(ch[2], ch[1])
        self.dec1 = ConvBlock3D(ch[1] * 2, ch[1])
        self.up0 = _upsample_conv(ch[1], ch[0])
        self.dec0 = ConvBlock3D(ch[0] * 2, ch[0])

        # Raw logit; sigmoid is applied downstream.
        self.head = nn.Conv3d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> dict:
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return {"logits": self.head(d0)}


@register_model("imagecas_patch")
class UNetPlusPlus3D(BaseLumenModel):
    """3D U-Net++ with dense skips and deep supervision, one checkpoint per patch
    scale — this class does not know its own scale, which is purely a property of the
    config that trains it. Training returns the head list, inference their mean."""

    def __init__(self, in_channels: int = 1, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        ch = [base_ch * (2 ** i) for i in range(depth)]
        self.pool = nn.MaxPool3d(2)

        self.enc = nn.ModuleList([ConvBlock3D(in_channels, ch[0])])
        for l in range(1, depth):
            self.enc.append(ConvBlock3D(ch[l - 1], ch[l]))

        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for j in range(1, depth):
            dec_row = nn.ModuleList()
            up_row = nn.ModuleList()
            for l in range(depth - j):
                dec_row.append(ConvBlock3D((j + 1) * ch[l], ch[l]))
                up_row.append(_upsample_conv(ch[l + 1], ch[l]))
            self.dec.append(dec_row)
            self.up.append(up_row)

        # Raw logits per deep-supervision head.
        self.heads = nn.ModuleList([
            nn.Conv3d(ch[0], 1, 1)
            for _ in range(depth - 1)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        nodes = {}
        nodes[(0, 0)] = self.enc[0](x)
        for l in range(1, self.depth):
            nodes[(l, 0)] = self.enc[l](self.pool(nodes[(l - 1, 0)]))

        for j in range(1, self.depth):
            for l in range(self.depth - j):
                skips = [nodes[(l, k)] for k in range(j)]
                up = self.up[j - 1][l](nodes[(l + 1, j - 1)])
                nodes[(l, j)] = self.dec[j - 1][l](torch.cat(skips + [up], dim=1))

        outputs = [self.heads[j - 1](nodes[(0, j)]) for j in range(1, self.depth)]

        if self.training:
            return {"logits": outputs}
        return {"logits": torch.mean(torch.stack(outputs, dim=0), dim=0)}


@register_model("imagecas_baseline")
class ImageCASBaseline(BaseLumenModel):
    """The full 3-stage pipeline, for inference once all 5 checkpoints are loaded.

    forward() runs Stage 1, then Stage 2 -> dilate -> drop small CCs -> skeletonise
    -> centres, then the three patch scales at those centres, then a majority vote
    over Stage 1 and the patch masks. In training mode it returns the Stage 1 output
    only, so that net can be fine-tuned with gradients.
    """

    def __init__(self, in_channels: int = 1, base_ch_coarse: int = 32,
                 base_ch_dilated: int = 32, base_ch_patch: int = 32,
                 patch_sizes: tuple = (16, 32, 64),
                 use_dilation: bool = True, dilation_radius: int = 5,
                 min_component_size: int = 100,
                 coarse_input_shape: tuple = (128, 128, 64)):
        super().__init__()
        self.coarse_net = UNet3DCoarse(in_channels=in_channels, base_ch=base_ch_coarse)
        self.dilated_net = UNet3DCoarse(in_channels=in_channels, base_ch=base_ch_dilated)
        self.patch_net_16 = UNetPlusPlus3D(in_channels=in_channels, base_ch=base_ch_patch)
        self.patch_net_32 = UNetPlusPlus3D(in_channels=in_channels, base_ch=base_ch_patch)
        self.patch_net_64 = UNetPlusPlus3D(in_channels=in_channels, base_ch=base_ch_patch)
        self.patch_sizes = list(patch_sizes)
        self.use_dilation = use_dilation
        self.dilation_radius = dilation_radius
        self.min_component_size = min_component_size
        self.coarse_input_shape = tuple(coarse_input_shape)
        self._patch_batch_size = 64  # patches per GPU forward pass

    def load_weights(self, checkpoint_path: str):
        raise NotImplementedError(
            "Use load_stage_weights(coarse_path, dilated_path, patch16_path, "
            "patch32_path, patch64_path) instead."
        )

    def load_stage_weights(self, coarse_path: str | None = None,
                            dilated_path: str | None = None,
                            patch16_path: str | None = None,
                            patch32_path: str | None = None,
                            patch64_path: str | None = None):
        if coarse_path:
            self.coarse_net.load_weights(coarse_path)
        if dilated_path:
            self.dilated_net.load_weights(dilated_path)
        if patch16_path:
            self.patch_net_16.load_weights(patch16_path)
        if patch32_path:
            self.patch_net_32.load_weights(patch32_path)
        if patch64_path:
            self.patch_net_64.load_weights(patch64_path)

    def _run_patch_stage(self, volume_np: np.ndarray, centers: np.ndarray,
                         patch_size: int, patch_net: "UNetPlusPlus3D",
                         device: torch.device) -> np.ndarray:
        """Extract patches at `centers`, run the net in mini-batches, assemble."""
        patch_net.eval()
        preds = []
        n = len(centers)
        bs = self._patch_batch_size

        for start in range(0, n, bs):
            batch_centers = centers[start:start + bs]
            patches = np.stack([_extract_patch(volume_np, c, patch_size) for c in batch_centers])
            t = torch.from_numpy(patches).float().unsqueeze(1).to(device)
            with torch.no_grad():
                out = patch_net(t)
            pred = torch.sigmoid(out["logits"]).cpu().numpy()
            if pred.ndim == 5:
                pred = pred[:, 0]  # (B, P, P, P)
            preds.extend([pred[i] for i in range(len(batch_centers))])

        return _assemble_predictions(preds, centers, patch_size, volume_np.shape)

    def _coarse_pass(self, net: "UNet3DCoarse", x: torch.Tensor,
                      orig_shape: tuple) -> torch.Tensor:
        """Resize -> net -> upsample back. Shared by Stage 1/2."""
        x_small = F.interpolate(x, size=self.coarse_input_shape,
                                mode="trilinear", align_corners=False)
        logits_small = net(x_small)["logits"]
        return F.interpolate(logits_small, size=orig_shape,
                             mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict:
        orig_shape = x.shape[2:]
        device = x.device

        coarse_logits = self._coarse_pass(self.coarse_net, x, orig_shape)

        if self.training:
            return {"logits": coarse_logits}

        coarse_mask = (torch.sigmoid(coarse_logits) > 0.5).float()

        # Stage 2 supplies skeleton centres only; it never votes.
        dilated_logits = self._coarse_pass(self.dilated_net, x, orig_shape)
        dilated_mask = (torch.sigmoid(dilated_logits) > 0.5).float()

        final_masks = []
        patch_nets = {16: self.patch_net_16, 32: self.patch_net_32, 64: self.patch_net_64}

        for b in range(x.shape[0]):
            coarse_mask_np = coarse_mask[b, 0].cpu().numpy().astype(np.uint8)
            dilated_mask_np = dilated_mask[b, 0].cpu().numpy().astype(np.uint8)
            vol_np = x[b, 0].cpu().numpy()

            # Dilate Stage 2's already-dilation-trained output again, per the paper's
            # "at test the output is further dilated". Same order as
            # stage3_generate_centers.py.
            skel_src = _dilate_mask(dilated_mask_np, self.dilation_radius) if self.use_dilation else dilated_mask_np
            skel_src = _remove_small_components(skel_src, self.min_component_size)
            skeleton = _extract_skeleton(skel_src)

            centers = np.argwhere(skeleton)  # (N, 3) voxel coordinates
            if len(centers) == 0:
                raise RuntimeError(
                    f"No skeleton points found for batch item {b} after skeletonising the "
                    f"Stage 2 (dilated) mask. That segmentation is likely empty or very sparse. "
                    f"Check model.dilated_checkpoint and verify input normalisation "
                    f"(HU window [-200, 1000])."
                )

            patch_masks = [
                self._run_patch_stage(vol_np, centers, ps, patch_nets[ps], device)
                for ps in self.patch_sizes
            ]

            final_masks.append(_majority_vote([coarse_mask_np] + patch_masks))

        # Downstream code expects logits, so emit unambiguous pseudo-logits that
        # sigmoid()>0.5 recovers exactly, whichever way the comparison is written.
        final = torch.from_numpy(np.stack(final_masks)).float().unsqueeze(1).to(device)
        final_logits = (final * 2.0 - 1.0) * 20.0
        return {"logits": final_logits, "coarse_logits": coarse_logits}
