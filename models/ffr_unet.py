"""3D-FFR-UNet: Feature-Fusion-and-Rectification 3D U-Net.

Source: https://github.com/along-song/3D_CAS
Paper: Song et al., "Automatic Coronary Artery Segmentation of CCTA Images With an
       Efficient Feature-Fusion-and-Rectification 3D-UNet." IEEE JBHI 26(8) (2022).

PyTorch port aligned to the paper's Fig. 2/3: each encoder stage is
`3x3x3 conv -> 3D-FEF-DenseBlock`, each decoder stage inserts a `3D-ResidualBlock`
after skip fusion to rectify features. Channel progression at 64^3 input:

    L0: 1   -> conv 32  -> FEF -> 48   (64^3)
    L1: 48  -> conv 64  -> FEF -> 96   (32^3)
    L2: 96  -> conv 128 -> FEF -> 192  (16^3)
    L3: 192 -> conv 256 -> FEF -> 384  (8^3, bottleneck)

HU is windowed to [-200, 1000] rather than the paper's [-500, 1500], for consistency
with the other benchmark methods. The paper's 2D slice-screening classifier is a
separate pipeline stage and is not implemented here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseLumenModel
from models.registry import register_model


class FEFDenseLayer(nn.Module):
    """1x1x1 bottleneck -> 3x3x3 conv, returned concatenated with the input."""

    def __init__(self, in_ch, growth):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_ch), nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, growth, 1, bias=False),            # 1x1x1 bottleneck
            nn.BatchNorm3d(growth), nn.ReLU(inplace=True),
            nn.Conv3d(growth, growth, 3, padding=1, bias=False),  # 3x3x3
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], dim=1)


class FEFDenseBlock(nn.Module):
    """Short (3-layer) thick dense block, growth = in_ch, followed by a transition
    compressing to round(transition_ratio * in_ch)."""

    def __init__(self, in_ch, n_layers=3, transition_ratio=1.5):
        super().__init__()
        growth = in_ch  # growth rate == stage input depth N
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(FEFDenseLayer(ch, growth))
            ch += growth
        self.dense = nn.Sequential(*layers)
        self.out_ch = int(round(transition_ratio * in_ch))
        self.transition = nn.Sequential(
            nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, self.out_ch, 1, bias=False),
        )

    def forward(self, x):
        return self.transition(self.dense(x))


class ResidualBlock(nn.Module):
    """Three 3x3x3 convs with an identity shortcut — the decoder's rectifier."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels), nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels), nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        return F.relu(self.conv(x) + x, inplace=True)


@register_model("ffr_unet")
class FFRUNet3D(BaseLumenModel):
    """4-level 3D-FFR-UNet: FEF-DenseBlock encoder + Residual-rectified decoder.

    Args:
        base_filters:     stage-0 depth N; stages are N, 2N, 4N, 8N.
        transition_ratio: T in the FEF transition (paper T=1.5).
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32,
                 transition_ratio: float = 1.5):
        super().__init__()
        stage_depths = [base_filters * (2 ** i) for i in range(4)]  # [32, 64, 128, 256]
        self.pool = nn.MaxPool3d(2)

        # Per stage: conv(in -> N) then FEF-DenseBlock(N -> T*N).
        self.stage_convs = nn.ModuleList()
        self.fef_blocks = nn.ModuleList()
        skip_chs = []
        in_ch = in_channels
        for N in stage_depths:
            self.stage_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, N, 3, padding=1, bias=False),
                nn.BatchNorm3d(N), nn.ReLU(inplace=True),
            ))
            fef = FEFDenseBlock(N, n_layers=3, transition_ratio=transition_ratio)
            self.fef_blocks.append(fef)
            skip_chs.append(fef.out_ch)   # [48, 96, 192, 384]
            in_ch = fef.out_ch

        # 3 up-steps, fusing the skips excluding the bottleneck.
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec_convs = nn.ModuleList()
        self.dec_res = nn.ModuleList()
        dec_ch = skip_chs[-1]  # 384 (bottleneck)
        for skip_ch in reversed(skip_chs[:-1]):  # 192, 96, 48
            self.dec_convs.append(nn.Sequential(
                nn.Conv3d(dec_ch + skip_ch, skip_ch, 3, padding=1, bias=False),
                nn.BatchNorm3d(skip_ch), nn.ReLU(inplace=True),
            ))
            self.dec_res.append(ResidualBlock(skip_ch))
            dec_ch = skip_ch

        self.head = nn.Conv3d(dec_ch, 1, 1)  # raw single-channel logit

    def forward(self, x: torch.Tensor) -> dict:
        skips = []
        n_stages = len(self.fef_blocks)
        for i, (conv, fef) in enumerate(zip(self.stage_convs, self.fef_blocks)):
            x = fef(conv(x))
            skips.append(x)
            if i < n_stages - 1:   # no pool after the bottleneck stage
                x = self.pool(x)

        x = skips[-1]  # bottleneck feature
        for conv, res, skip in zip(self.dec_convs, self.dec_res, reversed(skips[:-1])):
            x = self.up(x)
            x = torch.cat([x, skip], dim=1)
            x = res(conv(x))

        return {"logits": self.head(x)}
