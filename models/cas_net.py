"""CAS-Net: Multi-Attention Multi-Scale 3D Coronary Artery Segmentation.

Dong et al., Medical Image Analysis 85 (2023). Ported module-for-module from
https://github.com/Cassie-CV/CAS-Net/blob/main/model/casnet_3d.py.

Two deviations, both required by this framework's BaseLumenModel contract:
forward() returns a dict, and MSFA's final Sigmoid is omitted since models here
return raw logits — re-adding it would double-squash the output.

Output is 2-channel; the final mask is an argmax over dim 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseLumenModel
from models.registry import register_model


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    """Dual 3x3x3 conv block with a 1x1 conv residual shortcut."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class Decoder3d(nn.Module):
    """Dual 3x3x3 conv block consuming the AGFF+deconv concat."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


class AGFF(nn.Module):
    """Attention-Guided Feature Fusion: channel attention over the concatenated
    low/high feature gates only the low-level one; the high-level is added back
    unweighted."""

    def __init__(self, low_channels, high_channels, reduction=16):
        super().__init__()
        self.channel_reduce = nn.Conv3d(high_channels, low_channels, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(low_channels * 2, (low_channels * 2) // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear((low_channels * 2) // reduction, low_channels * 2, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, low_feat, high_feat):
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode="trilinear", align_corners=True)
        high_feat = self.channel_reduce(high_feat)
        fused = torch.cat([low_feat, high_feat], dim=1)

        b, c, _, _, _ = fused.size()
        avg_out = self.mlp(self.avg_pool(fused).view(b, c))
        max_out = self.mlp(self.max_pool(fused).view(b, c))
        att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)

        low_weighted = att[:, :low_feat.size(1), :, :, :] * low_feat
        return low_weighted + high_feat


class SAFE(nn.Module):
    """Scale-Aware Feature Enhancement: four channel groups gated by dilated-conv
    branches, then non-local self-attention over the concatenation as a residual,
    with anisotropic convs giving Q/K/V."""

    def __init__(self, in_channels, rates=(1, 2, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=r, dilation=r),
                nn.BatchNorm3d(in_channels // 4),
                nn.Sigmoid(),
            )
            for r in rates
        ])
        self.q_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.k_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.v_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        groups = torch.chunk(x, 4, dim=1)
        out_groups = [branch(groups[i]) * groups[i] for i, branch in enumerate(self.branches)]
        multi_scale = torch.cat(out_groups, dim=1)

        B, C, H, W, D = multi_scale.size()
        Q = self.q_conv(multi_scale).view(B, C, -1)
        K = self.k_conv(multi_scale).view(B, C, -1)
        V = self.v_conv(multi_scale).view(B, C, -1)

        att = self.softmax(torch.bmm(Q.permute(0, 2, 1), K) / (C ** 0.5))
        att_feat = torch.bmm(att, V.permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W, D)
        return multi_scale + att_feat


class MSFA(nn.Module):
    """Multi-Scale Feature Aggregation, top-down over [D4, D3, D2, D1]: each step
    gates the current scale by a conv over concat(current, upsampled aggregate), then
    adds that aggregate back directly."""

    def __init__(self, in_channels_list):
        super().__init__()
        self.compress = nn.ModuleList([nn.Conv3d(c, 8, kernel_size=1) for c in in_channels_list])
        self.att_convs = nn.ModuleList([
            nn.Conv3d(16, 8, kernel_size=3, padding=1) for _ in range(len(in_channels_list) - 1)
        ])
        self.sigmoid = nn.Sigmoid()
        self.out_conv = nn.Conv3d(8, 2, kernel_size=3, padding=1)

    def forward(self, features):
        # Smallest spatial scale first.
        compressed = [c(f) for c, f in zip(self.compress, features)]

        agg = compressed[0]
        for i in range(1, len(compressed)):
            up = F.interpolate(agg, size=compressed[i].shape[2:], mode="trilinear", align_corners=True)
            att = self.sigmoid(self.att_convs[i - 1](torch.cat([compressed[i], up], dim=1)))
            agg = att * compressed[i] + up

        # No final Sigmoid, unlike the reference — see module docstring.
        return self.out_conv(agg)


@register_model("cas_net")
class CASNet3D(BaseLumenModel):
    """5-level ResEncoder3d encoder, SAFE bottleneck, AGFF-fused decoder, MSFA head."""

    def __init__(self, classes: int = 2, channels: int = 1):
        super().__init__()
        # `classes` is accepted to match the reference signature but unused: the
        # official code also hardcodes 2 output channels in MSFA.out_conv.
        self.enc_input = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()

        self.safe = SAFE(256)

        self.agff4 = AGFF(low_channels=128, high_channels=256)
        self.decoder4 = Decoder3d(256, 128)
        self.deconv4 = deconv(256, 128)

        self.agff3 = AGFF(low_channels=64, high_channels=128)
        self.decoder3 = Decoder3d(128, 64)
        self.deconv3 = deconv(128, 64)

        self.agff2 = AGFF(low_channels=32, high_channels=64)
        self.decoder2 = Decoder3d(64, 32)
        self.deconv2 = deconv(64, 32)

        self.agff1 = AGFF(low_channels=16, high_channels=32)
        self.decoder1 = Decoder3d(32, 16)
        self.deconv1 = deconv(32, 16)

        self.msfa = MSFA([128, 64, 32, 16])

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> dict:
        e0 = self.enc_input(x)
        e1 = self.encoder1(self.downsample(e0))
        e2 = self.encoder2(self.downsample(e1))
        e3 = self.encoder3(self.downsample(e2))
        e4 = self.encoder4(self.downsample(e3))

        e4 = self.safe(e4)

        d4 = self.decoder4(torch.cat([self.agff4(e3, e4), self.deconv4(e4)], dim=1))
        d3 = self.decoder3(torch.cat([self.agff3(e2, d4), self.deconv3(d4)], dim=1))
        d2 = self.decoder2(torch.cat([self.agff2(e1, d3), self.deconv2(d3)], dim=1))
        d1 = self.decoder1(torch.cat([self.agff1(e0, d2), self.deconv1(d2)], dim=1))

        return {"logits": self.msfa([d4, d3, d2, d1])}
