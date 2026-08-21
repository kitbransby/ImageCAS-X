"""Swin UNETR: Swin Transformer encoder + CNN decoder for 3D segmentation.

Hatamizadeh et al., BrainLes/MICCAI 2022 (arXiv:2201.01266). Cross-checked against
https://github.com/LeonidAlekseev/Swin-UNETR. Only the fully-supervised architecture
of Sec. 3 is implemented; the reference repo's self-supervised pretraining is not
part of the paper's segmentation method.

    input -> encoder0 (residual conv, full-res skip)
          -> patch embed (2x2x2 conv) -> 4 Swin stages, each 2 blocks + patch merge,
             widths C, 2C, 4C, 8C -> 16C bottleneck
          -> decoder: (deconv -> concat skip -> residual conv) x5 back to full res
          -> 1x1x1 conv head

Each stage runs regular- then shifted-window attention (Eqs. 1-2) with a learned 3D
relative position bias. At the deepest stages the token grid can be smaller than the
window, so window and shift clamp down to the feature-map size.

Deviations, all to match this benchmark's conventions: the head returns raw logits;
out_channels defaults to 1 rather than the paper's 3 overlapping BraTS sub-regions;
input is single-channel HU-windowed CT rather than 4-channel z-scored MRI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseLumenModel
from models.registry import register_model


def get_window_size(x_size, window_size, shift_size):
    """Clamp window/shift down to x_size on any axis smaller than the window."""
    use_window_size = list(window_size)
    use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            use_shift_size[i] = 0
    return tuple(use_window_size), tuple(use_shift_size)


def window_partition(x, window_size):
    """(B,D,H,W,C) -> (num_windows*B, wd*wh*ww, C)."""
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wd * wh * ww, C)


def window_reverse(windows, window_size, B, D, H, W):
    """Inverse of window_partition."""
    wd, wh, ww = window_size
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


def _axis_slices(size, window, shift):
    if shift == 0:
        return [slice(0, size)]
    return [slice(0, -window), slice(-window, -shift), slice(-shift, None)]


def compute_attn_mask(dims, window_size, shift_size, device):
    """Additive mask blocking attention between the non-adjacent sub-regions the
    cyclic shift creates (Sec. 3.1)."""
    D, H, W = dims
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in _axis_slices(D, window_size[0], shift_size[0]):
        for h in _axis_slices(H, window_size[1], shift_size[1]):
            for w in _axis_slices(W, window_size[2], shift_size[2]):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)  # (nW, N)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class DropPath(nn.Module):
    """Stochastic depth: drops the residual branch per sample during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class WindowAttention3D(nn.Module):
    """3D window multi-head self-attention with a learned relative position bias.

    `window_size` is the configured (max) window, which sizes the bias table.
    `forward` takes the ACTUAL window for a call, possibly smaller after clamping,
    and looks up a subset of that table — its offsets are always a subset too.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        wd, wh, ww = window_size
        table_size = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self._index_cache: dict = {}

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _relative_position_index(self, window_size, device):
        if window_size not in self._index_cache:
            wd, wh, ww = window_size
            Wd, Wh, Ww = self.window_size
            coords = torch.stack(torch.meshgrid(
                torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"))
            coords_flat = torch.flatten(coords, 1)
            rel = coords_flat[:, :, None] - coords_flat[:, None, :]
            rel = rel.permute(1, 2, 0).contiguous().float()
            rel[:, :, 0] += Wd - 1
            rel[:, :, 1] += Wh - 1
            rel[:, :, 2] += Ww - 1
            rel[:, :, 0] *= (2 * Wh - 1) * (2 * Ww - 1)
            rel[:, :, 1] *= (2 * Ww - 1)
            self._index_cache[window_size] = rel.sum(-1).long()
        return self._index_cache[window_size].to(device)

    def forward(self, x, window_size, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        index = self._relative_position_index(window_size, x.device)
        bias = self.relative_position_bias_table[index.view(-1)].view(N, N, -1)
        bias = bias.permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock3D(nn.Module):
    """LN -> (S)W-MSA -> residual -> LN -> MLP -> residual (Eq. 1)."""

    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x, window_size, shift_size, attn_mask):
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if any(shift_size):
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        x_windows = window_partition(x, window_size)
        attn_windows = self.attn(x_windows, window_size, mask=attn_mask)
        x = window_reverse(attn_windows, window_size, B, D, H, W)
        if any(shift_size):
            x = torch.roll(x, shifts=shift_size, dims=(1, 2, 3))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    """Concatenate 2x2x2 neighbouring patches to 8C, then LayerNorm -> Linear."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, out_dim, bias=False)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        if D % 2 or H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2, 0, D % 2))
        slices = [
            x[:, :, i::2, j::2, k::2]
            for i in (0, 1) for j in (0, 1) for k in (0, 1)
        ]
        x = torch.cat(slices, dim=1)  # (B, 8C, D/2, H/2, W/2)
        Bn, _, Dn, Hn, Wn = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, 8C)
        x = self.reduction(self.norm(x))
        return x.transpose(1, 2).view(Bn, -1, Dn, Hn, Wn)


class BasicLayer(nn.Module):
    """`depth` alternating regular/shifted Swin blocks, then patch merging."""

    def __init__(self, dim, depth, num_heads, window_size, downsample_out_dim,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size)
        drop_path = drop_path if isinstance(drop_path, (list, tuple)) else [drop_path] * depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim, num_heads, window_size, mlp_ratio,
                                    drop, attn_drop, drop_path[i])
            for i in range(depth)
        ])
        self.downsample = PatchMerging3D(dim, downsample_out_dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        window_size, _ = get_window_size((D, H, W), self.window_size, (0, 0, 0))
        _, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        attn_mask = compute_attn_mask((Dp, Hp, Wp), window_size, shift_size, x.device) \
            if any(shift_size) else None

        for i, blk in enumerate(self.blocks):
            blk_shift = (0, 0, 0) if i % 2 == 0 else shift_size
            blk_mask = attn_mask if i % 2 == 1 else None
            x = blk(x, window_size, blk_shift, blk_mask)

        x = x[:, :D, :H, :W, :].permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        x_down = self.downsample(x)
        return x, x_down


class PatchEmbed3D(nn.Module):
    """Non-overlapping patch partition + linear projection, then LayerNorm."""

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        pad_d, pad_h, pad_w = (pd - D % pd) % pd, (ph - H % ph) % ph, (pw - W % pw) % pw
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        B, C, D2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).view(B, C, D2, H2, W2)


class ResidualConvBlock3D(nn.Module):
    """Two 3x3x3 conv-instancenorm-leakyrelu layers with an identity/1x1 shortcut."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


class UpBlock3D(nn.Module):
    """Deconv upsample -> concat skip -> residual fuse."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = ResidualConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


@register_model("swin_unetr")
class SwinUNETR(BaseLumenModel):
    """Swin encoder with a UNETR-style residual-conv decoder fused via skips at every
    resolution. Defaults match the paper's Table 1."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 feature_size: int = 48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=(7, 7, 7), patch_size=(2, 2, 2), mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1):
        super().__init__()
        n_stages = len(depths)
        dims = [feature_size * (2 ** i) for i in range(n_stages + 1)]  # [C,2C,4C,8C,16C]

        self.patch_embed = PatchEmbed3D(in_channels, dims[0], patch_size)

        total_depth = sum(depths)
        dpr = [d.item() for d in torch.linspace(0, drop_path_rate, total_depth)]
        self.stages = nn.ModuleList()
        i = 0
        for s in range(n_stages):
            self.stages.append(BasicLayer(
                dim=dims[s], depth=depths[s], num_heads=num_heads[s],
                window_size=window_size, downsample_out_dim=dims[s + 1],
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i:i + depths[s]],
            ))
            i += depths[s]

        self.encoder0 = ResidualConvBlock3D(in_channels, feature_size)
        self.encoder1 = ResidualConvBlock3D(dims[0], dims[0])
        self.encoder2 = ResidualConvBlock3D(dims[1], dims[1])
        self.encoder3 = ResidualConvBlock3D(dims[2], dims[2])
        self.encoder4 = ResidualConvBlock3D(dims[3], dims[3])
        self.encoder_bottleneck = ResidualConvBlock3D(dims[4], dims[4])

        self.decoder4 = UpBlock3D(dims[4], dims[3], dims[3])
        self.decoder3 = UpBlock3D(dims[3], dims[2], dims[2])
        self.decoder2 = UpBlock3D(dims[2], dims[1], dims[1])
        self.decoder1 = UpBlock3D(dims[1], dims[0], dims[0])
        self.decoder0 = UpBlock3D(dims[0], feature_size, feature_size)

        self.head = nn.Conv3d(feature_size, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        enc0 = self.encoder0(x)
        t = self.patch_embed(x)

        x1, x1_down = self.stages[0](t)
        enc1 = self.encoder1(x1)
        x2, x2_down = self.stages[1](x1_down)
        enc2 = self.encoder2(x2)
        x3, x3_down = self.stages[2](x2_down)
        enc3 = self.encoder3(x3)
        x4, x4_down = self.stages[3](x3_down)
        enc4 = self.encoder4(x4)

        bottleneck = self.encoder_bottleneck(x4_down)

        d4 = self.decoder4(bottleneck, enc4)
        d3 = self.decoder3(d4, enc3)
        d2 = self.decoder2(d3, enc2)
        d1 = self.decoder1(d2, enc1)
        d0 = self.decoder0(d1, enc0)

        return {"logits": self.head(d0)}
