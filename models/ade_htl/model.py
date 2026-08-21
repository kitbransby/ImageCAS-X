"""ADE-HTL: anatomy- and topology-preserving coronary artery segmentation.

Zhang et al., IEEE TMI 43(2):723-733, 2024. Reimplemented from the paper; no
reference implementation exists.

The HTL (hierarchical topology learning) half only. The ADE half is offline:
precompute_ade.py builds the ROI and the five anatomical distance field maps, which
arrive as input channels 1-5 alongside the image (in_channels=6).

Three decoders share one residual-U-Net encoder (Sec. III-B): neighbour connectivity
(27 ch, the segmentation output), centerline heatmap (Eqs. 5-6) and key points
(Eq. 3).

Deviations from the paper:
  * It cites "ResUNet [13]" with no widths, depth or block count, so base_ch and
    n_levels are free choices here.
  * K/V are average-pooled by kv_pool before attention. Query resolution -- what
    "second and third blocks" refers to -- is untouched; the paper says nothing
    about the key set.
  * Eq. 7 as printed has no softmax and no 1/sqrt(d) scaling, and transposes the QK
    product rather than K. Standard scaled dot-product attention is used inside its
    residual/LayerNorm/MLP structure.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseLumenModel
from models.registry import register_model


# Raster order over {-1,0,1}^3. Defined here only; the dataset builds its target
# against the same ordering.
NEIGHBOUR_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))
N_CONNECTIVITY = len(NEIGHBOUR_OFFSETS)          # 27
SELF_CHANNEL = NEIGHBOUR_OFFSETS.index((0, 0, 0))  # 13


def connectivity_votes(prob, threshold: float = 0.5):
    """Fig. 3's pairwise-consistency vote over a 27-channel connectivity map:

        [P's centre channel]  x  ( 1 + #{neighbours Q that mutually agree} )

    where P and Q at offset delta agree only when P's channel for delta and Q's for
    -delta are both set. Channels are binarised at `threshold`; returns a uint8 count
    in [0, 27].

    The centre channel both seeds and gates the count. Seeding is what makes the
    paper's threshold of 2 work: a boundary voxel scores 1 x (1 + 1) = 2 and
    survives, an isolated one scores 1 and does not. The gate is read off Fig. 3,
    whose background cells all print 0 even where four or five neighbours agree — an
    ungated sum would score those and ADD them to the mask, which Fig. 3's fusion
    never does. The two forms agree on a ground-truth encoding and diverge only on
    predictions, whose directional channels fire on a boundary halo.

    Must be evaluated on the grid the network predicted on, since the +-1 shifts
    below only carry the channels' meaning at that spacing.
    """
    import numpy as np

    binary = np.asarray(prob) > threshold
    if binary.ndim != 4 or binary.shape[0] != N_CONNECTIVITY:
        raise ValueError(
            f"connectivity_votes expects a ({N_CONNECTIVITY}, X, Y, Z) connectivity "
            f"probability map, got shape {binary.shape}."
        )

    index = {off: i for i, off in enumerate(NEIGHBOUR_OFFSETS)}
    spatial = binary.shape[1:]
    self_mask = binary[SELF_CHANNEL]
    votes = self_mask.astype(np.uint8)  # seed; max 1 + 26 so uint8 is safe

    for c, offset in enumerate(NEIGHBOUR_OFFSETS):
        if offset == (0, 0, 0):
            continue  # the seed, not a connection
        # Shift the opposite channel so that at P we read Q's own opinion about the
        # P<-Q direction.
        src, dst = [], []
        for axis, d in enumerate(offset):
            n = spatial[axis]
            src.append(slice(max(d, 0), n + min(d, 0)))
            dst.append(slice(max(-d, 0), n + min(-d, 0)))
        opposite = binary[index[tuple(-v for v in offset)]]
        agreed = np.zeros(spatial, dtype=bool)
        agreed[tuple(dst)] = opposite[tuple(src)]
        votes += (binary[c] & agreed).astype(np.uint8)

    votes *= self_mask
    return votes


class ResidualConvBlock3D(nn.Module):
    """Two 3x3x3 conv-instancenorm-leakyrelu layers with an identity/1x1 shortcut.
    Instance norm because this method's patch size forces a small batch, where batch
    statistics are worthless."""

    def __init__(self, in_ch: int, out_ch: int):
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


class DownBlock3D(nn.Module):
    """Strided-conv downsample then a residual block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1, bias=False)
        self.block = ResidualConvBlock3D(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.down(x))


class UpBlock3D(nn.Module):
    """Deconv upsample -> concat skip -> residual fuse."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = ResidualConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class BAIBlock(nn.Module):
    """One bottom-up attention interaction block (Eq. 7). Query from the sparser
    branch's F2, key/value from the denser F1; the result replaces F2:

        h    = F2 + Attn(Q(F2), K(F1), V(F1))
        F2'  = MLP(LN(h)) + h

    Softmax-scaled dot-product attention, since Eq. 7's printed form omits both the
    softmax and the scaling and cannot be trained as written. K/V are average-pooled
    by `kv_pool` first, which keeps cross-attention affordable without changing what
    the query side expresses.
    """

    def __init__(self, dim: int, kv_dim: int, num_heads: int = 4,
                 mlp_ratio: float = 2.0, kv_pool: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.kv_pool = kv_pool
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _tokens(self, x):
        """(B,C,X,Y,Z) -> (B, N, C)."""
        return x.flatten(2).transpose(1, 2)

    def _heads(self, t, b, n):
        """(B,N,C) -> (B, heads, N, C/heads)."""
        return t.view(b, n, self.num_heads, -1).transpose(1, 2)

    def forward(self, sparse, dense):
        b, c = sparse.shape[:2]
        spatial = sparse.shape[2:]

        if self.kv_pool > 1:
            dense = F.avg_pool3d(dense, kernel_size=self.kv_pool, ceil_mode=True)

        q_tok = self._tokens(sparse)
        kv_tok = self._tokens(dense)
        n_q, n_kv = q_tok.shape[1], kv_tok.shape[1]

        q = self._heads(self.to_q(self.norm_q(q_tok)), b, n_q)
        kv = self.norm_kv(kv_tok)
        k = self._heads(self.to_k(kv), b, n_kv)
        v = self._heads(self.to_v(kv), b, n_kv)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = self.proj(attn.transpose(1, 2).reshape(b, n_q, c))

        h = q_tok + attn
        h = self.mlp(self.norm_out(h)) + h
        return h.transpose(1, 2).view(b, c, *spatial)


class _Decoder(nn.Module):
    """One branch's decoder: UpBlock3Ds followed by a 1x1 head."""

    def __init__(self, dims: list, out_channels: int):
        super().__init__()
        # dims is shallow -> deep; block i upsamples dims[-1-i] to dims[-2-i].
        self.blocks = nn.ModuleList([
            UpBlock3D(dims[-1 - i], dims[-2 - i], dims[-2 - i])
            for i in range(len(dims) - 1)
        ])
        self.head = nn.Conv3d(dims[0], out_channels, 1)

    def forward(self, x, skips, bai=None, bai_sites=(), dense_feats=None):
        """`skips` is deep -> shallow. `bai_sites` lists which block indices get a BAI
        module, `bai` is aligned with it, and `dense_feats` supplies the denser
        branch's feature map per block."""
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
            if bai is not None and dense_feats is not None and i in bai_sites:
                x = bai[bai_sites.index(i)](x, dense_feats[i])
            feats.append(x)
        return self.head(x), feats


@register_model("ade_htl")
class ADEHTL(BaseLumenModel):
    """Shared residual-U-Net encoder, three topology decoders, bottom-up attention
    interaction between them.

    in_channels is 6: the CT image plus ADE's five distance field maps. n_levels must
    be >= 5 so the decoder has four blocks and BAI can sit on the second and third,
    as Sec. III-B.4 states. Returns logits, seg_logits, centerline and keypoints.
    """

    def __init__(self, in_channels: int = 6, base_ch: int = 16, n_levels: int = 5,
                 num_heads: int = 4, kv_pool: int = 4):
        super().__init__()
        if n_levels < 5:
            raise ValueError(
                f"n_levels must be >= 5 so BAI can sit on the second and third of four "
                f"decoder blocks (Sec. III-B.4), got {n_levels}."
            )
        dims = [base_ch * (2 ** i) for i in range(n_levels)]
        self.dims = dims

        self.stem = ResidualConvBlock3D(in_channels, dims[0])
        self.downs = nn.ModuleList([DownBlock3D(dims[i], dims[i + 1])
                                    for i in range(n_levels - 1)])

        self.dec_conn = _Decoder(dims, N_CONNECTIVITY)
        self.dec_cl = _Decoder(dims, 1)
        self.dec_kp = _Decoder(dims, 1)

        # Second and third blocks, first and last untouched. Block 0 is coarsest, so
        # token count -- and attention cost, quadratic in it -- grows 8x per step
        # towards full resolution: placement dominates BAI's cost.
        self.bai_sites = (1, 2)
        bai_dims = [dims[-2 - i] for i in self.bai_sites]

        # Bottom-up: connectivity feeds centerline, centerline feeds key points.
        self.bai_cl = nn.ModuleList([
            BAIBlock(d, d, num_heads=num_heads, kv_pool=kv_pool) for d in bai_dims
        ])
        self.bai_kp = nn.ModuleList([
            BAIBlock(d, d, num_heads=num_heads, kv_pool=kv_pool) for d in bai_dims
        ])

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        bottleneck = skips[-1]
        skips = skips[:-1][::-1]  # deep -> shallow, excluding the bottleneck itself

        conn_logits, conn_feats = self.dec_conn(bottleneck, skips)
        cl_logits, cl_feats = self.dec_cl(bottleneck, skips, bai=self.bai_cl,
                                          bai_sites=self.bai_sites,
                                          dense_feats=conn_feats)
        kp_logits, _ = self.dec_kp(bottleneck, skips, bai=self.bai_kp,
                                   bai_sites=self.bai_sites,
                                   dense_feats=cl_feats)

        return {
            "logits": conn_logits,
            # The self offset: "this voxel is vessel". Validation Dice reads this
            # rather than argmaxing 27 independent-sigmoid channels.
            "seg_logits": conn_logits[:, SELF_CHANNEL:SELF_CHANNEL + 1],
            "centerline": cl_logits,
            "keypoints": kp_logits,
        }

    def mirror_channel_permutation(self, axes) -> torch.Tensor:
        """Channel permutation induced by mirroring the given spatial axes.

        Flipping the volume flips which physical direction each offset points in, so
        un-flipping spatially is not enough: channel c still describes delta_c in the
        flipped frame. Its own inverse.
        """
        index = {off: i for i, off in enumerate(NEIGHBOUR_OFFSETS)}
        perm = []
        for offset in NEIGHBOUR_OFFSETS:
            mirrored = tuple(-v if a in axes else v for a, v in enumerate(offset))
            perm.append(index[mirrored])
        return torch.as_tensor(perm, dtype=torch.long)
