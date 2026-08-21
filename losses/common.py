"""Loss functions shared across the benchmark's methods.

A multi-term loss sets `self.last_components` = {term: weighted contribution}, which
train.py averages over an epoch and prints. Diagnostic only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()
        intersection = (probs * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class ChannelwiseDiceLoss(nn.Module):
    """Soft Dice scored per channel and averaged, rather than over one flattened blob.

    `DiceLoss` pools every axis, which is right for a single foreground channel but
    wrong for ADE-HTL's 27 independent connectivity channels: pooling normalises by
    the total positive count, so dense axis-aligned offsets dominate and sparse
    diagonal ones drift. Per-channel scoring is what the inference-time fusion needs,
    since it thresholds all 27 at the same 0.5.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        # Reduce over batch + spatial, keeping the channel axis separate.
        dims = [0] + list(range(2, targets.dim()))
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        return (1 - (2 * intersection + self.smooth) / (denom + self.smooth)).mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(logits.squeeze(1), targets.float())
            dice = self.dice(logits, targets)
        else:
            # Multi-channel logits: standard CE, Dice on the foreground channel.
            ce = F.cross_entropy(logits, targets)
            dice = self.dice(logits[:, 1], (targets == 1).float())
        dice, ce = self.dice_weight * dice, self.ce_weight * ce
        self.last_components = {"dice": dice.item(), "ce": ce.item()}
        return dice + ce


# --- CAS-Net losses ---

class MulticlassSoftmaxDiceLoss(nn.Module):
    """Softmax Dice averaged over all classes — matches the CAS-Net reference."""

    def __init__(self, smooth: float = 1.0, p: int = 2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, D, H, W); targets: (B, D, H, W) long
        probs = torch.softmax(logits, dim=1)
        C = logits.shape[1]
        target_onehot = F.one_hot(targets, C).permute(0, 4, 1, 2, 3).float()
        total = 0.0
        for c in range(C):
            pred_c = probs[:, c].reshape(-1)
            tgt_c = target_onehot[:, c].reshape(-1)
            num = 2.0 * (pred_c * tgt_c).sum() + self.smooth
            den = (pred_c.pow(self.p) + tgt_c.pow(self.p)).sum() + self.smooth
            total += 1.0 - num / den
        return total / C


# --- ImageCAS losses ---

class WeightedSimilarityCoeffLoss(nn.Module):
    """ImageCAS Eq. 2: weighted Dice biased toward oversized predictions.

        L = 1 - (TP + s) / (a * |pred| + (1-a) * |gt| + s)

    With a=0.01 the denominator weights pred lightly and gt heavily, so FN is
    penalised far more than FP — thin vessels then survive skeletonisation.
    """

    def __init__(self, a: float = 0.01, smooth: float = 1.0):
        super().__init__()
        self.a = a
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        tp = (probs * targets).sum()
        return 1 - (tp + self.smooth) / (
            self.a * probs.sum() + (1 - self.a) * targets.sum() + self.smooth
        )


# --- ADE-HTL losses (Zhang et al., IEEE TMI 43(2) 2024) ---

class WeightedHausdorffLoss(nn.Module):
    """Weighted Hausdorff distance between a probability map and a point set.

    Eqs. 3-4, from Ribera et al. (CVPR 2019). Supervises the key-point branch without
    needing the number of key points fixed in advance:

        L = 1/(S+eps) * sum_x  u_x * min_y d(x,y)
          + 1/|gt|   * sum_y  M_a[ u_x * d(x,y) + (1 - u_x) * d_max ]

    with the generalized mean M_a at a = -1 and d_max the volume's diagonal, in voxel
    units. The first term's `min_y d(x,y)` is an EDT seeded on the key points, which
    the dataset precomputes per crop, so it costs one elementwise product rather than
    an |x| x |y| distance matrix. The second needs d(x, y) per key point and is
    chunked to bound peak memory.
    """

    def __init__(self, alpha: float = -1.0, eps: float = 1e-4, chunk: int = 8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.chunk = chunk

    @staticmethod
    def _coord_grid(shape, device, dtype):
        axes = [torch.arange(s, device=device, dtype=dtype) for s in shape]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).view(-1, 3)

    def forward(self, probs: torch.Tensor, kp_edt: torch.Tensor,
                kp_coords: torch.Tensor, n_kp: torch.Tensor) -> torch.Tensor:
        """kp_coords is (B, K_max, 3) zero-padded; n_kp says how many rows are real."""
        b = probs.shape[0]
        spatial = probs.shape[1:]
        device, dtype = probs.device, probs.dtype
        d_max = float(torch.tensor(spatial, dtype=torch.float64).pow(2).sum().sqrt())

        grid = self._coord_grid(spatial, device, dtype)  # (N, 3)
        flat_probs = probs.reshape(b, -1)
        flat_edt = kp_edt.reshape(b, -1).to(dtype)

        total = probs.new_zeros(())
        for i in range(b):
            u = flat_probs[i]
            # Every voxel the model lights up pays its distance to the nearest key point.
            term1 = (u * flat_edt[i]).sum() / (u.sum() + self.eps)

            k = int(n_kp[i])
            if k == 0:
                # Term 2 averages over an empty set. Term 1 alone is still right:
                # with kp_edt at d_max it pushes the whole map towards zero.
                total = total + term1
                continue

            pts = kp_coords[i, :k].to(dtype)
            term2 = probs.new_zeros(())
            for lo in range(0, k, self.chunk):
                hi = min(lo + self.chunk, k)
                # (chunk, N) distance from each key point to every voxel.
                d = torch.cdist(pts[lo:hi], grid)
                f = u.unsqueeze(0) * d + (1.0 - u).unsqueeze(0) * d_max
                # Clamped away from zero: f**alpha is a reciprocal here, and a voxel
                # sitting on a key point with u = 1 gives f = 0.
                m = f.clamp(min=self.eps).pow(self.alpha).mean(dim=1).pow(1.0 / self.alpha)
                term2 = term2 + m.sum()
            total = total + term1 + term2 / k
        return total / b


class ADEHTLLoss(nn.Module):
    """ADE-HTL's total loss, Eq. 8:

        L = lam1 * L_DSC + (1 - lam1) * L_CE + lam2 * (L_WHD + L_MSE)

    Takes both dicts rather than a (logits, targets) pair, since the four terms
    supervise different heads; `expects_dict` routes it through train.py accordingly.
    Dice and CE both go on all 27 connectivity channels, MSE on the centerline
    heatmap, WHD on the key points — an interpretive reading of Sec. III-C, which
    says only that WCE is "for the neighbor connectivity branch" and DSC is added
    "to ensure whole vessel segmentation performance".

    The CE term is deliberately UNWEIGHTED. Sec. III-C names "weighted cross entropy"
    but specifies no scheme, and inverse-frequency weighting sets the weighted-BCE
    optimum at a base-rate voxel to sigma* = wp/(wp + 1 - p) = 1/2 exactly, for every
    p — precisely the 0.5 the connectivity fusion binarises at, leaving the cut no
    margin and producing a one-voxel halo on the 26 channels Dice does not dominate.
    Dice already handles the imbalance, being scale-invariant to class frequency.
    """

    expects_dict = True

    def __init__(self, lambda1: float = 0.5, lambda2: float = 0.1, whd_chunk: int = 8):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dice = ChannelwiseDiceLoss()
        self.whd = WeightedHausdorffLoss(chunk=whd_chunk)

    def forward(self, output: dict, targets: dict) -> torch.Tensor:
        connectivity = targets["connectivity"].float()
        dice = self.lambda1 * self.dice(output["logits"], connectivity)
        ce = (1.0 - self.lambda1) * F.binary_cross_entropy_with_logits(
            output["logits"], connectivity)

        # Sigmoid before MSE: the target is a Gaussian in [0, 1] (Eq. 6), so bounding
        # the prediction likewise is what makes the regression well posed. The paper
        # states no activation for this head.
        pred_cl = torch.sigmoid(output["centerline"]).squeeze(1)
        mse = self.lambda2 * F.mse_loss(pred_cl, targets["cl_heatmap"].float())

        kp_probs = torch.sigmoid(output["keypoints"]).squeeze(1)
        whd = self.lambda2 * self.whd(kp_probs, targets["kp_edt"],
                                      targets["kp_coords"], targets["n_kp"])

        self.last_components = {"dice": dice.item(), "ce": ce.item(),
                                "mse": mse.item(), "whd": whd.item()}
        return dice + ce + mse + whd


class DeepSupervisionLoss(nn.Module):
    """Averages a base loss over U-Net++'s deep-supervision head list."""

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            return sum(self.base_loss(o, targets) for o in logits) / len(logits)
        return self.base_loss(logits, targets)
