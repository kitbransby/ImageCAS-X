import torch.nn as nn

from utils.config import BenchmarkConfig
from losses.common import (
    DiceLoss, DiceCELoss, MulticlassSoftmaxDiceLoss,
    WeightedSimilarityCoeffLoss, DeepSupervisionLoss, ADEHTLLoss,
)

_LOSS_REGISTRY: dict[str, type] = {
    "dice": DiceLoss,
    "dice_ce": DiceCELoss,
    # CAS-Net
    "multiclass_dice": MulticlassSoftmaxDiceLoss,
    # ImageCAS patch stage
    "weighted_sim": WeightedSimilarityCoeffLoss,
    "deep_supervision_dice": lambda **kw: DeepSupervisionLoss(DiceLoss(**kw)),
    # ADE-HTL
    "ade_htl": ADEHTLLoss,
}


def build_loss(config: BenchmarkConfig) -> nn.Module:
    name = config.loss.name
    cls = _LOSS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown loss '{name}'. Options: {list(_LOSS_REGISTRY)}")
    return cls(**config.loss.params)
