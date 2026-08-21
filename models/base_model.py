from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLumenModel(nn.Module, ABC):
    """Models take a (B, C, X, Y, Z) float tensor and return a dict with at least
    'logits', raw and unactivated, so postprocessing is uniform."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:
        """e.g. {'logits': Tensor[B, C, X, Y, Z]}."""
        raise NotImplementedError

    def load_weights(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
