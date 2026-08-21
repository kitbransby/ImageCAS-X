from utils.config import BenchmarkConfig
from models.base_model import BaseLumenModel

_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator: @register_model("my_unet") on a BaseLumenModel subclass."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def is_staged_model(name: str) -> bool:
    """Whether this model loads weights per stage rather than from a single
    config.model.checkpoint."""
    cls = _MODEL_REGISTRY.get(name)
    return cls is not None and hasattr(cls, "load_stage_weights")


def build_model(config: BenchmarkConfig) -> BaseLumenModel:
    name = config.model.name
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model '{name}'. Registered: {list(_MODEL_REGISTRY)}")

    model = cls(**config.model.params)

    if hasattr(model, "load_stage_weights"):
        model.load_stage_weights(
            coarse_path=config.model.coarse_checkpoint,
            dilated_path=config.model.dilated_checkpoint,
            patch16_path=config.model.patch_checkpoint_16,
            patch32_path=config.model.patch_checkpoint_32,
            patch64_path=config.model.patch_checkpoint_64,
        )
    elif config.model.checkpoint:
        model.load_weights(config.model.checkpoint)

    return model


# Import model modules so their @register_model decorators fire at startup.
from models import cas_net, ffr_unet, imagecas_baseline, swin_unetr, ade_htl  # noqa: F401
