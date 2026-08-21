import json
import os
import dataclasses
from dataclasses import dataclass, field


# Exporting these is the documented way to point the benchmark at your data; editing
# pipeline.json still works, but an exported variable wins.
DATA_PATH_ENV = "ImageCAS_X_data_path"
RESULTS_PATH_ENV = "ImageCAS_X_results_path"


def _pick(cls, raw: dict) -> dict:
    """Only the keys of `raw` matching declared fields on `cls`."""
    known = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in known}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` onto `base`. Nested dicts merge key-by-key; anything else,
    lists included, is replaced outright rather than concatenated."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


@dataclass
class DataConfig:
    data_root: str = ""           # base path; other dirs are relative to it unless absolute
    volume_dir: str = ""
    gt_mask_dir: str = ""
    volume_suffix: str = ""       # e.g. ".img.nii.gz"; falls back to file_extension if empty
    mask_suffix: str = ""         # e.g. ".coronary.nii.gz"; falls back to file_extension if empty
    filelist_dir: str = ""        # holds train.txt / val.txt / test.txt, one scan_id per line
    train_ids: list = field(default_factory=list)
    val_ids: list = field(default_factory=list)
    test_ids: list = field(default_factory=list)
    file_extension: str = ".nii.gz"
    # Method-specific extras: patch size, centre dirs, etc.
    params: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = ""
    params: dict = field(default_factory=dict)
    checkpoint: str | None = None
    # Staged-training checkpoints for ImageCAS's 3-stage baseline.
    coarse_checkpoint: str | None = None
    dilated_checkpoint: str | None = None
    patch_checkpoint_16: str | None = None
    patch_checkpoint_32: str | None = None
    patch_checkpoint_64: str | None = None


@dataclass
class LossConfig:
    name: str = "dice_ce"
    params: dict = field(default_factory=dict)


@dataclass
class TrainingConfig:
    batch_size: int = 2
    epochs: int = 100
    lr: float = 1e-4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    num_workers: int = 8
    scheduler: str | None = None
    scheduler_params: dict = field(default_factory=dict)
    output_dir: str = "results/"
    # "dice" follows nnU-Net's practice of tracking validation pseudo-Dice, since
    # loss can improve for reasons unrelated to boundary-overlap quality. Methods
    # whose Dice doesn't reflect the whole training signal should use "loss".
    checkpoint_metric: str = "dice"
    # An "epoch" is a fixed number of iterations rather than one pass over the
    # dataset, so cadence is comparable across methods.
    train_iters_per_epoch: int = 1000
    val_iters_per_epoch: int = 1000
    # bf16 autocast around the forward pass; the loss still runs in fp32, so only
    # the network's internal arithmetic changes. Opt-in per method rather than shared,
    # since it makes a run numerically non-identical to already-trained fp32 ones.
    amp: bool = False
    # cuDNN autotuning, sound only when every batch has one shape -- a variable-shape
    # loader would re-autotune per scan and get slower. Compatible with
    # cudnn.deterministic: autotuning then picks within the deterministic set.
    cudnn_benchmark: bool = False


@dataclass
class PreprocessingConfig:
    steps: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class PostprocessingConfig:
    steps: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    metrics: list = field(default_factory=lambda: ["dice", "hd95", "cl_dice"])
    save_predictions: bool = True
    output_dir: str = "results/"
    # Scan-level descriptor table used to stratify the summary. Relative paths
    # resolve against data.data_root; "" disables stratification.
    descriptors_file: str = "Descriptors.xlsx"
    descriptors_id_column: str = "Scan ID"
    stratify_by: list = field(default_factory=lambda: ["Image Quality", "Dominance", "Disease"])
    # Local-Dice-per-centerline-point breakdown; see evaluate.py::_evaluate_points.
    # Keys: roi_size_mm and bins ({covariate: [cut point, ...]}). Runs only when the
    # precomputed sample files exist.
    point_metrics: dict = field(default_factory=dict)


class BenchmarkConfig:
    def __init__(self, raw: dict):
        self.method_name: str = raw["method_name"]
        self.description: str = raw.get("description", "")
        self.input_type: str = raw.get("input_type", "volume")
        self.output_type: str = raw.get("output_type", "mask")
        self.results_root: str = raw.get("results_root", "")

        # _pick keeps unknown/comment keys in the JSON from crashing the dataclasses
        self.data = DataConfig(**_pick(DataConfig, raw.get("data", {})))

        # Exported paths override whatever the merged JSON says.
        env_data_root = os.environ.get(DATA_PATH_ENV, "").strip()
        if env_data_root:
            self.data.data_root = env_data_root
        env_results_root = os.environ.get(RESULTS_PATH_ENV, "").strip()
        if env_results_root:
            self.results_root = env_results_root

        self.model = ModelConfig(**_pick(ModelConfig, raw.get("model", {})))
        self.loss = LossConfig(**_pick(LossConfig, raw.get("loss", {})))
        self.training = TrainingConfig(**_pick(TrainingConfig, raw.get("training", {})))
        self.preprocessing = PreprocessingConfig(**_pick(PreprocessingConfig, raw.get("preprocessing", {})))
        self.postprocessing = PostprocessingConfig(**_pick(PostprocessingConfig, raw.get("postprocessing", {})))
        self.evaluation = EvaluationConfig(**_pick(EvaluationConfig, raw.get("evaluation", {})))

        # {"steps": [...], "params": {...}}, applied to the training split only.
        self.augmentation: dict = raw.get("augmentation", {})

        # IDs in the filelist txt files take precedence over any inline ids here.
        if self.data.filelist_dir:
            self._load_filelist_ids()

    def _resolve_data_path(self, rel_or_abs: str) -> str:
        """Join with data_root unless already absolute."""
        if not rel_or_abs or os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.data.data_root, rel_or_abs)

    def _load_filelist_ids(self):
        filelist_dir = self._resolve_data_path(self.data.filelist_dir)

        # Scans listed in exclude.txt are dropped from every split.
        exclude = set()
        exclude_txt = os.path.join(filelist_dir, "exclude.txt")
        if os.path.exists(exclude_txt):
            with open(exclude_txt, encoding="utf-8") as f:
                exclude = {line.strip() for line in f if line.strip()}

        for split, attr in [("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")]:
            txt = os.path.join(filelist_dir, f"{split}.txt")
            if os.path.exists(txt):
                with open(txt, encoding="utf-8") as f:
                    ids = [line.strip() for line in f
                           if line.strip() and line.strip() not in exclude]
                setattr(self.data, attr, ids)

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkConfig":
        """Load a method config over the shared defaults in a sibling pipeline.json;
        method keys win on conflict."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        pipeline_path = os.path.join(os.path.dirname(os.path.abspath(path)), "pipeline.json")
        if os.path.exists(pipeline_path) and os.path.abspath(pipeline_path) != os.path.abspath(path):
            with open(pipeline_path, encoding="utf-8") as f:
                base = json.load(f)
            raw = _deep_merge(base, raw)

        return cls(raw)

    def validate(self):
        assert self.data.volume_dir or self.data.data_root, \
            "data.volume_dir or data.data_root must be set in config"
        assert self.data.gt_mask_dir, "data.gt_mask_dir must be set in config"
        assert self.model.name, "model.name must be set in config"
        assert self.training.checkpoint_metric in ("dice", "loss"), \
            f"training.checkpoint_metric must be 'dice' or 'loss', got '{self.training.checkpoint_metric}'"
