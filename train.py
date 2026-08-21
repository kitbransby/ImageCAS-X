"""Entry point for training a benchmark method.

Usage:
    python -m train -c configs/<method>.json
"""
import argparse
import os
import sys
import time
from tqdm import tqdm

# Must be set before numpy/torch/SimpleITK are imported: each spawns a thread pool
# sized to the full core count PER PROCESS, so num_workers>1 oversubscribes the CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from utils.config import BenchmarkConfig
from utils.seeding import seed_everything
from dataloading.factory import build_dataloader
from models.registry import build_model
from losses.factory import build_loss
from preprocessing.pipeline import build_preprocessing
from augmentation.pipeline import build_augmentation


class _Tee:
    """Mirror all stdout writes to a log file as well as the terminal."""
    def __init__(self, path: str):
        self._file = open(path, "w", buffering=1)
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_optimizer(model, config: BenchmarkConfig):
    name = config.training.optimizer.lower()
    lr = config.training.lr
    wd = config.training.weight_decay
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config: BenchmarkConfig):
    name = config.training.scheduler
    if name is None:
        return None
    p = config.training.scheduler_params
    if name == "poly":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=config.training.epochs,
            power=p.get("power", 0.9),
        )
    raise ValueError(f"Unknown scheduler: '{name}'")


def _is_better(metric_name: str, candidate: float, best: float) -> bool:
    """Whether `candidate` should replace `best` ("dice": higher is better;
    "loss": lower). NaN never replaces anything, so an epoch whose val_dice comes
    back NaN is skipped rather than saving a meaningless checkpoint."""
    if metric_name == "dice":
        return candidate > best
    return candidate < best


def _batch_targets(batch: dict, mask: torch.Tensor, device) -> dict:
    """Everything a dict-taking loss might supervise against, moved to `device`:
    the mask plus every other tensor the dataset emitted. Collected generically so a
    new multi-target method needs a dataset and a loss, not an edit here."""
    skip = {"volume", "mask", "scan_id", "spacing", "centerlines",
            "crop_origin", "pre_crop_shape", "center"}
    targets = {"mask": mask}
    for key, value in batch.items():
        if key not in skip and torch.is_tensor(value):
            targets[key] = value.to(device)
    return targets


def _amp_dtype(cfg: BenchmarkConfig, device) -> torch.dtype | None:
    """The autocast dtype, or None for fp32. Falls back rather than failing, so the
    same config runs anywhere."""
    if not cfg.training.amp:
        return None
    if device.type != "cuda":
        print("[train] training.amp is set but device is not CUDA -- running in fp32.")
        return None
    if not torch.cuda.is_bf16_supported():
        print("[train] training.amp is set but this GPU has no bfloat16 support -- "
              "running in fp32.")
        return None
    return torch.bfloat16


def _logits_to_fp32(output: dict) -> dict:
    """Cast float outputs back to fp32 before the loss sees them.

    bf16's 8-bit mantissa is too coarse for Dice (sums over ~10^6 voxels) and the
    weighted Hausdorff distance (a reciprocal-power mean over the same). Casting here
    keeps every loss numerically identical to an fp32 run, so training.amp changes
    only the network's internal arithmetic. Autograd flows through the cast.
    """
    def cast(v):
        if isinstance(v, (list, tuple)):
            return type(v)(cast(x) for x in v)
        if torch.is_tensor(v) and v.is_floating_point():
            return v.float()
        return v

    return {k: cast(v) for k, v in output.items()}


def _compute_loss(loss_fn, output: dict, targets: dict) -> torch.Tensor:
    """Dispatch on what the model returned: a list of logits is averaged over heads;
    a loss setting `expects_dict` gets both dicts, since it supervises several heads
    against several ground truths; anything else is the standard call."""
    logits = output.get("logits")
    mask = targets["mask"]

    if isinstance(logits, (list, tuple)):
        return sum(loss_fn(lg, mask) for lg in logits) / len(logits)

    if getattr(loss_fn, "expects_dict", False):
        return loss_fn(output, targets)

    return loss_fn(logits, mask)


def train(config_path: str):
    seed_everything()
    cfg = BenchmarkConfig.from_json(config_path)
    cfg.validate()

    if cfg.results_root:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        run_dir = os.path.join(cfg.results_root, f"{cfg.method_name}_{ts}")
        cfg.training.output_dir = run_dir

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    tee = _Tee(os.path.join(cfg.training.output_dir, "log.txt"))

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[train] method={cfg.method_name}  device={device}")
        print(f"[train] run_dir={cfg.training.output_dir}")

        amp_dtype = _amp_dtype(cfg, device)
        if cfg.training.cudnn_benchmark:
            # After seed_everything(), which turns it off. deterministic stays on, so
            # this autotunes within the deterministic set. Sound only because every
            # batch of a patch/crop method has one fixed shape.
            torch.backends.cudnn.benchmark = True
        print(f"[train] amp={'bf16' if amp_dtype else 'off'}  "
              f"cudnn_benchmark={torch.backends.cudnn.benchmark}")

        preprocessing = build_preprocessing(cfg)
        augmentation = build_augmentation(cfg.augmentation) if cfg.augmentation.get("steps") else None
        train_loader = build_dataloader(cfg, split="train", preprocessing=preprocessing,
                                        augmentation=augmentation)
        val_loader = build_dataloader(cfg, split="val", preprocessing=preprocessing)

        model = build_model(cfg).to(device)
        loss_fn = build_loss(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        checkpoint_metric = cfg.training.checkpoint_metric
        best_val_score = float("-inf") if checkpoint_metric == "dice" else float("inf")
        train_losses, val_losses, val_dices, lrs = [], [], [], []
        plot_path = os.path.join(cfg.training.output_dir, "training_curves.png")
        train_start = time.time()

        for epoch in range(cfg.training.epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_components: dict = {}

            for batch in tqdm(train_loader, total=len(train_loader)):
                volume = batch["volume"].to(device)
                mask = batch["mask"].to(device)

                optimizer.zero_grad()
                # Forward is autocast, loss is fp32. No GradScaler: bf16 has fp32's
                # exponent range, so gradients cannot underflow as fp16's would.
                with torch.autocast(device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    output = model(volume)
                loss = _compute_loss(loss_fn, _logits_to_fp32(output),
                                     _batch_targets(batch, mask, device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _accumulate_components(train_components, loss_fn)

            train_loss /= len(train_loader)
            val_loss, val_dice, val_components = _validate(model, val_loader, loss_fn, device,
                                                           cfg, amp_dtype=amp_dtype)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{cfg.training.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"dice={val_dice:.4f}  lr={current_lr:.2e}  "
                  f"time={_fmt_time(epoch_time)}"
                  + _fmt_components("train", train_components, len(train_loader))
                  + _fmt_components("val", val_components, len(val_loader)))

            if scheduler is not None:
                scheduler.step(val_loss) if is_plateau else scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            lrs.append(current_lr)
            _save_training_plot(train_losses, val_losses, val_dices, lrs, cfg.training.epochs, plot_path)

            current_score = val_dice if checkpoint_metric == "dice" else val_loss
            if _is_better(checkpoint_metric, current_score, best_val_score):
                best_val_score = current_score
                ckpt_path = os.path.join(cfg.training.output_dir, f"{cfg.method_name}_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> saved checkpoint to {ckpt_path}  ({checkpoint_metric}={current_score:.4f})")

        total_time = time.time() - train_start
        print(f"\nTraining complete.  Total time: {_fmt_time(total_time)}")

    finally:
        tee.close()


def _accumulate_components(totals: dict, loss_fn) -> None:
    """Accumulate this batch's per-term loss breakdown. Purely diagnostic."""
    components = getattr(loss_fn, "last_components", None)
    if not components:
        return
    for name, value in components.items():
        totals[name] = totals.get(name, 0.0) + value


def _fmt_components(label: str, totals: dict, n_batches: int) -> str:
    """One epoch-line fragment, e.g. `val[dice=0.0123 ce=0.0004]`."""
    if not totals:
        return ""
    terms = " ".join(f"{k}={v / max(n_batches, 1):.4f}" for k, v in totals.items())
    return f"  {label}[{terms}]"


def _binarize_logits(logits) -> torch.Tensor:
    """Logits to a binary mask: argmax for multi-channel, sigmoid > 0.5 otherwise."""
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.shape[1] > 1:
        pred = logits.argmax(dim=1)
    else:
        pred = (logits.sigmoid() > 0.5).squeeze(1).long()
    return pred.cpu()


def _dice_score(pred: torch.Tensor, gt: torch.Tensor) -> list:
    """Per-sample Dice, skipping GT patches with no foreground: small crops can still
    land on pure background despite the foreground-quota sampler, and scoring those as
    a trivial 1.0 drowns out the patches that do contain vessel."""
    if gt.dim() == pred.dim() + 1:
        gt = gt.squeeze(1)
    gt = (gt > 0).long().cpu()
    scores = []
    for p, g in zip(pred, gt):
        if g.sum().item() == 0:
            continue
        tp = (p * g).sum().item()
        denom = p.sum().item() + g.sum().item()
        scores.append(2 * tp / denom)
    return scores


def _save_training_plot(train_losses, val_losses, val_dices, lrs, total_epochs, path):
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, train_losses, label="train loss", color="steelblue")
    ax1.plot(epochs, val_losses, label="val loss", color="tomato")
    ax1.set_ylabel("Loss")
    ax1.set_xlim(1, total_epochs + 1)
    ax1.legend(loc="upper left")

    ax1b = ax1.twinx()
    ax1b.plot(epochs, val_dices, label="val dice", color="darkorchid", linestyle="--")
    ax1b.set_ylabel("Dice (lumen)")
    ax1b.set_ylim(0, 1)
    ax1b.legend(loc="upper right")

    ax2.plot(epochs, lrs, color="mediumseagreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning rate")
    ax2.set_xlim(1, total_epochs + 1)

    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _validate(model, loader, loss_fn, device, cfg: BenchmarkConfig,
              amp_dtype: torch.dtype | None = None) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    components: dict = {}
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            mask = batch["mask"].to(device)
            volume = batch["volume"].to(device)
            # Same split as training, so val_loss stays comparable to an fp32 run.
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                output = model(volume)
            output = _logits_to_fp32(output)
            loss = _compute_loss(loss_fn, output, _batch_targets(batch, mask, device))
            total_loss += loss.item()
            _accumulate_components(components, loss_fn)

            # A model whose "logits" are not per-class scores (ADE-HTL's 27
            # independent-sigmoid channels) exposes its plain vessel logit separately,
            # since _binarize_logits would otherwise argmax non-competing channels.
            logits = output.get("seg_logits", output.get("logits"))
            if logits is not None:
                pred = _binarize_logits(logits)
                dice_scores.extend(_dice_score(pred, mask))
    val_loss = total_loss / max(len(loader), 1)
    val_dice = float(sum(dice_scores) / len(dice_scores)) if dice_scores else float("nan")
    return val_loss, val_dice, components


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to method config JSON")
    args = parser.parse_args()
    train(args.config)
