"""Efficiency table for the paper: #Params (M), Memory (GB), #Models, Inference (s).

    python -m utils.computational_cost_analysis                       # params/#models only
    python -m utils.computational_cost_analysis \
        --run ffr_unet=<run_dir> --run cas_net=<run_dir> ...          # + memory/time
    python -m utils.computational_cost_analysis --run ... --latex     # paste-ready rows

#Params is counted here, by instantiating each method's model from its config with
random weights. Each METHODS row names the one config whose model covers the whole
deployed pipeline, since a composite module already holds its stage nets. #Models is
declared per row rather than derived: it is the number of independently trained
networks needed for one prediction.

Memory and Inference come from <run_dir>/computational_cost.json, so both require
having run inference for that method on this machine — comparing them across machines
is meaningless. Produce that file with

    python -m inference -c <config> -r <run_dir> --computational_analysis

which times scans [1:11] of the test split and writes nothing else; without that flag, a
run_dir that already has predictions/ times nothing, cached scans being skipped. Note it
overwrites any computational_cost.json already in that run_dir. nnU-Net rows run
outside this framework and must be filled in by hand.
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import torch

from utils.config import BenchmarkConfig
from models.registry import _MODEL_REGISTRY
import models.registry  # noqa: F401  -- fires every @register_model decorator


@dataclass
class MethodSpec:
    key: str            # --run <key>=<run_dir>
    latex: str          # row label, exactly as it appears in the paper's table
    config: str | None  # config whose model covers the whole deployed pipeline
    n_models: int | None
    note: str = ""      # printed under the table; explains #Models / any external part
    drop_params: list = field(default_factory=list)  # model.params keys to omit


# Row order follows the paper's table.
METHODS = [
    MethodSpec("ffr_unet", r"3D-FFR-UNet~\cite{song2022automatic}",
               "configs/ffr_unet.json", 1),
    MethodSpec("imagecas_baseline", "ImageCAS",
               "configs/imagecas_inference.json", 5,
               note="5 nets: Stage 1 coarse U-Net, Stage 2 dilated U-Net, "
                    "Stage 3 U-Net++ at 16/32/64. All 5 run at inference (Stage 2 "
                    "supplies skeleton centres); 4 of them vote."),
    MethodSpec("swin_unetr", r"Swin-UNETR~\cite{swinunetr}",
               "configs/swin_unetr.json", 1),
    MethodSpec("nnunet", r"nnUNet~\cite{isensee2021nnu}", None, None,
               note="External (nnU-Net's own CLI) -- fill in by hand."),
    MethodSpec("nnunet_cldice", r"nnUNet + clDice~\cite{shit2021cldice}", None, None,
               note="External (nnU-Net's own CLI) -- fill in by hand."),
    MethodSpec("cas_net", r"CAS-Net~\cite{dong2023novel}",
               "configs/cas_net.json", 1),
]


def count_params(spec: MethodSpec) -> int:
    """Total parameters of `spec.config`'s model."""
    cfg = BenchmarkConfig.from_json(spec.config)
    cls = _MODEL_REGISTRY.get(cfg.model.name)
    if cls is None:
        raise ValueError(
            f"{spec.key}: model '{cfg.model.name}' is not registered "
            f"(registered: {sorted(_MODEL_REGISTRY)})."
        )
    params = {k: v for k, v in cfg.model.params.items() if k not in spec.drop_params}
    model = cls(**params)
    return sum(p.numel() for p in model.parameters())


def load_cost(spec: MethodSpec, run_dir: str) -> dict:
    """Read <run_dir>/computational_cost.json."""
    if not os.path.isabs(run_dir) and spec.config:
        results_root = BenchmarkConfig.from_json(spec.config).results_root
        if results_root:
            run_dir = os.path.join(results_root, run_dir)
    path = os.path.join(run_dir, "computational_cost.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{spec.key}: no {path}. It is written by `python -m inference -c "
            f"{spec.config} -r <run_dir>`, but only for scans actually predicted "
            f"during that session -- a run whose predictions/ was already complete "
            f"times nothing."
        )
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", default=[], metavar="KEY=RUN_DIR",
                        help=f"Run folder to read memory/time from, per method. "
                             f"Repeatable. Keys: {', '.join(m.key for m in METHODS)}")
    parser.add_argument("--latex", action="store_true",
                        help="Also print paste-ready LaTeX table rows.")
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run expects KEY=RUN_DIR, got '{item}'")
        key, run_dir = item.split("=", 1)
        if key not in {m.key for m in METHODS}:
            parser.error(f"unknown method key '{key}'")
        runs[key] = run_dir

    rows = []
    for spec in METHODS:
        params_m = None
        if spec.config:
            try:
                params_m = count_params(spec) / 1e6
            except ImportError as e:
                # Don't lose the whole table to one method's missing optional dep.
                print(f"[skip] {spec.key} params: {e}")
        cost = load_cost(spec, runs[spec.key]) if spec.key in runs else {}
        rows.append((spec, params_m, cost))

    fmt = "{:<20} {:>12} {:>12} {:>8} {:>14} {:>7}"
    print(fmt.format("method", "#Params (M)", "Memory (GB)", "#Models", "Inference (s)", "n"))
    print("-" * 78)
    for spec, params_m, cost in rows:
        mem = cost.get("peak_memory_gb")
        mean, std = cost.get("inference_s_mean"), cost.get("inference_s_std")
        print(fmt.format(
            spec.key,
            f"{params_m:.2f}" if params_m is not None else "--",
            f"{mem:.2f}" if mem is not None else "--",
            str(spec.n_models) if spec.n_models is not None else "--",
            f"{mean:.1f} +/- {std:.1f}" if mean is not None else "--",
            str(cost.get("n_scans", "--")),
        ))

    devices = {c["gpu_name"] for _, _, c in rows if c.get("gpu_name")}
    if devices:
        print(f"\nmeasured on: {', '.join(sorted(devices))}")
        if len(devices) > 1:
            print("WARNING: memory/time were measured on different GPUs -- not comparable.")
    for spec, _, _ in rows:
        if spec.note:
            print(f"  [{spec.key}] {spec.note}")

    if args.latex:
        print("\n% --- table rows ---")
        for spec, params_m, cost in rows:
            mem = cost.get("peak_memory_gb")
            mean = cost.get("inference_s_mean")
            cells = [
                f"{params_m:.2f}" if params_m is not None else "00",
                f"{mem:.2f}" if mem is not None else "00",
                str(spec.n_models) if spec.n_models is not None else "00",
                f"{mean:.1f}" if mean is not None else "00",
            ]
            print(f"{spec.latex} &\n{' & '.join(cells)}  \\\\")


if __name__ == "__main__":
    main()
