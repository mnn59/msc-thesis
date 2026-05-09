#!/usr/bin/env python3
"""
Grouped bar charts from plain log/ folders.

X-axis: Abi Light, Abi Heavy
Y-axis: one metric: delay, throughput, or loss
Bars: SPR, QoSR, LBR, DRL-OR, MAPPO-OR

It computes the mean of the last --tail samples from each log file.
For metric values by service type:
  - delay and throughput: default averages Type I, II, III (0,1,2)
  - loss: default uses Type IV (3), because packet loss is usually discussed for loss-sensitive traffic.
You can override with --types.

Examples:
  python plot_ch5_topology_bar_from_logs.py \
    --root ./log \
    --out-dir ./figures_ch5_bars \
    --metrics delay throughput loss \
    --methods SPR QoSR LBR DRL-OR MAPPO-OR \
    --tail 5000

  # For delay Type I only:
  python plot_ch5_topology_bar_from_logs.py --root ./log --out-dir ./figures --metrics delay --types 0

  # For loss average over all types:
  python plot_ch5_topology_bar_from_logs.py --root ./log --out-dir ./figures --metrics loss --types 0 1 2 3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


LOADS = [("Abi Light", "light"), ("Abi Heavy", "heavy")]

METHOD_LABELS = {
    "SPR": "SPR",
    "QoSR": "QoSR",
    "LBR": "LBR",
    "DRL-OR": "DRL-OR",
    "MAPPO-OR": "MAPPO-OR",
}

METRIC_TO_FILE_PREFIX = {
    "delay": "delay",
    "latency": "delay",
    "throughput": "throughput",
    "loss": "loss",
    "packet_loss": "loss",
}

METRIC_LABELS = {
    "delay": "Average latency (ms)",
    "throughput": "Average throughput ratio",
    "loss": "Average packet loss ratio (%)",
    "packet_loss": "Average packet loss ratio (%)",
}

METHOD_HATCHES = {
    "SPR": "xx",
    "QoSR": "\\\\",
    "LBR": "////",
    "DRL-OR": "..",
    "MAPPO-OR": "",
}


def canonical_method(method: str) -> str:
    return {"SHR": "SPR", "WP": "LBR", "PPO": "DRL-OR", "MAPPO": "MAPPO-OR"}.get(method.strip(), method.strip())


def canonical_metric(metric: str) -> str:
    m = metric.lower().strip()
    return {"latency": "delay", "packet_loss": "loss", "loss_ratio": "loss", "throughput_ratio": "throughput"}.get(m, m)


def run_dir(root: Path, method: str, load: str) -> Optional[Path]:
    method = canonical_method(method)
    suffix = "heavyload" if load == "heavy" else "lightload"

    if method == "SPR":
        name = f"Abi_SHR_5000_{suffix}"
    elif method == "LBR":
        name = f"Abi_WP_5000_{suffix}"
    elif method == "QoSR":
        name = f"Abi_QoS_5000_{suffix}"
    elif method == "DRL-OR":
        name = "ppo_initialization_heavy" if load == "heavy" else "ppo_initialization"
    elif method == "MAPPO-OR":
        name = "mappo_initialization_heavy" if load == "heavy" else "mappo_initialization"
    else:
        return None

    p = root / name
    return p if p.exists() else None


def parse_log(path: Path) -> np.ndarray:
    number = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = [float(v) for v in number.findall(line)]
            if nums:
                vals.append(nums[-1])
    return np.asarray(vals, dtype=float)


def tail_mean(values: np.ndarray, tail: int) -> float:
    if len(values) == 0:
        return np.nan
    if tail > 0:
        values = values[-min(tail, len(values)):]
    return float(np.nanmean(values))


def method_metric_value(root: Path, method: str, load: str, metric: str, types: List[int], tail: int) -> float:
    d = run_dir(root, method, load)
    if d is None:
        print(f"[WARN] Missing directory: method={method}, load={load}")
        return np.nan

    prefix = METRIC_TO_FILE_PREFIX[metric]
    vals = []
    for t in types:
        f = d / f"{prefix}_type{t}.log"
        if not f.exists():
            print(f"[WARN] Missing file: {f}")
            continue
        arr = parse_log(f)
        if len(arr):
            vals.append(tail_mean(arr, tail))
    if not vals:
        return np.nan
    return float(np.nanmean(vals))


def default_types_for_metric(metric: str) -> List[int]:
    # Match common DRL-OR presentation:
    # latency/throughput are generally shown for Type I-III,
    # packet loss is most meaningful for Type IV.
    if metric == "loss":
        return [3]
    return [0, 1, 2]


def plot_metric(root: Path, out_dir: Path, metric: str, methods: List[str], types: List[int], tail: int) -> None:
    metric = canonical_metric(metric)
    methods = [canonical_method(m) for m in methods]

    x = np.arange(len(LOADS))
    n = len(methods)
    width = min(0.8 / n, 0.16)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    for i, method in enumerate(methods):
        ys = []
        for _, load in LOADS:
            ys.append(method_metric_value(root, method, load, metric, types, tail))
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            ys,
            width,
            label=METHOD_LABELS.get(method, method),
            hatch=METHOD_HATCHES.get(method, ""),
            edgecolor="black",
            linewidth=0.7,
            alpha=0.90,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in LOADS], fontweight="bold")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} on Abilene Light/Heavy Load", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(len(methods), 5), loc="best")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"bar_abi_light_heavy_{metric}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("./log"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["SPR", "QoSR", "LBR", "DRL-OR", "MAPPO-OR"])
    parser.add_argument("--metrics", nargs="+", default=["delay", "throughput", "loss"])
    parser.add_argument("--types", nargs="*", type=int, default=None, help="Service type IDs to average. If omitted, defaults depend on metric.")
    parser.add_argument("--tail", type=int, default=5000)
    args = parser.parse_args()

    for metric in args.metrics:
        metric = canonical_metric(metric)
        types = args.types if args.types is not None else default_types_for_metric(metric)
        plot_metric(args.root, args.out_dir, metric, args.methods, types, args.tail)

    print(f"Bar charts saved under: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
