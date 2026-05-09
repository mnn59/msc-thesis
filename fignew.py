#!/usr/bin/env python3
"""
Plot Chapter 5 learning curves directly from your plain log/ folders.

This script is tailored for your current folder structure:

log/
  Abi_QoS_5000_heavyload
  Abi_QoS_5000_lightload
  Abi_SHR_5000_heavyload
  Abi_SHR_5000_lightload
  Abi_WP_5000_heavyload
  Abi_WP_5000_lightload
  mappo_initialization
  mappo_initialization_heavy
  mappo_link_failure
  mappo_traffic_change
  ppo_initialization
  ppo_initialization_heavy
  ppo_link_failure
  ppo_traffic_change

Each folder contains files like:
  delay_type0.log ... delay_type3.log
  throughput_type0.log ... throughput_type3.log
  loss_type0.log ... loss_type3.log
  dist_type0.log ... dist_type3.log
  circle.log
  globalrwd.log

Metrics:
  delay       -> delay_typeX.log
  throughput  -> throughput_typeX.log
  loss        -> loss_typeX.log
  stretch     -> dist_typeX.log

Method mapping:
  MAPPO-OR       -> mappo_* folders
  DRL-OR (Base)  -> ppo_* folders
  SPR/SHR        -> Abi_SHR_* folders
  LBR/WP         -> Abi_WP_* folders
  QoSR           -> Abi_QoS_* folders

Examples
--------
1) MAPPO-OR vs DRL-OR for all three scenarios, light load:
python plot_ch5_from_plain_logs.py \
  --root ./log \
  --load light \
  --methods DRL-OR MAPPO-OR \
  --metrics delay throughput stretch \
  --scenarios initialization link_failure traffic_change \
  --event link_failure:10000 \
  --event traffic_change:10000 \
  --out-dir ./figures_ch5_curves \
  --combined

2) Initialization with classical baselines too:
python plot_ch5_from_plain_logs.py \
  --root ./log \
  --load light \
  --methods SPR LBR QoSR DRL-OR MAPPO-OR \
  --metrics delay throughput stretch loss \
  --scenarios initialization \
  --out-dir ./figures_ch5_initialization_all_methods

3) Heavy-load initialization:
python plot_ch5_from_plain_logs.py \
  --root ./log \
  --load heavy \
  --methods SPR LBR QoSR DRL-OR MAPPO-OR \
  --metrics delay throughput stretch loss \
  --scenarios initialization \
  --out-dir ./figures_ch5_heavy_initialization

4) Separate figures for each service type:
python plot_ch5_from_plain_logs.py \
  --root ./log \
  --load light \
  --methods DRL-OR MAPPO-OR \
  --metrics delay throughput stretch \
  --scenarios initialization link_failure traffic_change \
  --split-by-type \
  --out-dir ./figures_ch5_curves_by_type
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


TYPE_LABELS = {
    0: "Type I",
    1: "Type II",
    2: "Type III",
    3: "Type IV",
}

SCENARIO_LABELS = {
    "initialization": "Initialization",
    "link_failure": "Link Failure",
    "traffic_change": "Traffic Change",
}

METRIC_TO_FILE_PREFIX = {
    "delay": "delay",
    "latency": "delay",
    "throughput": "throughput",
    "loss": "loss",
    "packet_loss": "loss",
    "stretch": "dist",
    "dist": "dist",
}

METRIC_LABELS = {
    "delay": "Delay (ms)",
    "latency": "Delay (ms)",
    "throughput": "Throughput ratio",
    "loss": "Packet loss ratio (%)",
    "packet_loss": "Packet loss ratio (%)",
    "stretch": "Path stretch",
    "dist": "Path stretch",
}

METHOD_LABELS = {
    "SPR": "SPR / SHR",
    "SHR": "SPR / SHR",
    "LBR": "LBR / WP",
    "WP": "LBR / WP",
    "QoSR": "QoSR",
    "DRL-OR": "DRL-OR (Base)",
    "PPO": "DRL-OR (Base)",
    "MAPPO-OR": "MAPPO-OR (Ours)",
    "MAPPO": "MAPPO-OR (Ours)",
}

METHOD_STYLES = {
    "SPR": {"linestyle": "--", "linewidth": 1.5},
    "SHR": {"linestyle": "--", "linewidth": 1.5},
    "LBR": {"linestyle": "-.", "linewidth": 1.5},
    "WP": {"linestyle": "-.", "linewidth": 1.5},
    "QoSR": {"linestyle": (0, (3, 1, 1, 1)), "linewidth": 1.5},
    "DRL-OR": {"linestyle": ":", "linewidth": 1.8},
    "PPO": {"linestyle": ":", "linewidth": 1.8},
    "MAPPO-OR": {"linestyle": "-", "linewidth": 1.8},
    "MAPPO": {"linestyle": "-", "linewidth": 1.8},
}


def canonical_method(method: str) -> str:
    m = method.strip()
    aliases = {
        "SHR": "SPR",
        "WP": "LBR",
        "PPO": "DRL-OR",
        "MAPPO": "MAPPO-OR",
    }
    return aliases.get(m, m)


def canonical_metric(metric: str) -> str:
    m = metric.lower().strip()
    aliases = {
        "latency": "delay",
        "packet_loss": "loss",
        "loss_ratio": "loss",
        "path_stretch": "stretch",
        "dist": "stretch",
    }
    return aliases.get(m, m)


def sanitize(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "_")
    )


def run_dir(root: Path, method: str, scenario: str, load: str) -> Optional[Path]:
    method = canonical_method(method)
    load = load.lower()

    if load not in {"light", "heavy"}:
        raise ValueError("--load must be either light or heavy")

    if method == "MAPPO-OR":
        if scenario == "initialization":
            name = "mappo_initialization_heavy" if load == "heavy" else "mappo_initialization"
        elif scenario == "link_failure":
            name = "mappo_link_failure"
        elif scenario == "traffic_change":
            name = "mappo_traffic_change"
        else:
            return None
        p = root / name
        return p if p.exists() else None

    if method == "DRL-OR":
        if scenario == "initialization":
            name = "ppo_initialization_heavy" if load == "heavy" else "ppo_initialization"
        elif scenario == "link_failure":
            name = "ppo_link_failure"
        elif scenario == "traffic_change":
            name = "ppo_traffic_change"
        else:
            return None
        p = root / name
        return p if p.exists() else None

    # Classical baselines are available as light/heavy load folders and are mainly for initialization/table plots.
    suffix = "heavyload" if load == "heavy" else "lightload"
    if method == "SPR":
        name = f"Abi_SHR_5000_{suffix}"
    elif method == "LBR":
        name = f"Abi_WP_5000_{suffix}"
    elif method == "QoSR":
        name = f"Abi_QoS_5000_{suffix}"
    else:
        return None

    p = root / name
    return p if p.exists() else None


def parse_log_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports:
    - one value per line: value
    - two or more numeric columns: uses first as x and last as y
    """
    number = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            vals = [float(v) for v in number.findall(line)]
            if not vals:
                continue
            if len(vals) >= 2:
                xs.append(vals[0])
                ys.append(vals[-1])
            else:
                xs.append(idx)
                ys.append(vals[0])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def read_series(root: Path, method: str, scenario: str, metric: str, type_id: int, load: str) -> Optional[Tuple[np.ndarray, np.ndarray, Path]]:
    metric = canonical_metric(metric)
    prefix = METRIC_TO_FILE_PREFIX.get(metric, metric)

    d = run_dir(root, method, scenario, load)
    if d is None:
        print(f"[WARN] Missing directory for method={method}, scenario={scenario}, load={load}")
        return None

    f = d / f"{prefix}_type{type_id}.log"
    if not f.exists():
        print(f"[WARN] Missing file: {f}")
        return None

    x, y = parse_log_file(f)
    if len(y) == 0:
        print(f"[WARN] Empty file: {f}")
        return None

    return x, y, f


def ema_smooth(y: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0 or len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = smoothing * out[i - 1] + (1 - smoothing) * y[i]
    return out


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    window = min(window, len(y))
    return np.convolve(y, np.ones(window) / window, mode="valid")


def average_types(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Average type curves on a common x grid to avoid unequal-length log problems."""
    curves = [(x, y) for x, y in curves if len(x) > 1 and len(y) > 1]
    if not curves:
        return None

    start = max(float(x[0]) for x, _ in curves)
    end = min(float(x[-1]) for x, _ in curves)

    if end <= start:
        n = min(len(y) for _, y in curves)
        x = curves[0][0][:n]
        y = np.vstack([yy[:n] for _, yy in curves]).mean(axis=0)
        return x, y

    n = min(min(len(x) for x, _ in curves), 4000)
    grid = np.linspace(start, end, n)
    vals = [np.interp(grid, x, y) for x, y in curves]
    return grid, np.vstack(vals).mean(axis=0)


def get_curve(root: Path, method: str, scenario: str, metric: str, load: str, type_id: Optional[int], num_types: int) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    sources: List[str] = []

    if type_id is not None:
        s = read_series(root, method, scenario, metric, type_id, load)
        if s is None:
            return None
        x, y, f = s
        return x, y, [str(f)]

    curves = []
    for tid in range(num_types):
        s = read_series(root, method, scenario, metric, tid, load)
        if s is None:
            continue
        x, y, f = s
        curves.append((x, y))
        sources.append(str(f))

    avg = average_types(curves)
    if avg is None:
        return None

    return avg[0], avg[1], sources


def parse_events(items: List[str]) -> Dict[str, float]:
    out = {}
    for item in items:
        scenario, step = item.split(":", 1)
        out[scenario.strip()] = float(step)
    return out


def plot_single(
    root: Path,
    scenario: str,
    metric: str,
    methods: List[str],
    load: str,
    out_path: Path,
    event_step: Optional[float],
    type_id: Optional[int],
    args,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plotted = False

    for method_raw in methods:
        method = canonical_method(method_raw)
        curve = get_curve(root, method, scenario, metric, load, type_id, args.num_types)
        if curve is None:
            continue

        x, y, sources = curve

        if args.moving_window > 1:
            y = moving_average(y, args.moving_window)
            x = x[-len(y):]
        elif args.smoothing > 0:
            y = ema_smooth(y, args.smoothing)

        style = METHOD_STYLES.get(method, {"linestyle": "-", "linewidth": 1.5})
        ax.plot(
            x / args.x_divisor,
            y,
            label=METHOD_LABELS.get(method, method),
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    if event_step is not None:
        ax.axvline(event_step / args.x_divisor, color="black", linestyle="--", linewidth=1.2)
        ax.text(event_step / args.x_divisor, ax.get_ylim()[1], "event", rotation=90, va="top", ha="right", fontsize=9)

    metric_label = METRIC_LABELS.get(canonical_metric(metric), metric)
    scenario_label = SCENARIO_LABELS.get(scenario, scenario)
    type_suffix = "" if type_id is None else f" — {TYPE_LABELS.get(type_id, 'Type '+str(type_id))}"
    load_label = "Heavy load" if load == "heavy" else "Light load"

    ax.set_title(f"{metric_label} under {scenario_label} ({load_label}){type_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Timeslots (× $10^3$)" if args.x_divisor == 1000 else "Timeslots")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_combined(root: Path, scenario: str, metrics: List[str], methods: List[str], load: str, out_path: Path, event_step: Optional[float], type_id: Optional[int], args) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.4, 3.4 * len(metrics)), squeeze=False)
    axes = axes.ravel()

    any_plot = False
    for ax, metric in zip(axes, metrics):
        for method_raw in methods:
            method = canonical_method(method_raw)
            curve = get_curve(root, method, scenario, metric, load, type_id, args.num_types)
            if curve is None:
                continue
            x, y, _ = curve

            if args.moving_window > 1:
                y = moving_average(y, args.moving_window)
                x = x[-len(y):]
            elif args.smoothing > 0:
                y = ema_smooth(y, args.smoothing)

            style = METHOD_STYLES.get(method, {"linestyle": "-", "linewidth": 1.5})
            ax.plot(
                x / args.x_divisor,
                y,
                label=METHOD_LABELS.get(method, method),
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
            any_plot = True

        if event_step is not None:
            ax.axvline(event_step / args.x_divisor, color="black", linestyle="--", linewidth=1.2)
            ax.text(event_step / args.x_divisor, ax.get_ylim()[1], "event", rotation=90, va="top", ha="right", fontsize=9)

        ax.set_ylabel(METRIC_LABELS.get(canonical_metric(metric), metric))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    if not any_plot:
        plt.close(fig)
        return

    axes[-1].set_xlabel("Timeslots (× $10^3$)" if args.x_divisor == 1000 else "Timeslots")
    scenario_label = SCENARIO_LABELS.get(scenario, scenario)
    load_label = "Heavy load" if load == "heavy" else "Light load"
    type_suffix = "" if type_id is None else f" — {TYPE_LABELS.get(type_id, 'Type '+str(type_id))}"
    fig.suptitle(f"Performance under {scenario_label} ({load_label}){type_suffix}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("./log"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--load", choices=["light", "heavy"], default="light")
    parser.add_argument("--methods", nargs="+", default=["DRL-OR", "MAPPO-OR"])
    parser.add_argument("--scenarios", nargs="+", default=["initialization", "link_failure", "traffic_change"])
    parser.add_argument("--metrics", nargs="+", default=["delay", "throughput", "stretch"])
    parser.add_argument("--num-types", type=int, default=4)
    parser.add_argument("--split-by-type", action="store_true")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--event", action="append", default=[], help="scenario:step, e.g., link_failure:10000")
    parser.add_argument("--smoothing", type=float, default=0.90)
    parser.add_argument("--moving-window", type=int, default=0)
    parser.add_argument("--x-divisor", type=float, default=1000.0)
    args = parser.parse_args()

    events = parse_events(args.event)
    metrics = [canonical_metric(m) for m in args.metrics]
    methods = [canonical_method(m) for m in args.methods]

    type_ids: List[Optional[int]]
    if args.split_by_type:
        type_ids = list(range(args.num_types))
    else:
        type_ids = [None]

    for type_id in type_ids:
        type_dir = "avg_types" if type_id is None else f"type{type_id}"
        for scenario in args.scenarios:
            event_step = events.get(scenario)
            for metric in metrics:
                out = args.out_dir / args.load / type_dir / scenario / f"{metric}.png"
                plot_single(args.root, scenario, metric, methods, args.load, out, event_step, type_id, args)

            if args.combined:
                out = args.out_dir / args.load / type_dir / scenario / f"combined_{'_'.join(metrics)}.png"
                plot_combined(args.root, scenario, metrics, methods, args.load, out, event_step, type_id, args)

    print(f"Figures saved under: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
