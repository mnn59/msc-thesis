#!/usr/bin/env python3
"""
Create one 3x3 figure from plain log folders.

Columns: initialization, link_failure, traffic_change
Rows: delay, throughput, loss

This version is "heavy-ready": for DRL-OR and MAPPO-OR it first looks for
heavy-specific directories for *all* scenarios and falls back to the old names
if they do not exist.

Examples:
python plot_ch5_3x3_plain_logs_heavy_ready.py \
  --root ./log \
  --load heavy \
  --methods DRL-OR MAPPO-OR \
  --event link_failure:10000 \
  --event traffic_change:10000 \
  --out ./figures_ch5_3x3/combined_3x3_heavy.png \
  --moving-window 100 \
  --verbose
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCENARIO_LABELS = {
    "initialization": "Initialization",
    "link_failure": "Link Failure",
    "traffic_change": "Traffic Change",
}

METRIC_LABELS = {
    "delay": "Delay (ms)",
    "throughput": "Throughput ratio",
    "loss": "Packet loss ratio (%)",
}

METHOD_LABELS = {
    "DRL-OR": "DRL-OR (Base)",
    "MAPPO-OR": "MAPPO-OR (Ours)",
    "PPO": "DRL-OR (Base)",
    "MAPPO": "MAPPO-OR (Ours)",
    "SPR": "SPR / SHR",
    "LBR": "LBR / WP",
    "QoSR": "QoSR",
}

METHOD_STYLES = {
    "DRL-OR": {"linestyle": "-", "linewidth": 1.8},
    "MAPPO-OR": {"linestyle": "-", "linewidth": 1.8},
    "SPR": {"linestyle": "--", "linewidth": 1.4},
    "LBR": {"linestyle": "-.", "linewidth": 1.4},
    "QoSR": {"linestyle": (0, (3, 1, 1, 1)), "linewidth": 1.4},
}


def canonical_method(method: str) -> str:
    return {"PPO": "DRL-OR", "MAPPO": "MAPPO-OR", "SHR": "SPR", "WP": "LBR"}.get(method.strip(), method.strip())


def canonical_metric(metric: str) -> str:
    return {
        "latency": "delay",
        "packet_loss": "loss",
        "loss_ratio": "loss",
        "packet_loss_ratio": "loss",
    }.get(metric.lower().strip(), metric.lower().strip())


def first_existing(root: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists():
            return p
    return None


def run_dir(root: Path, method: str, scenario: str, load: str) -> Optional[Path]:
    method = canonical_method(method)
    load = load.lower()

    if method == "MAPPO-OR":
        if scenario == "initialization":
            candidates = [
                "mappo_initialization_heavy" if load == "heavy" else "mappo_initialization",
                "mappo_initialization",
            ]
        elif scenario == "link_failure":
            candidates = [
                "mappo_link_failure_heavy" if load == "heavy" else "mappo_link_failure",
                "mappo_link_failure",
            ]
        elif scenario == "traffic_change":
            candidates = [
                "mappo_traffic_change_heavy" if load == "heavy" else "mappo_traffic_change",
                "mappo_traffic_change",
            ]
        else:
            return None
        return first_existing(root, candidates)

    if method == "DRL-OR":
        if scenario == "initialization":
            candidates = [
                "ppo_initialization_heavy" if load == "heavy" else "ppo_initialization",
                "ppo_initialization",
            ]
        elif scenario == "link_failure":
            candidates = [
                "ppo_link_failure_heavy" if load == "heavy" else "ppo_link_failure",
                "ppo_link_failure",
            ]
        elif scenario == "traffic_change":
            candidates = [
                "ppo_traffic_change_heavy" if load == "heavy" else "ppo_traffic_change",
                "ppo_traffic_change",
            ]
        else:
            return None
        return first_existing(root, candidates)

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


def read_series(root: Path, method: str, scenario: str, metric: str, type_id: int, load: str, verbose: bool = False):
    metric = canonical_metric(metric)
    d = run_dir(root, method, scenario, load)
    if d is None:
        if verbose:
            print(f"[WARN] Missing dir: method={method}, scenario={scenario}, load={load}")
        return None

    f = d / f"{metric}_type{type_id}.log"
    if not f.exists():
        if verbose:
            print(f"[WARN] Missing file: {f}")
        return None

    x, y = parse_log_file(f)
    if len(y) == 0:
        if verbose:
            print(f"[WARN] Empty file: {f}")
        return None

    if verbose:
        print(f"[OK] {method} {scenario} {metric} type{type_id}: {f}")
    return x, y


def average_types(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    curves = [(x, y) for x, y in curves if len(x) > 1 and len(y) > 1]
    if not curves:
        return None

    start = max(float(x[0]) for x, _ in curves)
    end = min(float(x[-1]) for x, _ in curves)

    if end <= start:
        n = min(len(y) for _, y in curves)
        return curves[0][0][:n], np.vstack([yy[:n] for _, yy in curves]).mean(axis=0)

    n = min(min(len(x) for x, _ in curves), 4000)
    grid = np.linspace(start, end, n)
    vals = [np.interp(grid, x, y) for x, y in curves]
    return grid, np.vstack(vals).mean(axis=0)


def get_curve(root: Path, method: str, scenario: str, metric: str, load: str, type_id: Optional[int], num_types: int, verbose: bool = False):
    if type_id is not None:
        return read_series(root, method, scenario, metric, type_id, load, verbose)

    curves = []
    for tid in range(num_types):
        c = read_series(root, method, scenario, metric, tid, load, verbose)
        if c is not None:
            curves.append(c)
    return average_types(curves)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    window = min(window, len(y))
    return np.convolve(y, np.ones(window) / window, mode="valid")


def smooth_curve(x: np.ndarray, y: np.ndarray, moving_window: int) -> Tuple[np.ndarray, np.ndarray]:
    if moving_window <= 1:
        return x, y
    y2 = moving_average(y, moving_window)
    x2 = x[-len(y2):]
    return x2, y2


def parse_events(items: List[str]) -> Dict[str, float]:
    out = {}
    for item in items:
        scenario, step = item.split(":", 1)
        out[scenario.strip()] = float(step)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("./log"))
    parser.add_argument("--out", type=Path, default=Path("./figures_ch5_3x3/combined_3x3_delay_throughput_loss.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("./figures_ch5_3x3_by_type"))
    parser.add_argument("--load", choices=["light", "heavy"], default="light")
    parser.add_argument("--methods", nargs="+", default=["DRL-OR", "MAPPO-OR"])
    parser.add_argument("--scenarios", nargs="+", default=["initialization", "link_failure", "traffic_change"])
    parser.add_argument("--metrics", nargs="+", default=["delay", "throughput", "loss"])
    parser.add_argument("--event", action="append", default=[])
    parser.add_argument("--num-types", type=int, default=4)
    parser.add_argument("--split-by-type", action="store_true")
    parser.add_argument("--moving-window", type=int, default=100)
    parser.add_argument("--x-divisor", type=float, default=1000.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    methods = [canonical_method(m) for m in args.methods]
    metrics = [canonical_metric(m) for m in args.metrics]
    events = parse_events(args.event)

    type_ids: List[Optional[int]] = list(range(args.num_types)) if args.split_by_type else [None]

    for type_id in type_ids:
        fig, axes = plt.subplots(
            len(metrics),
            len(args.scenarios),
            figsize=(5.8 * len(args.scenarios), 3.35 * len(metrics)),
            squeeze=False,
        )

        for r, metric in enumerate(metrics):
            for c, scenario in enumerate(args.scenarios):
                ax = axes[r][c]
                plotted = False

                for method in methods:
                    curve = get_curve(args.root, method, scenario, metric, args.load, type_id, args.num_types, args.verbose)
                    if curve is None:
                        continue
                    x, y = smooth_curve(curve[0], curve[1], args.moving_window)

                    style = METHOD_STYLES.get(method, {"linestyle": "-", "linewidth": 1.5})
                    ax.plot(
                        x / args.x_divisor,
                        y,
                        label=METHOD_LABELS.get(method, method),
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                    )
                    plotted = True

                if scenario in events:
                    ex = events[scenario] / args.x_divisor
                    ax.axvline(ex, color="black", linestyle="--", linewidth=1.1)
                    ax.text(ex, ax.get_ylim()[1], "event", rotation=90, va="top", ha="right", fontsize=8)

                if r == 0:
                    ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=13, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
                if r == len(metrics) - 1:
                    ax.set_xlabel("Timeslots (× $10^3$)" if args.x_divisor == 1000 else "Timeslots")

                ax.grid(True, alpha=0.25)
                if not plotted:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                if r == 0 and c == 0 and plotted:
                    ax.legend(loc="best", fontsize=9)

        load_label = "Heavy load" if args.load == "heavy" else "Light load"
        type_suffix = "" if type_id is None else f" — Type {type_id}"
        fig.suptitle(
            f"MAPPO-OR vs DRL-OR across Evaluation Scenarios ({load_label}){type_suffix}",
            fontsize=16,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if type_id is None:
            out = args.out
        else:
            out = args.out_dir / args.load / f"type{type_id}" / "combined_3x3_delay_throughput_loss.png"

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
