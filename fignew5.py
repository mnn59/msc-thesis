#!/usr/bin/env python3
"""
Create one combined global-reward figure for all three scenarios.

Columns:
  Initialization | Link Failure | Traffic Change

Each subplot compares:
  DRL-OR (Base) vs MAPPO-OR (Ours)

It uses the same folder mapping as the previous working script:
  MAPPO-OR:
    initialization  -> log/mappo_initialization
    link_failure    -> log/mappo_link_failure
    traffic_change  -> log/mappo_traffic_change

  DRL-OR:
    initialization  -> log/ppo_initialization
    link_failure    -> log/ppo_link_failure
    traffic_change  -> log/ppo_traffic_change

For heavy initialization:
  MAPPO-OR -> log/mappo_initialization_heavy
  DRL-OR   -> log/ppo_initialization_heavy

The metric is read from:
  globalrwd.log

Example:
python plot_ch5_global_reward_combined.py \
  --root ./log \
  --load light \
  --methods DRL-OR MAPPO-OR \
  --scenarios initialization link_failure traffic_change \
  --event link_failure:10000 \
  --event traffic_change:10000 \
  --out ./figures_ch5_global_reward/global_reward_combined.png \
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

METHOD_LABELS = {
    "DRL-OR": "DRL-OR (Base)",
    "PPO": "DRL-OR (Base)",
    "MAPPO-OR": "MAPPO-OR (Ours)",
    "MAPPO": "MAPPO-OR (Ours)",
}

METHOD_STYLES = {
    "DRL-OR": {"linestyle": "-", "linewidth": 1.8},
    "PPO": {"linestyle": ":", "linewidth": 1.8},
    "MAPPO-OR": {"linestyle": "-", "linewidth": 1.8},
    "MAPPO": {"linestyle": "-", "linewidth": 1.8},
}


def canonical_method(method: str) -> str:
    return {
        "PPO": "DRL-OR",
        "MAPPO": "MAPPO-OR",
    }.get(method.strip(), method.strip())


def run_dir(root: Path, method: str, scenario: str, load: str) -> Optional[Path]:
    method = canonical_method(method)
    load = load.lower()

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

    return None


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


def read_global_reward(root: Path, method: str, scenario: str, load: str, verbose: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    d = run_dir(root, method, scenario, load)
    if d is None:
        if verbose:
            print(f"[WARN] Missing directory for method={method}, scenario={scenario}, load={load}")
        return None

    f = d / "globalrwd.log"
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
        print(f"[OK] {method:8s} | {scenario:14s} | {f}")

    return x, y


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    window = min(window, len(y))
    return np.convolve(y, np.ones(window) / window, mode="valid")


def ema_smooth(y: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0 or len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = smoothing * out[i - 1] + (1 - smoothing) * y[i]
    return out


def smooth_curve(x: np.ndarray, y: np.ndarray, moving_window: int, smoothing: float) -> Tuple[np.ndarray, np.ndarray]:
    if moving_window > 1:
        y_sm = moving_average(y, moving_window)
        x_sm = x[-len(y_sm):]
        return x_sm, y_sm
    if smoothing > 0:
        return x, ema_smooth(y, smoothing)
    return x, y


def parse_events(items: List[str]) -> Dict[str, float]:
    out = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid event format: {item}. Use scenario:step")
        scenario, step = item.split(":", 1)
        out[scenario.strip()] = float(step)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("./log"))
    parser.add_argument("--out", type=Path, default=Path("./figures_ch5_global_reward/global_reward_combined.png"))
    parser.add_argument("--load", choices=["light", "heavy"], default="light")
    parser.add_argument("--methods", nargs="+", default=["DRL-OR", "MAPPO-OR"])
    parser.add_argument("--scenarios", nargs="+", default=["initialization", "link_failure", "traffic_change"])
    parser.add_argument("--event", action="append", default=[], help="scenario:step, e.g., link_failure:10000")
    parser.add_argument("--moving-window", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--x-divisor", type=float, default=1000.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    methods = [canonical_method(m) for m in args.methods]
    events = parse_events(args.event)

    n_cols = len(args.scenarios)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.2), squeeze=False)
    axes = axes.ravel()

    for ax, scenario in zip(axes, args.scenarios):
        plotted = False

        for method in methods:
            curve = read_global_reward(args.root, method, scenario, args.load, args.verbose)
            if curve is None:
                continue

            x, y = smooth_curve(curve[0], curve[1], moving_window=args.moving_window, smoothing=args.smoothing)

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
            event_x = events[scenario] / args.x_divisor
            ax.axvline(event_x, color="black", linestyle="--", linewidth=1.1)
            ax.text(event_x, ax.get_ylim()[1], "event", rotation=90, va="top", ha="right", fontsize=8)

        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=13, fontweight="bold")
        ax.set_xlabel("Timeslots (× $10^3$)" if args.x_divisor == 1000 else "Timeslots")
        ax.grid(True, alpha=0.25)

        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    axes[0].set_ylabel("Global reward")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True)

    load_label = "Heavy load" if args.load == "heavy" else "Light load"
    fig.suptitle(
        f"Global Reward Comparison across Evaluation Scenarios ({load_label})",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)

    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
