#!/usr/bin/env python3
"""
plot_safe_fig5.py
Plot DRL-OR Fig.5-like curves for YOUR safe-learning logs only.

Expected directory layout by default:
  LOG_BASE/
    mappo_initialization/
      delay_type0.log, throughput_type0.log, loss_type0.log, globalrwd.log, circle.log, ...
    mappo_link_failure/
      delay_type0.log, throughput_type0.log, ...
    mappo_traffic_change/
      delay_type0.log, throughput_type0.log, ...

Usage:
  python plot_safe_fig5.py --log-base ./log --out ./figures/fig5_safe.png --num-types 3
  python plot_safe_fig5.py --log-base ./log --metrics delay throughput loss --num-types 4
"""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENARIOS = [
    ("mappo_initialization", "Initialization", None),
    ("mappo_link_failure", "Link failure", 10000),
    ("mappo_traffic_change", "Traffic change", 10000),
]

TYPE_LABELS = {
    0: "Type I: latency-sensitive",
    1: "Type II: throughput-sensitive",
    2: "Type III: latency-throughput-sensitive",
    3: "Type IV: latency-loss-sensitive",
}
TYPE_SHORT = {0: "Type I", 1: "Type II", 2: "Type III", 3: "Type IV"}

METRIC_LABELS = {
    "delay": "Average latency (ms)",
    "throughput": "Throughput ratio",
    "loss": "Packet loss ratio (%)",
    "globalrwd": "Global reward",
}


def read_log(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([])
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                pass
    return np.asarray(vals, dtype=float)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="valid")


def plot_safe_fig5(log_base: Path, out: Path, metrics, num_types: int, window: int, x_scale: float):
    n_rows = len(metrics)
    n_cols = len(SCENARIOS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.2 * n_rows), squeeze=False)
    fig.suptitle("MAPPO-OR with Safe Learning — Fig.5-style Learning Curves", fontsize=15, fontweight="bold")

    for r, metric in enumerate(metrics):
        for c, (scenario_dir, scenario_title, event_step) in enumerate(SCENARIOS):
            ax = axes[r][c]
            sdir = log_base / scenario_dir
            has_data = False

            if metric == "globalrwd":
                data = read_log(sdir / "globalrwd.log")
                if len(data):
                    y = moving_average(data, window)
                    x = np.arange(len(y)) / x_scale
                    ax.plot(x, y, linewidth=1.2, label="global reward")
                    has_data = True
            else:
                for t in range(num_types):
                    data = read_log(sdir / f"{metric}_type{t}.log")
                    if len(data) == 0:
                        continue
                    if metric == "loss":
                        # In simenv logs, loss is usually a ratio; convert to percentage.
                        data = data * 100.0
                    y = moving_average(data, window)
                    x = np.arange(len(y)) / x_scale
                    ax.plot(x, y, linewidth=1.1, label=TYPE_SHORT[t])
                    has_data = True

            if event_step is not None:
                # Event in code is at step 10000 for link_failure and traffic_change.
                # Adjust for smoothing window shift approximately.
                ax.axvline(event_step / x_scale, linestyle="--", linewidth=0.9, alpha=0.7)
                ax.text(event_step / x_scale, 0.95, " event", rotation=90,
                        transform=ax.get_xaxis_transform(), va="top", fontsize=8)

            ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {scenario_title}", fontsize=10, fontweight="bold")
            ax.set_xlabel(f"Timeslot (×{int(x_scale):,})")
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            if has_data:
                ax.legend(fontsize=7, loc="best", framealpha=0.75)
            else:
                ax.text(0.5, 0.5, f"No data\n{sdir}", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-base", default="./log", help="Base directory containing mappo_* log folders")
    parser.add_argument("--out", default="./figures/fig5_safe.png", help="Output image path")
    parser.add_argument("--metrics", nargs="+", default=["delay", "throughput"],
                        choices=["delay", "throughput", "loss", "globalrwd"])
    parser.add_argument("--num-types", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--window", type=int, default=500, help="Moving-average smoothing window")
    parser.add_argument("--x-scale", type=float, default=1000.0, help="X axis scale; 1000 means ×10^3")
    args = parser.parse_args()
    plot_safe_fig5(Path(args.log_base), Path(args.out), args.metrics, args.num_types, args.window, args.x_scale)


if __name__ == "__main__":
    main()
