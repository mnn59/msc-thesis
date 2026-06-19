#!/usr/bin/env python3
"""
fallback_table_and_plot.py
Compute fallback-policy trigger ratio similar to DRL-OR Table II.

Expected circle.log values:
  0 = no fallback triggered
  1 = fallback / safe-learning mechanism triggered

Usage:
  python fallback_table_and_plot.py --log-base ./log --window 5000 --out ./figures/fallback_ratio.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENARIOS = [
    ("mappo_initialization", "Initialization"),
    ("mappo_link_failure", "Link failure"),
    ("mappo_traffic_change", "Traffic change"),
]


def read_circle(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([])
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(1 if float(s) > 0 else 0)
            except ValueError:
                pass
    return np.asarray(vals, dtype=int)


def ratio(x: np.ndarray) -> float:
    return 0.0 if len(x) == 0 else 100.0 * float(np.sum(x == 1)) / float(len(x))


def moving_window_ratio(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    if len(x) < window:
        return np.array([ratio(x)])
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x.astype(float), kernel, mode="valid") * 100.0


def analyze(log_base: Path, window: int):
    rows = []
    curves = {}
    for folder, label in SCENARIOS:
        data = read_circle(log_base / folder / "circle.log")
        if len(data) == 0:
            print(f"Warning: no circle.log found for {folder}")
            continue
        rows.append({
            "folder": folder,
            "label": label,
            "n": len(data),
            "begin": ratio(data[:window]),
            "converged": ratio(data[-window:]),
            "overall": ratio(data),
        })
        curves[label] = moving_window_ratio(data, window)
    return rows, curves


def print_table(rows, window: int):
    print("\nFallback policy trigger ratio (%)")
    print(f"Window size: {window:,} requests/steps")
    print("-" * 78)
    print(f"{'Scenario':<18}{'Beginning':>14}{'After convergence':>22}{'Overall':>14}{'N':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<18}{r['begin']:>13.2f}%{r['converged']:>21.2f}%{r['overall']:>13.2f}%{r['n']:>10,d}")
    print("-" * 78)


def plot_curves(curves, out: Path, window: int, x_scale: float):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for label, y in curves.items():
        if len(y) == 0:
            continue
        x = np.arange(len(y)) / x_scale
        ax.plot(x, y, linewidth=1.2, label=label)
    ax.set_title(f"Fallback Policy Trigger Ratio (window={window:,})", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Timeslot (×{int(x_scale):,})")
    ax.set_ylabel("Trigger ratio (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-base", default="./log")
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--out", default="./figures/fallback_ratio.png")
    parser.add_argument("--x-scale", type=float, default=1000.0)
    args = parser.parse_args()
    rows, curves = analyze(Path(args.log_base), args.window)
    print_table(rows, args.window)
    if curves:
        plot_curves(curves, Path(args.out), args.window, args.x_scale)


if __name__ == "__main__":
    main()
