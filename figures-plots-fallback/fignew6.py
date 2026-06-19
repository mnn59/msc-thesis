#!/usr/bin/env python3
"""
Create one combined global-reward figure for all three scenarios and
measure recovery / settling speed.

Definitions:
- For scenarios with an event:
    recovery time = first time after the event that a rolling mean enters
    the steady-state band and stays there for hold_window consecutive points.
- For initialization (no event by default):
    settling time = first time from the start that the rolling mean enters
    the steady-state band and stays there for hold_window consecutive points.

Steady state is estimated from the final tail_fraction of the curve
(by default the last 20% of the points after the event, or of the full
curve if no event exists).

The metric is computed from the RAW curve by default and can optionally be
computed from the smoothed curve via --measure-on smoothed.
"""

from __future__ import annotations

import argparse
import csv
import math
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


def nearest_index_at_or_after(x: np.ndarray, value: float) -> int:
    idx = np.searchsorted(x, value, side="left")
    return min(max(int(idx), 0), len(x) - 1)


def rolling_mean_same_length(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) == 0:
        return y.copy()
    window = max(1, min(window, len(y)))
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def estimate_steady_state(
    x: np.ndarray,
    y: np.ndarray,
    event_step: Optional[float],
    tail_fraction: float,
) -> Tuple[float, float, int]:
    n = len(y)
    if n == 0:
        return math.nan, math.nan, 0

    if event_step is None:
        post_start = 0
    else:
        post_start = nearest_index_at_or_after(x, event_step)

    post_n = n - post_start
    tail_n = max(10, int(post_n * tail_fraction))
    tail_start = max(post_start, n - tail_n)

    tail = y[tail_start:]
    return float(np.mean(tail)), float(np.std(tail)), int(tail_start)


def first_consecutive_true(mask: np.ndarray, hold_window: int, start_idx: int) -> Optional[int]:
    if len(mask) == 0:
        return None
    hold_window = max(1, hold_window)
    count = 0
    for i in range(start_idx, len(mask)):
        if bool(mask[i]):
            count += 1
            if count >= hold_window:
                return i - hold_window + 1
        else:
            count = 0
    return None


def measure_recovery(
    x: np.ndarray,
    y: np.ndarray,
    event_step: Optional[float],
    *,
    settle_window: int = 100,
    hold_window: int = 50,
    tail_fraction: float = 0.2,
    band_pct: float = 0.05,
    band_abs_min: float = 0.05,
) -> Dict[str, float]:
    """
    Measure recovery / settling time.

    Returns a dict with:
      steady_mean, steady_std, band, settle_idx, settle_x, delta_x, event_idx
    """
    if len(y) == 0:
        return {
            "steady_mean": math.nan,
            "steady_std": math.nan,
            "band": math.nan,
            "settle_idx": math.nan,
            "settle_x": math.nan,
            "delta_x": math.nan,
            "event_idx": math.nan,
        }

    steady_mean, steady_std, tail_start = estimate_steady_state(x, y, event_step, tail_fraction)
    band = max(abs(steady_mean) * band_pct, band_abs_min)

    y_roll = rolling_mean_same_length(y, settle_window)
    within = np.abs(y_roll - steady_mean) <= band

    if event_step is None:
        start_idx = 0
        origin_x = float(x[0])
    else:
        start_idx = nearest_index_at_or_after(x, event_step)
        origin_x = float(event_step)

    settle_idx = first_consecutive_true(within, hold_window=hold_window, start_idx=start_idx)

    if settle_idx is None:
        settle_x = math.nan
        delta_x = math.nan
    else:
        settle_x = float(x[settle_idx])
        delta_x = float(settle_x - origin_x)

    return {
        "steady_mean": steady_mean,
        "steady_std": steady_std,
        "band": float(band),
        "settle_idx": float(settle_idx) if settle_idx is not None else math.nan,
        "settle_x": settle_x,
        "delta_x": delta_x,
        "event_idx": float(start_idx),
        "tail_start_idx": float(tail_start),
    }


def save_metrics_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_num(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "NA"
    return f"{v:.{digits}f}"


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

    # recovery metric args
    parser.add_argument("--measure-on", choices=["raw", "smoothed"], default="raw")
    parser.add_argument("--settle-window", type=int, default=100, help="rolling mean window for recovery metric")
    parser.add_argument("--hold-window", type=int, default=50, help="must stay in steady band for this many points")
    parser.add_argument("--tail-fraction", type=float, default=0.2, help="last fraction of curve used to estimate steady state")
    parser.add_argument("--band-pct", type=float, default=0.05, help="steady band = max(abs(steady_mean)*band_pct, band_abs_min)")
    parser.add_argument("--band-abs-min", type=float, default=0.05)
    parser.add_argument("--metrics-out", type=Path, default=None, help="optional CSV path for recovery metrics")
    parser.add_argument("--annotate-recovery", action="store_true", help="annotate delta_x on each subplot")
    args = parser.parse_args()

    methods = [canonical_method(m) for m in args.methods]
    events = parse_events(args.event)

    n_cols = len(args.scenarios)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.2), squeeze=False)
    axes = axes.ravel()

    metric_rows: List[Dict[str, object]] = []

    for ax, scenario in zip(axes, args.scenarios):
        plotted = False
        annotation_lines = []

        for method in methods:
            curve = read_global_reward(args.root, method, scenario, args.load, args.verbose)
            if curve is None:
                continue

            x_raw, y_raw = curve
            x_plot, y_plot = smooth_curve(x_raw, y_raw, moving_window=args.moving_window, smoothing=args.smoothing)

            style = METHOD_STYLES.get(method, {"linestyle": "-", "linewidth": 1.5})
            ax.plot(
                x_plot / args.x_divisor,
                y_plot,
                label=METHOD_LABELS.get(method, method),
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
            plotted = True

            event_step = events.get(scenario)

            # measure on raw or on the same smoothed curve
            if args.measure_on == "raw":
                mx, my = x_raw, y_raw
            else:
                mx, my = x_plot, y_plot

            metrics = measure_recovery(
                mx,
                my,
                event_step=event_step,
                settle_window=args.settle_window,
                hold_window=args.hold_window,
                tail_fraction=args.tail_fraction,
                band_pct=args.band_pct,
                band_abs_min=args.band_abs_min,
            )

            metric_rows.append({
                "scenario": scenario,
                "method": method,
                "load": args.load,
                "measure_on": args.measure_on,
                "event_step": event_step if event_step is not None else "",
                "steady_mean": metrics["steady_mean"],
                "steady_std": metrics["steady_std"],
                "band": metrics["band"],
                "settle_x": metrics["settle_x"],
                "delta_x": metrics["delta_x"],
                "settle_window": args.settle_window,
                "hold_window": args.hold_window,
                "tail_fraction": args.tail_fraction,
                "band_pct": args.band_pct,
                "band_abs_min": args.band_abs_min,
            })

            label_short = "MAPPO" if method == "MAPPO-OR" else "DRL"
            if event_step is None:
                annotation_lines.append(f"{label_short}: Ts={fmt_num(metrics['delta_x'], 1)}")
            else:
                annotation_lines.append(f"{label_short}: Tr={fmt_num(metrics['delta_x'], 1)}")

        if scenario in events:
            event_x = events[scenario] / args.x_divisor
            ax.axvline(event_x, color="black", linestyle="--", linewidth=1.1)
            ax.text(event_x, ax.get_ylim()[1], "event", rotation=90, va="top", ha="right", fontsize=8)

        if args.annotate_recovery and annotation_lines:
            ax.text(
                0.03, 0.03,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                fontsize=8.5,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.7")
            )

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

    print(f"Saved figure: {args.out.resolve()}")

    # print compact report
    print("\nRecovery / settling report")
    print("-" * 78)
    for row in metric_rows:
        label = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        scen = SCENARIO_LABELS.get(str(row["scenario"]), str(row["scenario"]))
        kind = "settling" if row["scenario"] == "initialization" and row["event_step"] == "" else "recovery"
        print(
            f"{scen:15s} | {label:18s} | "
            f"{kind}={fmt_num(float(row['delta_x']), 1):>8s} | "
            f"steady={fmt_num(float(row['steady_mean']), 3):>8s} ± {fmt_num(float(row['steady_std']), 3):>8s}"
        )

    if args.metrics_out is not None:
        save_metrics_csv(args.metrics_out, metric_rows)
        print(f"Saved metrics CSV: {args.metrics_out.resolve()}")


if __name__ == "__main__":
    main()
