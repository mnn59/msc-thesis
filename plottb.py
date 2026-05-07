#!/usr/bin/env python3
"""
Plot MAPPO-OR thesis figures directly from TensorBoard event files.

Designed for this directory layout:

log/
  mappo_initialization/
    tb/
      events.out.tfevents....
  mappo_traffic_change/
    tb/
      events.out.tfevents....
  mappo_link_failure/
    tb/
      events.out.tfevents....

Main features:
- Lists TensorBoard scalar tags.
- Auto-detects tags for delay/throughput/loss/stretch/reward by type.
- Uses real TensorBoard steps, not just line index.
- Applies TensorBoard-like exponential smoothing.
- Can crop all service-type curves in a panel to their common x-range, so one line does not look "incomplete".
- Can plot Fig.5-style 2x3 or 3x3 panels.

Install dependency if needed:
    pip install tensorboard matplotlib numpy

Examples:
    # 1) First inspect available tags:
    python plot_mappo_tb_from_events.py --root ./log --list-tags

    # 2) Plot delay + throughput like Fig.5:
    python plot_mappo_tb_from_events.py --root ./log --metrics delay throughput --num-types 3 --out ./figures_tb/fig5_tb.png

    # 3) Plot delay + throughput + stretch:
    python plot_mappo_tb_from_events.py --root ./log --metrics delay throughput stretch --num-types 3 --out ./figures_tb/fig5_tb_with_stretch.png

    # 4) If the blue curve is shorter and you want to avoid an incomplete-looking thesis figure:
    python plot_mappo_tb_from_events.py --root ./log --metrics delay throughput --num-types 3 --crop-common-x --out ./figures_tb/fig5_tb_cropped.png

    # 5) If automatic tag matching fails, provide explicit tag names:
    python plot_mappo_tb_from_events.py --root ./log --metrics delay throughput --num-types 3 \
      --tag delay:type0:your/tag/name0 --tag delay:type1:your/tag/name1 --tag delay:type2:your/tag/name2 \
      --out ./figures_tb/fig5_tb_manual.png
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:
    raise SystemExit(
        "TensorBoard is required for reading event files. Install it with:\n"
        "    pip install tensorboard\n"
        f"Original import error: {exc}"
    )


SCENARIO_DIRS = {
    "initialization": "mappo_initialization",
    "link_failure": "mappo_link_failure",
    "traffic_change": "mappo_traffic_change",
}

SCENARIO_TITLES = {
    "initialization": "Initialization",
    "link_failure": "Link failure",
    "traffic_change": "Traffic change",
}

TYPE_LABELS = {
    0: "Type I",
    1: "Type II",
    2: "Type III",
    3: "Type IV",
}

TYPE_PATTERNS = {
    0: ["type0", "type_0", "typei", "type_i", "service0", "service_0", "eta0", "eta_0"],
    1: ["type1", "type_1", "typeii", "type_ii", "service1", "service_1", "eta1", "eta_1"],
    2: ["type2", "type_2", "typeiii", "type_iii", "service2", "service_2", "eta2", "eta_2"],
    3: ["type3", "type_3", "typeiv", "type_iv", "service3", "service_3", "eta3", "eta_3"],
}

METRIC_ALIASES = {
    "delay": ["delay", "latency", "avgdelay", "avglatency", "average_delay", "average_latency"],
    "latency": ["delay", "latency", "avgdelay", "avglatency", "average_delay", "average_latency"],
    "throughput": ["throughput", "throughputratio", "throughput_ratio", "thr", "avgthroughput"],
    "loss": ["loss", "lossratio", "loss_ratio", "packetloss", "packet_loss", "avgloss"],
    "stretch": ["stretch", "pathstretch", "path_stretch", "deltadist", "delta_dist", "deltadistance", "delta_distance"],
    "reward": ["reward", "globalrwd", "globalreward", "global_reward", "rwd"],
    "fallback": ["fallback", "circle", "safe", "trigger", "safe_trigger", "fallback_trigger"],
}

METRIC_YLABEL = {
    "delay": "Average latency (ms)",
    "latency": "Average latency (ms)",
    "throughput": "Throughput ratio",
    "loss": "Average loss ratio",
    "stretch": "Path stretch",
    "reward": "Global reward",
    "fallback": "Fallback trigger ratio",
}


def normalize_tag(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def scenario_tb_dir(root: Path, scenario: str) -> Path:
    return root / SCENARIO_DIRS[scenario] / "tb"


def load_event_accumulator(tb_dir: Path) -> EventAccumulator:
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def scalar_tags(acc: EventAccumulator) -> List[str]:
    return sorted(acc.Tags().get("scalars", []))


def parse_manual_tags(items: List[str]) -> Dict[Tuple[str, int], str]:
    """
    Parse items like:
      delay:type0:tag/name
      throughput:1:tag/name
    """
    result: Dict[Tuple[str, int], str] = {}
    for item in items:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid --tag format: {item}. Use metric:type_id:tag_name")
        metric, type_part, tag = parts
        metric = metric.strip()
        m = re.search(r"\d+", type_part)
        if not m:
            raise ValueError(f"Cannot parse type id from: {type_part}")
        type_id = int(m.group(0))
        result[(metric, type_id)] = tag
    return result


def find_best_tag(tags: List[str], metric: str, type_id: int, manual: Dict[Tuple[str, int], str]) -> Optional[str]:
    if (metric, type_id) in manual:
        return manual[(metric, type_id)]

    aliases = [normalize_tag(a) for a in METRIC_ALIASES.get(metric, [metric])]
    type_patterns = [normalize_tag(p) for p in TYPE_PATTERNS.get(type_id, [f"type{type_id}"])]

    candidates: List[Tuple[int, str]] = []
    for tag in tags:
        nt = normalize_tag(tag)

        metric_score = 0
        for a in aliases:
            if a and a in nt:
                metric_score = max(metric_score, len(a))

        type_score = 0
        for p in type_patterns:
            if p and p in nt:
                type_score = max(type_score, len(p))

        if metric_score > 0 and type_score > 0:
            # Prefer tags where both matches are strong and the tag is not too generic.
            score = metric_score * 10 + type_score - len(nt) * 0.001
            candidates.append((int(score * 1000), tag))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def read_scalar(acc: EventAccumulator, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    events = acc.Scalars(tag)
    x = np.asarray([e.step for e in events], dtype=float)
    y = np.asarray([e.value for e in events], dtype=float)

    # Sort just in case.
    order = np.argsort(x)
    return x[order], y[order]


def ema(y: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0 or len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    alpha = smoothing
    for i in range(1, len(y)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * y[i]
    return out


def crop_to_common_range(curves: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    if not curves:
        return curves
    start = max(float(x[0]) for x, _ in curves.values() if len(x))
    end = min(float(x[-1]) for x, _ in curves.values() if len(x))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return curves

    cropped: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for type_id, (x, y) in curves.items():
        mask = (x >= start) & (x <= end)
        cropped[type_id] = (x[mask], y[mask])
    return cropped


def plot_fig(
    data: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, str]],
    scenarios: List[str],
    metrics: List[str],
    num_types: int,
    out: Path,
    smoothing: float,
    crop_common_x: bool,
    x_divisor: float,
    event_steps: Dict[str, float],
    title: str,
) -> None:
    n_rows = len(metrics)
    n_cols = len(scenarios)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.7 * n_rows), squeeze=False)

    for r, metric in enumerate(metrics):
        for c, scenario in enumerate(scenarios):
            ax = axes[r][c]

            curves: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            source_tags: Dict[int, str] = {}

            for type_id in range(num_types):
                key = (scenario, metric, type_id)
                if key in data:
                    x, y, tag = data[key]
                    curves[type_id] = (x.copy(), y.copy())
                    source_tags[type_id] = tag

            if crop_common_x and len(curves) >= 2:
                curves = crop_to_common_range(curves)

            for type_id, (x, y) in curves.items():
                if len(x) == 0:
                    continue
                ys = ema(y, smoothing)
                ax.plot(x / x_divisor, ys, linewidth=1.35, label=TYPE_LABELS.get(type_id, f"Type {type_id}"))

            if scenario in event_steps:
                ax.axvline(event_steps[scenario] / x_divisor, linestyle="--", linewidth=1)
                ax.text(
                    event_steps[scenario] / x_divisor,
                    ax.get_ylim()[1],
                    "event",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=9,
                )

            ax.set_title(f"{METRIC_YLABEL.get(metric, metric)} — {SCENARIO_TITLES.get(scenario, scenario)}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Step" if x_divisor == 1 else "Step (×1,000)")
            ax.set_ylabel(METRIC_YLABEL.get(metric, metric))
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=9)

            if not curves:
                ax.text(0.5, 0.5, "No matching tag", transform=ax.transAxes, ha="center", va="center")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("./log"), help="Root log directory. Default: ./log")
    p.add_argument("--scenarios", nargs="+", default=["initialization", "link_failure", "traffic_change"])
    p.add_argument("--metrics", nargs="+", default=["delay", "throughput"], help="delay throughput loss stretch reward fallback")
    p.add_argument("--num-types", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("./figures_tb/fig5_tb.png"))
    p.add_argument("--smoothing", type=float, default=0.9, help="EMA smoothing like TensorBoard, e.g. 0.6 to 0.95")
    p.add_argument("--crop-common-x", action="store_true", help="Crop each panel to common x-range of all shown types")
    p.add_argument("--x-divisor", type=float, default=1000.0, help="Use 1000 for x-axis label Step ×1,000")
    p.add_argument("--title", default="MAPPO-OR with Safe Learning — TensorBoard Learning Curves")
    p.add_argument("--list-tags", action="store_true", help="Only list scalar tags in each scenario and exit")
    p.add_argument("--tag", action="append", default=[], help="Manual tag mapping: metric:type_id:tag_name")
    p.add_argument(
        "--event-step",
        action="append",
        default=[],
        help="Optional event line, e.g. link_failure:10000 or traffic_change:10000",
    )
    args = p.parse_args()

    manual = parse_manual_tags(args.tag)

    event_steps: Dict[str, float] = {}
    for item in args.event_step:
        scenario, value = item.split(":", 1)
        event_steps[scenario] = float(value)

    accs: Dict[str, EventAccumulator] = {}
    for scenario in args.scenarios:
        tb_dir = scenario_tb_dir(args.root, scenario)
        try:
            accs[scenario] = load_event_accumulator(tb_dir)
        except Exception as exc:
            print(f"[WARN] Could not load {scenario}: {exc}")

    if args.list_tags:
        for scenario, acc in accs.items():
            print(f"\n=== {scenario} | {scenario_tb_dir(args.root, scenario)} ===")
            tags = scalar_tags(acc)
            if not tags:
                print("No scalar tags found.")
            for t in tags:
                print(t)
        return

    data: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, str]] = {}

    print("\nSelected TensorBoard tags:")
    for scenario, acc in accs.items():
        tags = scalar_tags(acc)
        for metric in args.metrics:
            for type_id in range(args.num_types):
                tag = find_best_tag(tags, metric, type_id, manual)
                if tag is None:
                    print(f"  [MISSING] scenario={scenario}, metric={metric}, type={type_id}")
                    continue
                try:
                    x, y = read_scalar(acc, tag)
                except Exception as exc:
                    print(f"  [ERROR] {scenario} {metric} type{type_id}: {tag}: {exc}")
                    continue
                data[(scenario, metric, type_id)] = (x, y, tag)
                print(f"  {scenario:16s} {metric:12s} type{type_id}: {tag}  (n={len(y)}, step={x[0] if len(x) else 'NA'}..{x[-1] if len(x) else 'NA'})")

    plot_fig(
        data=data,
        scenarios=args.scenarios,
        metrics=args.metrics,
        num_types=args.num_types,
        out=args.out,
        smoothing=args.smoothing,
        crop_common_x=args.crop_common_x,
        x_divisor=args.x_divisor,
        event_steps=event_steps,
        title=args.title,
    )

    print(f"\nSaved figure: {args.out.resolve()}")


if __name__ == "__main__":
    main()
