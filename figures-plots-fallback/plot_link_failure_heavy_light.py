#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUM_TYPES = 3  # اگر type4 هم داری، این را 4 کن

TYPE_LABELS = {
    0: "Type 1",
    1: "Type 2",
    2: "Type 3",
    3: "Type 4",
}

ABILENE_AVG_SHR_DIST = 2.4182


def load_log(filepath):
    vals = []
    if not os.path.exists(filepath):
        print(f"[WARN] Missing file: {filepath}")
        return np.array([])

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue

    return np.array(vals)


def get_converged_mean(data, last_n=2000):
    if len(data) == 0:
        return np.nan
    n = min(last_n, len(data))
    return float(np.mean(data[-n:]))


def compute_stretch(delta_dist_array, avg_shr_dist=ABILENE_AVG_SHR_DIST):
    dd = np.array(delta_dist_array, dtype=float)
    dd = np.maximum(dd, 0)
    return 1.0 + dd / avg_shr_dist


def collect_metric(log_root, subdirs, metric, last_n):
    """
    output:
        data[load_name][type_id] = mean_value
    """
    data = {}

    for load_name, subdir in subdirs.items():
        data[load_name] = {}

        for t in range(NUM_TYPES):
            if metric == "stretch":
                fpath = os.path.join(log_root, subdir, f"dist_type{t}.log")
                vals = load_log(fpath)
                if len(vals) > 0:
                    vals = compute_stretch(vals)
            else:
                fpath = os.path.join(log_root, subdir, f"{metric}_type{t}.log")
                vals = load_log(fpath)

            mean_val = get_converged_mean(vals, last_n=last_n)

            if metric == "loss" and not np.isnan(mean_val):
                mean_val *= 100.0

            data[load_name][t] = mean_val

    return data


def plot_grouped_bar(data, metric, ylabel, title, save_path):
    loads = list(data.keys())
    x = np.arange(NUM_TYPES)

    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, load_name in enumerate(loads):
        values = [data[load_name].get(t, np.nan) for t in range(NUM_TYPES)]
        offsets = x + (i - (len(loads) - 1) / 2) * bar_width

        ax.bar(
            offsets,
            values,
            bar_width,
            label=load_name,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
        )

        for ox, val in zip(offsets, values):
            if not np.isnan(val):
                ax.text(
                    ox,
                    val,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax.set_xlabel("Flow Type", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[t] for t in range(NUM_TYPES)])
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


def save_summary_csv(all_results, save_path):
    with open(save_path, "w") as f:
        f.write("metric,load,type,value\n")

        for metric, data in all_results.items():
            for load_name, type_values in data.items():
                for t, value in type_values.items():
                    f.write(f"{metric},{load_name},type{t},{value}\n")

    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-root",
        type=str,
        default="./gat-mappo/log",
        help="Root log directory",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="./figures_ppo_light_heavy_link_failure",
        help="Directory for saving figures",
    )

    parser.add_argument(
        "--last-n",
        type=int,
        default=2000,
        help="Number of last samples used for averaging",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    subdirs = {
        "Light Load": "ppo_link_failure",
        "Heavy Load": "ppo_link_failure_heavy",
    }

    metrics = {
        "delay": {
            "ylabel": "Average Latency (ms)",
            "title": "PPO Link Failure: Light Load vs Heavy Load - Latency",
            "filename": "ppo_link_failure_light_heavy_delay.png",
        },
        "throughput": {
            "ylabel": "Throughput Ratio",
            "title": "PPO Link Failure: Light Load vs Heavy Load - Throughput",
            "filename": "ppo_link_failure_light_heavy_throughput.png",
        },
        "loss": {
            "ylabel": "Average Packet Loss Ratio (%)",
            "title": "PPO Link Failure: Light Load vs Heavy Load - Packet Loss",
            "filename": "ppo_link_failure_light_heavy_loss.png",
        },
        "stretch": {
            "ylabel": "Average Stretch",
            "title": "PPO Link Failure: Light Load vs Heavy Load - Stretch",
            "filename": "ppo_link_failure_light_heavy_stretch.png",
        },
    }

    all_results = {}

    for metric, info in metrics.items():
        data = collect_metric(
            log_root=args.log_root,
            subdirs=subdirs,
            metric=metric,
            last_n=args.last_n,
        )

        all_results[metric] = data

        save_path = os.path.join(args.save_dir, info["filename"])

        plot_grouped_bar(
            data=data,
            metric=metric,
            ylabel=info["ylabel"],
            title=info["title"],
            save_path=save_path,
        )

    csv_path = os.path.join(args.save_dir, "ppo_link_failure_light_heavy_summary.csv")
    save_summary_csv(all_results, csv_path)


if __name__ == "__main__":
    main()