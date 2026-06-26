# table55_boxplot.py

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FOLDERS = {
    "Abi_SHR_5000_lightload": ("SPR", "Light"),
    "Abi_SHR_5000_heavyload": ("SPR", "Heavy"),

    "Abi_WP_5000_lightload": ("LBR", "Light"),
    "Abi_WP_5000_heavyload": ("LBR", "Heavy"),

    "Abi_QoS_5000_lightload": ("QoSR", "Light"),
    "Abi_QoS_5000_heavyload": ("QoSR", "Heavy"),

    "ppo_initialization": ("DRL-OR", "Light"),
    "ppo_initialization_heavy": ("DRL-OR", "Heavy"),

    "mappo_initialization": ("MAPPO-OR", "Light"),
    "mappo_initialization_heavy": ("MAPPO-OR", "Heavy"),
}

TYPE_LABEL = {
    0: "I",
    1: "II",
    2: "III",
    3: "IV",
}

# دقیقاً معیارهای جدول 5.5
TARGETS = [
    ("delay", 0),       # Delay Type I
    ("delay", 2),       # Delay Type III
    ("delay", 3),       # Delay Type IV
    ("throughput", 1),  # Throughput Type II
    ("throughput", 2),  # Throughput Type III
    ("loss", 3),        # Loss Type IV
]


def read_values(path, tail=5000):
    vals = []
    number_pattern = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = number_pattern.findall(line)
            if not nums:
                continue

            # اگر هر خط فقط یک عدد باشد همان را می‌خواند.
            # اگر هر خط چند عدد داشته باشد، عدد آخر را metric فرض می‌کند.
            vals.append(float(nums[-1]))

    if tail is not None and tail > 0:
        vals = vals[-tail:]

    return vals


def collect_data(root, tail):
    rows = []

    for folder, (method, load) in FOLDERS.items():
        folder_path = os.path.join(root, folder)

        if not os.path.isdir(folder_path):
            print(f"[WARN] folder not found: {folder_path}")
            continue

        for metric, flow_type in TARGETS:
            file_path = os.path.join(folder_path, f"{metric}_type{flow_type}.log")

            if not os.path.isfile(file_path):
                print(f"[WARN] file not found: {file_path}")
                continue

            values = read_values(file_path, tail=tail)

            # اگر throughput/loss در بازه 0..1 ذخیره شده باشد، درصدی‌اش می‌کنیم.
            # if metric in ["throughput", "loss"] and values and max(values) <= 1.5:
            #     values = [v * 100 for v in values]
            if metric in ["throughput", "loss"] and np.nanmedian(values) < 2:
                values = [v * 100 for v in values]

            for v in values:
                rows.append({
                    "Load": load,
                    "Method": method,
                    "Metric": metric,
                    "Type": TYPE_LABEL[flow_type],
                    "Value": v,
                    "Source": file_path,
                })

            print(f"[OK] {folder}/{metric}_type{flow_type}.log -> {len(values)} values")

    return pd.DataFrame(rows)


def save_summary(df, out_dir):
    summary = (
        df.groupby(["Load", "Method", "Metric", "Type"])
        .agg(
            Count=("Value", "count"),
            Mean=("Value", "mean"),
            Std=("Value", "std"),
            Median=("Value", "median"),
            Q1=("Value", lambda x: x.quantile(0.25)),
            Q3=("Value", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    path = os.path.join(out_dir, "table55_summary_check.csv")
    summary.to_csv(path, index=False)
    print(f"[OK] saved: {path}")


def make_boxplot(df, metric, types, ylabel, out_dir):
    sub = df[(df["Metric"] == metric) & (df["Type"].isin(types))].copy()

    if sub.empty:
        print(f"[WARN] no data for {metric}")
        return

    sub["Group"] = (
        sub["Load"] + " | Type " + sub["Type"] + " | " + sub["Method"]
    )

    groups = sorted(sub["Group"].unique())
    data = [sub[sub["Group"] == g]["Value"].values for g in groups]

    plt.figure(figsize=(max(12, 0.55 * len(groups)), 6))
    plt.boxplot(data, labels=groups, showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(f"Box plot of {metric} values for Table 5.5")
    plt.tight_layout()

    path = os.path.join(out_dir, f"table55_boxplot_{metric}.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[OK] saved: {path}")


def bootstrap_ci_diff(a, b, n_boot=5000, seed=1):
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)

    diffs = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(aa) - np.mean(bb))

    return (
        float(np.mean(a) - np.mean(b)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def save_ci_mappo_vs_drl(df, out_dir):
    rows = []

    for load in ["Light", "Heavy"]:
        for metric, flow_type in TARGETS:
            type_label = TYPE_LABEL[flow_type]

            sub = df[
                (df["Load"] == load)
                & (df["Metric"] == metric)
                & (df["Type"] == type_label)
            ]

            mappo = sub[sub["Method"] == "MAPPO-OR"]["Value"].values
            drl = sub[sub["Method"] == "DRL-OR"]["Value"].values

            if len(mappo) < 2 or len(drl) < 2:
                continue

            diff, low, high = bootstrap_ci_diff(mappo, drl)

            rows.append({
                "Load": load,
                "Metric": metric,
                "Type": type_label,
                "Mean_MAPPO_OR": np.mean(mappo),
                "Mean_DRL_OR": np.mean(drl),
                "Diff_MAPPO_minus_DRL": diff,
                "CI95_Low": low,
                "CI95_High": high,
                "CI_excludes_zero": not (low <= 0 <= high),
                "N_MAPPO": len(mappo),
                "N_DRL": len(drl),
            })

    ci = pd.DataFrame(rows)
    path = os.path.join(out_dir, "table55_ci_or_mappo_vs_or_drl.csv")
    ci.to_csv(path, index=False)
    print(f"[OK] saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./gat-mappo/log")
    parser.add_argument("--tail", type=int, default=5000)
    parser.add_argument("--out", default="./figures/table55_boxplots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = collect_data(args.root, args.tail)

    if df.empty:
        print("[ERROR] no data collected. Check --root path.")
        return

    raw_path = os.path.join(args.out, "table55_raw_values.csv")
    df.to_csv(raw_path, index=False)
    print(f"[OK] saved: {raw_path}")

    save_summary(df, args.out)

    make_boxplot(df, "delay", ["I", "III", "IV"], "Delay (ms)", args.out)
    make_boxplot(df, "throughput", ["II", "III"], "Throughput (%)", args.out)
    make_boxplot(df, "loss", ["IV"], "Loss (%)", args.out)

    save_ci_mappo_vs_drl(df, args.out)


if __name__ == "__main__":
    main()