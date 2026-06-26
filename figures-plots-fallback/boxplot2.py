from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# تنظیمات
# ============================================================

LOG_ROOT = Path("./gat-mappo/log")
OUTPUT_DIR = Path("./figures/table55_statistical_results")

TAIL = 5000
BLOCK_SIZE = 100
N_BOOTSTRAP = 10_000
RANDOM_SEED = 42

FOLDERS = {
    ("Light", "OR-DRL"): "ppo_initialization",
    ("Light", "OR-MAPPO"): "mappo_initialization",
    ("Heavy", "OR-DRL"): "ppo_initialization_heavy",
    ("Heavy", "OR-MAPPO"): "mappo_initialization_heavy",
}

TYPE_LABELS = {
    0: "I",
    1: "II",
    2: "III",
    3: "IV",
}

# معیارهایی که در جدول 5.5 استفاده شده‌اند
TARGETS = [
    ("delay", 0),
    ("delay", 2),
    ("delay", 3),
    ("throughput", 1),
    ("throughput", 2),
    ("loss", 3),
]

METRIC_LABELS = {
    "delay": "Delay (ms)",
    "throughput": "Throughput (%)",
    "loss": "Packet loss (%)",
}


# ============================================================
# خواندن و آماده‌سازی داده
# ============================================================

def read_log(path: Path, tail: int = 5000) -> np.ndarray:
    """اعداد موجود در یک فایل لاگ را می‌خواند."""

    values: list[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                # فایل‌های این پروژه در هر خط یک مقدار دارند
                value = float(text.split()[-1])
                values.append(value)
            except ValueError:
                print(
                    f"[WARN] Invalid numeric value ignored: "
                    f"{path}, line {line_number}"
                )

    if not values:
        raise ValueError(f"No numeric values found in {path}")

    array = np.asarray(values, dtype=float)

    if tail > 0:
        array = array[-tail:]

    return array


def convert_units(values: np.ndarray, metric: str) -> np.ndarray:
    """
    delay بر حسب میلی‌ثانیه ذخیره شده است.

    throughput و loss در لاگ‌های پروژه به صورت نسبت ذخیره شده‌اند؛
    بنابراین برای گزارش درصدی در 100 ضرب می‌شوند.
    """

    values = np.asarray(values, dtype=float)

    if metric in {"throughput", "loss"}:
        return values * 100.0

    return values


def calculate_block_means(
    values: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    نمونه‌های متوالی را به بلوک‌های غیرهم‌پوشان تقسیم می‌کند
    و میانگین هر بلوک را برمی‌گرداند.
    """

    values = np.asarray(values, dtype=float)

    number_of_complete_blocks = len(values) // block_size

    if number_of_complete_blocks < 2:
        raise ValueError(
            f"At least two complete blocks are required. "
            f"Received {len(values)} values with block size {block_size}."
        )

    usable_length = number_of_complete_blocks * block_size
    values = values[:usable_length]

    return values.reshape(number_of_complete_blocks, block_size).mean(axis=1)


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    دو DataFrame تولید می‌کند:

    raw_df:
        همه نمونه‌های خام استفاده‌شده از لاگ‌ها

    block_df:
        میانگین بلوک‌های 100 نمونه‌ای برای box plot و CI
    """

    raw_rows: list[dict] = []
    block_rows: list[dict] = []

    for (load, method), folder_name in FOLDERS.items():
        folder_path = LOG_ROOT / folder_name

        if not folder_path.is_dir():
            raise FileNotFoundError(
                f"Folder not found: {folder_path.resolve()}"
            )

        for metric, flow_type in TARGETS:
            filename = f"{metric}_type{flow_type}.log"
            file_path = folder_path / filename

            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Log file not found: {file_path.resolve()}"
                )

            values = read_log(file_path, tail=TAIL)
            values = convert_units(values, metric)

            block_means = calculate_block_means(
                values,
                block_size=BLOCK_SIZE,
            )

            for index, value in enumerate(values):
                raw_rows.append(
                    {
                        "Load": load,
                        "Method": method,
                        "Metric": metric,
                        "Type": TYPE_LABELS[flow_type],
                        "Sample": index + 1,
                        "Value": float(value),
                    }
                )

            for block_index, value in enumerate(block_means):
                block_rows.append(
                    {
                        "Load": load,
                        "Method": method,
                        "Metric": metric,
                        "Type": TYPE_LABELS[flow_type],
                        "Block": block_index + 1,
                        "Value": float(value),
                    }
                )

            print(
                f"[OK] {folder_name}/{filename}: "
                f"{len(values)} raw values, "
                f"{len(block_means)} block means"
            )

    return pd.DataFrame(raw_rows), pd.DataFrame(block_rows)


# ============================================================
# خلاصه آمار توصیفی
# ============================================================

def create_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df.groupby(
            ["Load", "Method", "Metric", "Type"],
            as_index=False,
        )
        .agg(
            Count=("Value", "count"),
            Mean=("Value", "mean"),
            Std=("Value", "std"),
            Median=("Value", "median"),
            Q1=("Value", lambda x: x.quantile(0.25)),
            Q3=("Value", lambda x: x.quantile(0.75)),
            Minimum=("Value", "min"),
            Maximum=("Value", "max"),
        )
    )

    summary["Mean"] = summary["Mean"].round(4)
    summary["Std"] = summary["Std"].round(4)
    summary["Median"] = summary["Median"].round(4)
    summary["Q1"] = summary["Q1"].round(4)
    summary["Q3"] = summary["Q3"].round(4)
    summary["Minimum"] = summary["Minimum"].round(4)
    summary["Maximum"] = summary["Maximum"].round(4)

    return summary


# ============================================================
# Bootstrap confidence interval
# ============================================================

def bootstrap_difference_ci(
    mappo_values: np.ndarray,
    drl_values: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    فاصله اطمینان bootstrap برای اختلاف میانگین:

        Mean(OR-MAPPO) - Mean(OR-DRL)

    را محاسبه می‌کند.
    """

    rng = np.random.default_rng(seed)

    mappo_values = np.asarray(mappo_values, dtype=float)
    drl_values = np.asarray(drl_values, dtype=float)

    bootstrap_differences = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sampled_mappo = rng.choice(
            mappo_values,
            size=len(mappo_values),
            replace=True,
        )

        sampled_drl = rng.choice(
            drl_values,
            size=len(drl_values),
            replace=True,
        )

        bootstrap_differences[index] = (
            sampled_mappo.mean() - sampled_drl.mean()
        )

    alpha = 1.0 - confidence

    lower = np.quantile(bootstrap_differences, alpha / 2.0)
    upper = np.quantile(
        bootstrap_differences,
        1.0 - alpha / 2.0,
    )

    observed_difference = (
        mappo_values.mean() - drl_values.mean()
    )

    return (
        float(observed_difference),
        float(lower),
        float(upper),
    )


def interpret_difference(
    metric: str,
    difference: float,
    ci_low: float,
    ci_high: float,
) -> str:
    significant = not (ci_low <= 0 <= ci_high)

    if not significant:
        return "Not statistically significant"

    if metric in {"delay", "loss"}:
        return (
            "OR-MAPPO better"
            if difference < 0
            else "OR-DRL better"
        )

    if metric == "throughput":
        return (
            "OR-MAPPO better"
            if difference > 0
            else "OR-DRL better"
        )

    return "Unknown"


def create_ci_table(block_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for load in ["Light", "Heavy"]:
        for metric, flow_type in TARGETS:
            type_label = TYPE_LABELS[flow_type]

            selection = block_df[
                (block_df["Load"] == load)
                & (block_df["Metric"] == metric)
                & (block_df["Type"] == type_label)
            ]

            mappo_values = selection[
                selection["Method"] == "OR-MAPPO"
            ]["Value"].to_numpy()

            drl_values = selection[
                selection["Method"] == "OR-DRL"
            ]["Value"].to_numpy()

            if len(mappo_values) == 0 or len(drl_values) == 0:
                raise ValueError(
                    f"Missing observations for "
                    f"{load}, {metric}, Type {type_label}"
                )

            difference, ci_low, ci_high = bootstrap_difference_ci(
                mappo_values=mappo_values,
                drl_values=drl_values,
                n_bootstrap=N_BOOTSTRAP,
                confidence=0.95,
                seed=RANDOM_SEED,
            )

            mean_mappo = float(mappo_values.mean())
            mean_drl = float(drl_values.mean())

            if mean_drl != 0:
                relative_change = (
                    (mean_mappo - mean_drl) / mean_drl
                ) * 100.0
            else:
                relative_change = np.nan

            rows.append(
                {
                    "Load": load,
                    "Metric": metric,
                    "Type": type_label,
                    "Mean_OR_MAPPO": mean_mappo,
                    "Mean_OR_DRL": mean_drl,
                    "Difference_MAPPO_minus_DRL": difference,
                    "CI95_Low": ci_low,
                    "CI95_High": ci_high,
                    "Relative_change_percent": relative_change,
                    "CI_excludes_zero": not (
                        ci_low <= 0 <= ci_high
                    ),
                    "Interpretation": interpret_difference(
                        metric,
                        difference,
                        ci_low,
                        ci_high,
                    ),
                    "N_blocks_MAPPO": len(mappo_values),
                    "N_blocks_DRL": len(drl_values),
                }
            )

    result = pd.DataFrame(rows)

    numeric_columns = [
        "Mean_OR_MAPPO",
        "Mean_OR_DRL",
        "Difference_MAPPO_minus_DRL",
        "CI95_Low",
        "CI95_High",
        "Relative_change_percent",
    ]

    result[numeric_columns] = result[numeric_columns].round(4)

    return result


# ============================================================
# Box plots
# ============================================================

def create_boxplot(
    block_df: pd.DataFrame,
    metric: str,
    flow_type: str,
) -> None:
    selection = block_df[
        (block_df["Metric"] == metric)
        & (block_df["Type"] == flow_type)
    ]

    order = [
        ("Light", "OR-DRL"),
        ("Light", "OR-MAPPO"),
        ("Heavy", "OR-DRL"),
        ("Heavy", "OR-MAPPO"),
    ]

    labels = [
        "DRL\nLight",
        "MAPPO\nLight",
        "DRL\nHeavy",
        "MAPPO\nHeavy",
    ]

    data = []

    for load, method in order:
        values = selection[
            (selection["Load"] == load)
            & (selection["Method"] == method)
        ]["Value"].to_numpy()

        if len(values) == 0:
            raise ValueError(
                f"No values found for "
                f"{load}, {method}, {metric}, Type {flow_type}"
            )

        data.append(values)

    figure, axis = plt.subplots(figsize=(7.2, 5.2))

    axis.boxplot(
        data,
        labels=labels,
        whis=1.5,
        showfliers=True,
        showmeans=True,
        meanline=True,
    )

    axis.set_ylabel(METRIC_LABELS[metric])
    axis.set_xlabel("Method and traffic load")
    axis.set_title(
        f"{METRIC_LABELS[metric]} — Service Type {flow_type}"
    )

    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    base_filename = f"boxplot_{metric}_type_{flow_type}"

    png_path = OUTPUT_DIR / f"{base_filename}.png"
    pdf_path = OUTPUT_DIR / f"{base_filename}.pdf"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


# ============================================================
# اجرای اصلی
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, block_df = collect_data()

    raw_df.to_csv(
        OUTPUT_DIR / "table55_raw_values.csv",
        index=False,
    )

    block_df.to_csv(
        OUTPUT_DIR / "table55_block_means.csv",
        index=False,
    )

    summary_df = create_summary_table(raw_df)
    summary_df.to_csv(
        OUTPUT_DIR / "table55_summary_from_logs.csv",
        index=False,
    )

    ci_df = create_ci_table(block_df)
    ci_df.to_csv(
        OUTPUT_DIR / "table55_ci_block_bootstrap.csv",
        index=False,
    )

    for metric, flow_type_number in TARGETS:
        create_boxplot(
            block_df=block_df,
            metric=metric,
            flow_type=TYPE_LABELS[flow_type_number],
        )

    print("\nAnalysis completed.")
    print(f"Results directory: {OUTPUT_DIR.resolve()}")
    print(
        "Check table55_summary_from_logs.csv before using "
        "the figures in the thesis."
    )


if __name__ == "__main__":
    main()