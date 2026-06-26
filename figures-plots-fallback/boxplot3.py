from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# تنظیمات
# ============================================================

# اگر فولدر log کنار این اسکریپت است:
# LOG_ROOT = Path("./log")

# اگر اسکریپت در ریشه ریپازیتوری است، ممکن است لازم باشد بنویسید:
LOG_ROOT = Path("./gat-mappo/log")

OUTPUT_DIR = Path("./figures/table55_statistical_results_new")

TAIL = 5000
BLOCK_SIZE = 100

FOLDERS = {
    ("Light", "DRL-OR"): "ppo_initialization",
    ("Light", "MAPPO-OR"): "mappo_initialization",
    ("Heavy", "DRL-OR"): "ppo_initialization_heavy",
    ("Heavy", "MAPPO-OR"): "mappo_initialization_heavy",
}

TYPE_LABELS = {
    0: "I",
    1: "II",
    2: "III",
    3: "IV",
}

# معیارهای مورد استفاده در جدول 5.5
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

# ترتیب زیرشکل‌ها در شکل مرکب
PANEL_SPECS = [
    {
        "metric": "delay",
        "type": "I",
        "title": "(a) Delay — Service Type I",
        "ylabel": "Delay (ms)",
        "ylim": (0, 210),
    },
    {
        "metric": "delay",
        "type": "III",
        "title": "(b) Delay — Service Type III",
        "ylabel": "Delay (ms)",
        "ylim": (0, 210),
    },
    {
        "metric": "delay",
        "type": "IV",
        "title": "(c) Delay — Service Type IV",
        "ylabel": "Delay (ms)",
        "ylim": (0, 210),
    },
    {
        "metric": "throughput",
        "type": "II",
        "title": "(d) Throughput — Service Type II",
        "ylabel": "Throughput (%)",
        "ylim": (70, 102),
    },
    {
        "metric": "throughput",
        "type": "III",
        "title": "(e) Throughput — Service Type III",
        "ylabel": "Throughput (%)",
        "ylim": (70, 102),
    },
    {
        "metric": "loss",
        "type": "IV",
        "title": "(f) Packet Loss — Service Type IV",
        "ylabel": "Packet loss (%)",
        "ylim": (0, 25.5),
    },
]


# ============================================================
# خواندن داده‌ها
# ============================================================

def read_log(path: Path, tail: int = 5000) -> np.ndarray:
    """
    مقادیر عددی یک فایل لاگ را می‌خواند.
    فرض بر این است که در هر خط یک مقدار عددی وجود دارد.
    """

    values: list[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                value = float(text.split()[-1])
                values.append(value)
            except ValueError:
                print(
                    f"[WARN] Ignored invalid value: "
                    f"{path}, line {line_number}"
                )

    if not values:
        raise ValueError(f"No numeric values found in: {path}")

    array = np.asarray(values, dtype=float)

    if tail > 0:
        array = array[-tail:]

    return array


def convert_units(values: np.ndarray, metric: str) -> np.ndarray:
    """
    delay در لاگ بر حسب میلی‌ثانیه است.

    throughput و loss در لاگ‌ها به صورت نسبت ذخیره شده‌اند؛
    بنابراین بدون استفاده از شرط max یا median، همیشه در 100
    ضرب می‌شوند.
    """

    values = np.asarray(values, dtype=float)

    if metric in {"throughput", "loss"}:
        values = values * 100.0

    return values


def calculate_block_means(
    values: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    داده‌ها را به بلوک‌های غیرهم‌پوشان تقسیم کرده و میانگین
    هر بلوک را محاسبه می‌کند.
    """

    values = np.asarray(values, dtype=float)

    number_of_blocks = len(values) // block_size

    if number_of_blocks < 2:
        raise ValueError(
            f"At least two blocks are needed. "
            f"Received {len(values)} samples with "
            f"block size {block_size}."
        )

    usable_length = number_of_blocks * block_size
    values = values[:usable_length]

    return values.reshape(number_of_blocks, block_size).mean(axis=1)


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    داده‌های خام و میانگین‌های بلوکی را جمع‌آوری می‌کند.
    """

    raw_rows: list[dict] = []
    block_rows: list[dict] = []

    for (load, method), folder_name in FOLDERS.items():
        folder_path = LOG_ROOT / folder_name

        if not folder_path.is_dir():
            raise FileNotFoundError(
                f"Folder not found: {folder_path.resolve()}"
            )

        for metric, flow_type_number in TARGETS:
            filename = f"{metric}_type{flow_type_number}.log"
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

            type_label = TYPE_LABELS[flow_type_number]

            for sample_number, value in enumerate(values, start=1):
                raw_rows.append(
                    {
                        "Load": load,
                        "Method": method,
                        "Metric": metric,
                        "Type": type_label,
                        "Sample": sample_number,
                        "Value": float(value),
                    }
                )

            for block_number, value in enumerate(
                block_means,
                start=1,
            ):
                block_rows.append(
                    {
                        "Load": load,
                        "Method": method,
                        "Metric": metric,
                        "Type": type_label,
                        "Block": block_number,
                        "Value": float(value),
                    }
                )

            print(
                f"[OK] {folder_name}/{filename}: "
                f"{len(values)} raw samples, "
                f"{len(block_means)} block means"
            )

    raw_df = pd.DataFrame(raw_rows)
    block_df = pd.DataFrame(block_rows)

    return raw_df, block_df


# ============================================================
# جدول خلاصه
# ============================================================

def create_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    آمار توصیفی داده‌های خام را تولید می‌کند.
    """

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

    numeric_columns = [
        "Mean",
        "Std",
        "Median",
        "Q1",
        "Q3",
        "Minimum",
        "Maximum",
    ]

    summary[numeric_columns] = summary[numeric_columns].round(4)

    return summary


def create_block_summary_table(
    block_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    آمار توصیفی میانگین‌های بلوکی را تولید می‌کند.
    """

    summary = (
        block_df.groupby(
            ["Load", "Method", "Metric", "Type"],
            as_index=False,
        )
        .agg(
            Number_of_blocks=("Value", "count"),
            Mean_of_blocks=("Value", "mean"),
            Std_of_blocks=("Value", "std"),
            Median_of_blocks=("Value", "median"),
            Q1_of_blocks=("Value", lambda x: x.quantile(0.25)),
            Q3_of_blocks=("Value", lambda x: x.quantile(0.75)),
        )
    )

    numeric_columns = [
        "Mean_of_blocks",
        "Std_of_blocks",
        "Median_of_blocks",
        "Q1_of_blocks",
        "Q3_of_blocks",
    ]

    summary[numeric_columns] = summary[numeric_columns].round(4)

    return summary


# ============================================================
# ترسیم هر زیرشکل
# ============================================================

def draw_boxplot_panel(
    axis: plt.Axes,
    block_df: pd.DataFrame,
    metric: str,
    flow_type: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    """
    یک زیرشکل شامل چهار جعبه را ترسیم می‌کند.
    """

    selection = block_df[
        (block_df["Metric"] == metric)
        & (block_df["Type"] == flow_type)
    ]

    order = [
        ("Light", "DRL-OR"),
        ("Light", "MAPPO-OR"),
        ("Heavy", "DRL-OR"),
        ("Heavy", "MAPPO-OR"),
    ]

    labels = [
        "DRL-OR\nLight",
        "MAPPO-OR\nLight",
        "DRL-OR\nHeavy",
        "MAPPO-OR\nHeavy",
    ]

    data: list[np.ndarray] = []

    for load, method in order:
        values = selection[
            (selection["Load"] == load)
            & (selection["Method"] == method)
        ]["Value"].to_numpy()

        if len(values) == 0:
            raise ValueError(
                f"No data for: {load}, {method}, "
                f"{metric}, Type {flow_type}"
            )

        data.append(values)

    axis.boxplot(
        data,
        labels=labels,
        widths=0.55,
        whis=1.5,
        showfliers=True,
        showmeans=True,
        meanline=True,
    )

    axis.set_title(title, fontsize=13)
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.tick_params(
        axis="x",
        labelsize=9,
    )

    axis.tick_params(
        axis="y",
        labelsize=9,
    )


# ============================================================
# شکل مرکب 6 قسمتی
# ============================================================

def create_combined_figure(block_df: pd.DataFrame) -> None:
    """
    شش نمودار را در یک شکل 3×2 قرار می‌دهد.
    """

    figure, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(13, 15),
    )

    axes_flat = axes.flatten()

    for axis, specification in zip(
        axes_flat,
        PANEL_SPECS,
    ):
        draw_boxplot_panel(
            axis=axis,
            block_df=block_df,
            metric=specification["metric"],
            flow_type=specification["type"],
            title=specification["title"],
            ylabel=specification["ylabel"],
            ylim=specification["ylim"],
        )

    figure.suptitle(
        "Distribution of Performance Metrics for "
        "DRL-OR and MAPPO-OR",
        fontsize=17,
        y=0.995,
    )

    figure.text(
        0.5,
        0.012,
        "Each observation is the mean of a non-overlapping "
        "block of 100 consecutive samples. "
        "Solid line: median; dashed line: mean.",
        ha="center",
        fontsize=10,
    )

    figure.tight_layout(
        rect=(0, 0.035, 1, 0.975),
        h_pad=2.6,
        w_pad=2.0,
    )

    png_path = (
        OUTPUT_DIR
        / "table55_combined_boxplots.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / "table55_combined_boxplots.pdf"
    )

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

    print(f"[OK] Combined PNG saved: {png_path}")
    print(f"[OK] Combined PDF saved: {pdf_path}")


# ============================================================
# اجرای اصلی
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_df, block_df = collect_data()

    raw_df.to_csv(
        OUTPUT_DIR / "table55_raw_values.csv",
        index=False,
    )

    block_df.to_csv(
        OUTPUT_DIR / "table55_block_means.csv",
        index=False,
    )

    raw_summary = create_summary_table(raw_df)
    raw_summary.to_csv(
        OUTPUT_DIR / "table55_summary_from_logs.csv",
        index=False,
    )

    block_summary = create_block_summary_table(block_df)
    block_summary.to_csv(
        OUTPUT_DIR / "table55_block_summary.csv",
        index=False,
    )

    create_combined_figure(block_df)

    print("\nAnalysis completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()