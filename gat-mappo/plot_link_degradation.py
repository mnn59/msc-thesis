#!/usr/bin/env python3
"""
Plot link_degradation scenario logs for both PPO and GAT-MAPPO.
Generates a 2-row x 2-col figure: Latency + Throughput for each method.
Matches DRL-OR Fig.5 style with vertical event markers.

Usage:
    python3 plot_link_degradation.py --log-dir ./log --save-dir ./figures
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
TYPE_COLORS  = {0: '#D32F2F', 1: '#1976D2', 2: '#388E3C', 3: '#F57C00'}
TYPE_LABELS  = {0: 'type1', 1: 'type2', 2: 'type3', 3: 'type4'}
NUM_TYPES    = 3          # change to 4 if you logged type4

# Degradation event timesteps (must match simenv.py change_env calls)
EVENTS = {
    10000:  ('60%',       'orange'),
    40000:  ('20%',       'red'),
    80000:  ('5%',        'darkred'),
    120000: ('Recovery',  'green'),
    150000: ('2nd 20%',   'darkorange'),
}

# Log subdirectory names — adjust to match your actual log paths
LOG_DIRS = {
    'PPO (DRL-OR)':      'ppo_link_degradation',
    'GAT-MAPPO (Ours)':  'mappo_link_degradation',
}
LINE_STYLES = {
    'PPO (DRL-OR)':     '--',
    'GAT-MAPPO (Ours)': '-',
}

# ── Utilities ─────────────────────────────────────────────────────────────────
def load_log(path):
    vals = []
    if not os.path.exists(path):
        return np.array([])
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                pass
    return np.array(vals)


def smooth(data, window=500):
    if len(data) < 2:
        return data
    w = min(window, max(1, len(data) // 5))
    return np.convolve(data, np.ones(w) / w, mode='valid')


def add_event_markers(ax, total_len, scale=1000):
    """Draw vertical lines for degradation events."""
    for ts, (label, color) in EVENTS.items():
        x = ts / scale
        if x <= total_len / scale:
            ax.axvline(x=x, color=color, linestyle=':', linewidth=1.2, alpha=0.8)
            ax.text(x + 0.5, ax.get_ylim()[1] * 0.97, label,
                    fontsize=6, color=color, rotation=90, va='top')


# ── Main plot ─────────────────────────────────────────────────────────────────
def plot_link_degradation(log_dir, save_dir, window=500):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GAT-MAPPO vs PPO — Link Degradation Scenario\n'
                 '(Bottleneck link 0-4 gradually degraded)',
                 fontsize=13, fontweight='bold')

    metric_configs = [
        # (metric_file, row, col_ppo, col_mappo, ylabel, title_suffix)
        ('delay',      0, 'Latency (ms)',       'Latency under Link Degradation'),
        ('throughput', 1, 'Throughput Ratio',   'Throughput Ratio under Link Degradation'),
    ]

    for metric, row, ylabel, title in metric_configs:
        for col, (method_name, log_subdir) in enumerate(LOG_DIRS.items()):
            ax = axes[row][col]

            any_data = False
            for t in range(NUM_TYPES):
                fpath = os.path.join(log_dir, log_subdir, f'{metric}_type{t}.log')
                raw = load_log(fpath)
                if len(raw) == 0:
                    continue

                # Convert throughput to ratio if needed (already ratio in your logs)
                smoothed = smooth(raw, window)
                x = np.arange(len(smoothed)) / 1000  # x-axis in 10^3 timeslots

                ax.plot(x, smoothed,
                        color=TYPE_COLORS[t],
                        linestyle=LINE_STYLES[method_name],
                        linewidth=1.4,
                        label=f'{TYPE_LABELS[t]}')
                any_data = True

            if any_data:
                add_event_markers(ax, len(smoothed), scale=1)

            ax.set_title(f'{method_name}\n{title}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Timeslot (×10³)', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if not any_data:
                ax.text(0.5, 0.5, f'No logs found:\n{log_subdir}',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=9, color='red',
                        bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Legend for event markers
    from matplotlib.lines import Line2D
    event_handles = [
        Line2D([0], [0], color=color, linestyle=':', linewidth=1.2, label=f't={ts//1000}k: {label}')
        for ts, (label, color) in EVENTS.items()
    ]
    fig.legend(handles=event_handles, loc='lower center', ncol=5,
               fontsize=8, title='Degradation Events', title_fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(save_dir, 'fig_link_degradation.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ── Single-panel version (DRL-OR Fig.5 style, side-by-side subplots) ─────────
def plot_link_degradation_fig5_style(log_dir, save_dir, window=500):
    """
    Closer to DRL-OR Fig.5 style: 2 panels (latency top, throughput bottom),
    both methods overlaid with different line styles.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    titles = ['(g) Latency under link degradation',
              '(h) Thrpt. ratio under link degradation']
    metrics = ['delay', 'throughput']
    ylabels = ['Latency (ms)', 'Throughput Ratio']

    for row, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[row]

        for method_name, log_subdir in LOG_DIRS.items():
            ls = LINE_STYLES[method_name]
            for t in range(NUM_TYPES):
                fpath = os.path.join(log_dir, log_subdir, f'{metric}_type{t}.log')
                raw = load_log(fpath)
                if len(raw) == 0:
                    continue
                smoothed = smooth(raw, window)
                x = np.arange(len(smoothed)) / 1000

                label = f'{method_name[:4]}-{TYPE_LABELS[t]}'
                ax.plot(x, smoothed, color=TYPE_COLORS[t],
                        linestyle=ls, linewidth=1.3, label=label, alpha=0.85)

        # Event markers
        ylim = ax.get_ylim()
        for ts, (lbl, color) in EVENTS.items():
            xv = ts / 1000
            ax.axvline(x=xv, color=color, linestyle=':', linewidth=1.0, alpha=0.75)
            ax.text(xv + 0.3, ylim[1], lbl, fontsize=6, color=color,
                    rotation=90, va='top')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Timeslot (×10³)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=6, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(save_dir, 'fig_link_degradation_fig5style.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir',  default='./log',     help='Root log directory')
    parser.add_argument('--save-dir', default='./figures', help='Output directory for figures')
    parser.add_argument('--window',   type=int, default=500, help='Smoothing window')
    parser.add_argument('--style',    choices=['grid', 'fig5', 'both'], default='both',
                        help='Which plot style to generate')
    args = parser.parse_args()

    print(f"Log dir:  {args.log_dir}")
    print(f"Save dir: {args.save_dir}")

    # Check log dirs exist
    for method, subdir in LOG_DIRS.items():
        full = os.path.join(args.log_dir, subdir)
        status = '✓' if os.path.isdir(full) else '✗ NOT FOUND'
        print(f"  [{status}] {method}: {full}")

    if args.style in ('grid', 'both'):
        plot_link_degradation(args.log_dir, args.save_dir, args.window)

    if args.style in ('fig5', 'both'):
        plot_link_degradation_fig5_style(args.log_dir, args.save_dir, args.window)

    print("Done.")


if __name__ == '__main__':
    main()