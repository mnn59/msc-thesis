"""
fallback_ratio.py — Calculate fallback policy trigger ratio
=============================================================
Reads circle.log (1 = loop detected / fallback triggered, 0 = safe route)
Produces Table II style output: trigger ratio at beginning vs after convergence.

Usage:
    python3 fallback_ratio.py --log-dir ./log/mappo_initialization
    python3 fallback_ratio.py --log-dir ./log/mappo_initialization --window 5000
    python3 fallback_ratio.py --all --log-base ./log    # all scenarios
"""

import os, argparse
import numpy as np


def read_circle_log(filepath):
    """Read circle.log: each line is 0 (safe) or 1 (loop/fallback)."""
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return None
    vals = []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    vals.append(int(float(s)))
                except ValueError:
                    pass
    return np.array(vals) if vals else None


def calc_trigger_ratio(data, start, end):
    """Calculate fallback trigger ratio (%) for a slice of data."""
    segment = data[start:end]
    if len(segment) == 0:
        return 0.0
    return 100.0 * np.sum(segment == 1) / len(segment)


def analyze_scenario(log_dir, scenario_name, window=5000):
    """Analyze one scenario's circle.log."""
    filepath = os.path.join(log_dir, 'circle.log')
    data = read_circle_log(filepath)
    if data is None:
        return None

    total_steps = len(data)
    total_loops = np.sum(data == 1)
    total_ratio = 100.0 * total_loops / total_steps

    # "At the beginning" = first `window` steps
    begin_ratio = calc_trigger_ratio(data, 0, window)

    # "After convergence" = last `window` steps
    converged_ratio = calc_trigger_ratio(data, max(0, total_steps - window), total_steps)

    # Sliding window trigger ratio over time (for plotting)
    ratios_over_time = []
    for i in range(0, total_steps - window + 1, window):
        r = calc_trigger_ratio(data, i, i + window)
        ratios_over_time.append((i, i + window, r))

    return {
        'scenario': scenario_name,
        'total_steps': total_steps,
        'total_loops': total_loops,
        'total_ratio': total_ratio,
        'begin_ratio': begin_ratio,
        'converged_ratio': converged_ratio,
        'over_time': ratios_over_time,
    }


def print_table2(results, window):
    """Print DRL-OR Table II style output."""
    print("\n" + "=" * 75)
    print("  Fallback Policy Trigger Ratio (%)")
    print(f"  Window size: {window:,} steps")
    print("=" * 75)
    print(f"  {'':24s} ", end="")
    for r in results:
        print(f"{r['scenario']:>18s}", end="")
    print()
    print("  " + "-" * (24 + 18 * len(results)))

    print(f"  {'At the beginning':24s} ", end="")
    for r in results:
        print(f"{r['begin_ratio']:17.2f}%", end="")
    print()

    print(f"  {'After convergence':24s} ", end="")
    for r in results:
        print(f"{r['converged_ratio']:17.2f}%", end="")
    print()

    print("  " + "-" * (24 + 18 * len(results)))
    print(f"  {'Overall':24s} ", end="")
    for r in results:
        print(f"{r['total_ratio']:17.2f}%", end="")
    print()
    print()

    # DRL-OR comparison
    print("  DRL-OR Paper (Table II) reference:")
    print("    Initialization:  Beginning=41.85%  Converged=2.30%")
    print("    Link failure:    Beginning=9.58%   Converged=2.39%")
    print("    Traffic change:  Beginning=3.18%   Converged=2.73%")
    print()


def print_detailed(result, window):
    """Print detailed breakdown over time for one scenario."""
    print(f"\n  [{result['scenario']}] Detailed breakdown ({window}-step windows):")
    print(f"  Total: {result['total_steps']:,} steps, {result['total_loops']:,} loops ({result['total_ratio']:.2f}%)")
    print(f"  {'Step range':>20s}  {'Trigger ratio':>15s}")
    print(f"  {'-'*20}  {'-'*15}")
    for start, end, ratio in result['over_time']:
        bar = '█' * int(ratio / 2) + '░' * (25 - int(ratio / 2))
        print(f"  {start:>8,} - {end:>8,}  {ratio:>13.2f}%  {bar}")


def plot_trigger_ratio(results, save_path, window):
    """Plot trigger ratio over time for all scenarios."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for i, r in enumerate(results):
        if r['over_time']:
            x = [(s + e) / 2 for s, e, _ in r['over_time']]
            y = [ratio for _, _, ratio in r['over_time']]
            ax.plot(x, y, color=colors[i % len(colors)], linewidth=1.5,
                    label=r['scenario'], alpha=0.9)

    ax.set_xlabel('Timeslot', fontsize=11)
    ax.set_ylabel('Fallback Trigger Ratio (%)', fontsize=11)
    ax.set_title(f'Fallback Policy Trigger Ratio Over Time (window={window:,})', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{x:.0f}'))

    # Add DRL-OR reference lines
    ax.axhline(y=41.85, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.annotate('DRL-OR init beginning (41.85%)', xy=(0.02, 0.95),
                xycoords='axes fraction', fontsize=7, color='gray')
    ax.axhline(y=2.30, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.annotate('DRL-OR converged (2.30%)', xy=(0.02, 0.05),
                xycoords='axes fraction', fontsize=7, color='gray')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved to: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Fallback policy trigger ratio (Table II)')
    parser.add_argument('--log-dir', default=None, help='Single scenario log dir')
    parser.add_argument('--log-base', default='./log', help='Base log dir for --all')
    parser.add_argument('--all', action='store_true', help='Analyze all scenarios')
    parser.add_argument('--window', type=int, default=5000,
                        help='Window size for beginning/convergence calculation')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plot')
    parser.add_argument('--save-path', default='./fallback_ratio.png')
    parser.add_argument('--detailed', action='store_true', help='Show step-by-step breakdown')
    args = parser.parse_args()

    results = []

    if args.all:
        scenarios = [
            ('mappo_initialization', 'MAPPO Init'),
            ('mappo_link_failure', 'MAPPO Link Fail'),
            ('mappo_traffic_change', 'MAPPO Traffic'),
            # PPO baselines for comparison
            ('ppo_initialization', 'PPO Init'),
            ('ppo_link_failure', 'PPO Link Fail'),
            ('ppo_traffic_change', 'PPO Traffic'),
        ]
        for dir_name, label in scenarios:
            log_dir = os.path.join(args.log_base, dir_name)
            if os.path.exists(os.path.join(log_dir, 'circle.log')):
                r = analyze_scenario(log_dir, label, args.window)
                if r:
                    results.append(r)
    elif args.log_dir:
        name = os.path.basename(args.log_dir.rstrip('/'))
        r = analyze_scenario(args.log_dir, name, args.window)
        if r:
            results.append(r)
    else:
        print("Specify --log-dir or --all")
        return

    if not results:
        print("No circle.log files found!")
        return

    print_table2(results, args.window)

    if args.detailed:
        for r in results:
            print_detailed(r, args.window)

    if args.plot and len(results) > 0:
        plot_trigger_ratio(results, args.save_path, args.window)


if __name__ == "__main__":
    main()