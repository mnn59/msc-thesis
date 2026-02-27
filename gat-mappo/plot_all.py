# #!/usr/bin/env python3
# """
# Comprehensive Plotting Suite for GAT-MAPPO vs PPO (DRL-OR)
# ===========================================================
# Generates all thesis figures from log files.

# Usage:
#     python3 plot_all.py --log-dir ./log --save-dir ./figures

# Expected log structure:
#     ./log/
#       ppo_initialization/       delay_type0.log, throughput_type0.log, loss_type0.log, ...
#       ppo_link_failure/         ...
#       ppo_traffic_change/       ...
#       mappo_initialization/     ...
#       mappo_link_failure/       ...
#       mappo_traffic_change/     ...
#       Abi_SHR_5000_lightload/   delay_type0.log, throughput_type0.log, loss_type0.log, ...
#       Abi_WP_5000_lightload/    ...
#       Abi_QoS_5000_lightload/   ...
# """

# import os
# import sys
# import argparse
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.patches import FancyBboxPatch
# from collections import defaultdict

# # ============================================================
# # Configuration
# # ============================================================
# TYPE_NAMES = {0: 'Type 1 (latency)', 1: 'Type 2 (throughput)',
#               2: 'Type 3 (lat-thrpt)', 3: 'Type 4 (lat-loss)'}
# TYPE_DEMANDS_KBPS = {0: 100, 1: 1500, 2: 1500, 3: 500}  # Kbps
# TYPE_COLORS = {0: '#D32F2F', 1: '#1976D2', 2: '#388E3C', 3: '#F57C00'}
# TYPE_SHORT = {0: 'type1', 1: 'type2', 2: 'type3', 3: 'type4'}

# # Abilene topology: average shortest path in hops (Floyd-Warshall, 11 nodes)
# # Used to compute stretch = 1 + delta_dist / avg_shr_dist
# ABILENE_AVG_SHR_DIST = 2.4182

# SCENARIOS = ['initialization', 'link_failure', 'traffic_change']
# SCENARIO_LABELS = {'initialization': 'Initialization',
#                    'link_failure': 'Link Failure',
#                    'traffic_change': 'Traffic Change'}

# # Methods for bar charts
# METHODS_ORDER = ['SPR', 'LBR', 'QoSR', 'PPO (DRL-OR)', 'GAT-MAPPO (Ours)']
# METHOD_LOG_MAP = {
#     'SPR':               'Abi_SHR_5000_lightload',
#     'LBR':               'Abi_WP_5000_lightload',
#     'QoSR':              'Abi_QoS_5000_lightload',
#     'PPO (DRL-OR)':      'ppo_initialization',
#     'GAT-MAPPO (Ours)':  'mappo_initialization',
# }
# # Colors for bar chart methods
# METHOD_COLORS = {
#     'SPR':               '#5B9BD5',
#     'LBR':               '#ED7D31',
#     'QoSR':              '#A5A5A5',
#     'PPO (DRL-OR)':      '#FFC000',
#     'GAT-MAPPO (Ours)':  '#70AD47',
# }
# METHOD_HATCHES = {
#     'SPR':               '///',
#     'LBR':               '\\\\\\',
#     'QoSR':              'xxx',
#     'PPO (DRL-OR)':      '...',
#     'GAT-MAPPO (Ours)':  '',
# }

# NUM_TYPES = 3  # For bar chart and fig5 (types 0,1,2). Change to 4 if type4 desired.

# # ============================================================
# # Utilities
# # ============================================================
# def load_log(filepath):
#     """Load a log file as numpy array, skip bad lines."""
#     vals = []
#     if not os.path.exists(filepath):
#         return np.array([])
#     with open(filepath, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 vals.append(float(line))
#             except ValueError:
#                 continue
#     return np.array(vals)


# def smooth(data, window=500):
#     """Moving average smoothing."""
#     if len(data) < window:
#         window = max(1, len(data) // 5)
#     if window < 2:
#         return data
#     kernel = np.ones(window) / window
#     return np.convolve(data, kernel, mode='valid')


# def truncate_to_min(*arrays):
#     """Truncate all arrays to the minimum length."""
#     min_len = min(len(a) for a in arrays if len(a) > 0)
#     return [a[:min_len] for a in arrays]


# def get_converged_mean(data, last_n=2000):
#     """Get mean of last N samples (converged region)."""
#     if len(data) == 0:
#         return 0.0
#     n = min(last_n, len(data))
#     return np.mean(data[-n:])


# def compute_stretch(delta_dist_array, avg_shr_dist=ABILENE_AVG_SHR_DIST):
#     """
#     Compute per-flow stretch from delta_dist log.
#     stretch_i = 1 + delta_dist_i / avg_shr_dist
    
#     delta_dist = (selected path hops) - (shortest path hops)
#       = 0 means shortest path was taken (stretch=1.0, ideal)
#       > 0 means path is longer than shortest (stretch>1.0)
#       < 0 can happen with fallback/loop detection (clamp to 0)
#     """
#     dd = np.array(delta_dist_array, dtype=float)
#     dd = np.maximum(dd, 0)  # clamp negatives
#     stretch = 1.0 + dd / avg_shr_dist
#     return stretch


# # ============================================================
# # Figure 1: Split-Panel Fig5 (PPO vs MAPPO, 3 metrics × 3 scenarios)
# # ============================================================
# def plot_fig5_split(log_dir, save_dir, window=500):
#     """
#     3 rows (Latency, Throughput, Packet Loss) × 3 cols (Init, Link Fail, Traffic Change).
#     Each cell: top=PPO, bottom=MAPPO.
#     Truncates x-axis to min length across all types within each method+scenario.
#     """
#     fig = plt.figure(figsize=(16, 22))
#     fig.suptitle('PPO (DRL-OR) vs GAT-MAPPO (Ours) — Abilene Light Load',
#                  fontsize=16, fontweight='bold', y=0.98)

#     outer = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.28,
#                               top=0.93, bottom=0.05, left=0.07, right=0.95)

#     metrics = ['delay', 'throughput', 'loss', 'stretch']
#     metric_labels = ['Latency (ms)', 'Throughput Ratio', 'Packet Loss Rate', 'Stretch']
#     panel_letters = [['(a)', '(b)', '(c)'],
#                      ['(d)', '(e)', '(f)'],
#                      ['(g)', '(h)', '(i)'],
#                      ['(j)', '(k)', '(l)']]

#     methods = [('ppo', 'PPO (DRL-OR)'), ('mappo', 'GAT-MAPPO (Ours)')]
#     method_bg = {'ppo': '#FFF8E1', 'mappo': '#E3F2FD'}
#     method_label_bg = {'ppo': '#FFC107', 'mappo': '#42A5F5'}

#     for row, metric in enumerate(metrics):
#         for col, scenario in enumerate(SCENARIOS):
#             inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col],
#                                                      hspace=0.08)
#             for mi, (method_key, method_name) in enumerate(methods):
#                 ax = fig.add_subplot(inner[mi])

#                 log_path = os.path.join(log_dir, f'{method_key}_{scenario}')

#                 # Load all types and find min length
#                 all_data = {}
#                 for t in range(NUM_TYPES):
#                     # stretch is computed from dist (delta_dist) logs
#                     if metric == 'stretch':
#                         fname = f'dist_type{t}.log'
#                     else:
#                         fname = f'{metric}_type{t}.log'
#                     d = load_log(os.path.join(log_path, fname))
#                     if len(d) > 0:
#                         if metric == 'stretch':
#                             d = compute_stretch(d)
#                         all_data[t] = d

#                 if not all_data:
#                     ax.text(0.5, 0.5, 'No data', ha='center', va='center',
#                             transform=ax.transAxes, fontsize=8, color='gray')
#                     continue

#                 # Truncate to min length across types
#                 min_len = min(len(v) for v in all_data.values())
#                 for t in all_data:
#                     all_data[t] = all_data[t][:min_len]

#                 # Plot each type
#                 for t in sorted(all_data.keys()):
#                     s = smooth(all_data[t], window)
#                     x = np.arange(len(s))
#                     x_scaled = x / 1000.0  # ×10³
#                     ax.plot(x_scaled, s, color=TYPE_COLORS[t],
#                             linewidth=0.8, label=TYPE_SHORT[t], alpha=0.9)

#                 ax.set_facecolor(method_bg[method_key])

#                 # Method label badge
#                 bbox_color = method_label_bg[method_key]
#                 ax.text(0.02, 0.92, method_name, transform=ax.transAxes,
#                         fontsize=6.5, fontweight='bold', va='top',
#                         bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color,
#                                   alpha=0.8, edgecolor='none'),
#                         color='white' if method_key == 'mappo' else 'black')

#                 # Legend (top-right, small)
#                 ax.legend(fontsize=5.5, loc='upper right', ncol=NUM_TYPES,
#                           framealpha=0.7, handlelength=1.2, columnspacing=0.5)

#                 # Axis labels
#                 if mi == 1:  # bottom sub-panel
#                     ax.set_xlabel(f'Timeslot (×10³)', fontsize=7)
#                 else:
#                     ax.set_xticklabels([])

#                 ax.tick_params(labelsize=6)
#                 ax.grid(True, alpha=0.2)

#             # Panel title
#             fig.text(outer[row, col].get_position(fig).x0 +
#                      outer[row, col].get_position(fig).width / 2,
#                      outer[row, col].get_position(fig).y1 + 0.005,
#                      f'{panel_letters[row][col]} {metric_labels[row]} — {SCENARIO_LABELS[scenario]}',
#                      ha='center', fontsize=8.5, fontweight='bold')

#     # Bottom legend
#     from matplotlib.lines import Line2D
#     legend_elements = [Line2D([0], [0], color=TYPE_COLORS[t], lw=2, label=TYPE_NAMES[t])
#                        for t in range(NUM_TYPES)]
#     legend_elements.append(Line2D([0], [0], color='none', label=''))
#     legend_elements.append(Line2D([0], [0], color='gray', lw=6, alpha=0.3,
#                                   label='Top = PPO (DRL-OR)'))
#     legend_elements.append(Line2D([0], [0], color='#90CAF9', lw=6, alpha=0.5,
#                                   label='Bottom = GAT-MAPPO (Ours)'))
#     fig.legend(handles=legend_elements, loc='lower center', ncol=6,
#                fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.005))

#     path = os.path.join(save_dir, 'fig5_split.png')
#     fig.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved: {path}")


# # ============================================================
# # Figure 2: Reward comparison (PPO vs MAPPO, 1 row × 3 scenarios)
# # ============================================================
# def plot_reward(log_dir, save_dir, window=500):
#     """Global reward: PPO vs MAPPO overlay for each scenario."""
#     fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
#     fig.suptitle('Global Reward — PPO (DRL-OR) vs GAT-MAPPO (Ours)',
#                  fontsize=14, fontweight='bold')

#     colors = {'ppo': '#1976D2', 'mappo': '#D32F2F'}
#     labels = {'ppo': 'PPO (DRL-OR)', 'mappo': 'GAT-MAPPO (Ours)'}

#     for i, scenario in enumerate(SCENARIOS):
#         ax = axes[i]
#         for method in ['ppo', 'mappo']:
#             fpath = os.path.join(log_dir, f'{method}_{scenario}', 'globalrwd.log')
#             data = load_log(fpath)
#             if len(data) == 0:
#                 continue
#             s = smooth(data, window)
#             x = np.arange(len(s)) / 1000.0
#             ax.plot(x, s, color=colors[method], linewidth=1.2,
#                     label=labels[method], alpha=0.9)

#         ax.set_title(f'Reward — {SCENARIO_LABELS[scenario]}', fontsize=11)
#         ax.set_xlabel('Timeslot (×10³)', fontsize=9)
#         ax.set_ylabel('Global Reward', fontsize=9)
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)
#         ax.tick_params(labelsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     path = os.path.join(save_dir, 'fig5_reward.png')
#     fig.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved: {path}")


# # ============================================================
# # Figure 3: Bar Charts (SPR, LBR, QoSR, PPO, MAPPO)
# # Like the uploaded paper style — grouped bars per type.
# # Uses CONVERGED values (last 2000 steps).
# # ============================================================
# def plot_bar_charts(log_dir, save_dir):
#     """
#     3 bar charts: Latency, Throughput Ratio, Avg Packet Loss Ratio.
#     Each has groups per flow type, bars per method.
#     """
#     metrics = {
#         'delay':      ('Average Latency (ms)', 'Latency Comparison — Abilene Light Load'),
#         'throughput':  ('Throughput Ratio', 'Throughput Ratio Comparison — Abilene Light Load'),
#         'loss':        ('Average Packet Loss Ratio (%)', 'Avg. Packet Loss Ratio Comparison — Abilene Light Load'),
#         'stretch':     ('Average Stretch', 'Average Stretch Comparison — Abilene Light Load'),
#     }

#     for metric, (ylabel, title) in metrics.items():
#         fig, ax = plt.subplots(figsize=(8, 5))

#         # Collect converged means
#         data = {}  # data[method][type] = mean
#         for method_name, log_subdir in METHOD_LOG_MAP.items():
#             data[method_name] = {}
#             for t in range(NUM_TYPES):
#                 if metric == 'stretch':
#                     fpath = os.path.join(log_dir, log_subdir, f'dist_type{t}.log')
#                     vals = load_log(fpath)
#                     if len(vals) > 0:
#                         vals = compute_stretch(vals)
#                     mean_val = get_converged_mean(vals, last_n=2000)
#                 else:
#                     fpath = os.path.join(log_dir, log_subdir, f'{metric}_type{t}.log')
#                     vals = load_log(fpath)
#                     mean_val = get_converged_mean(vals, last_n=2000)
#                 if metric == 'loss':
#                     mean_val *= 100  # Convert to percentage
#                 data[method_name][t] = mean_val

#         # Plot grouped bars
#         n_types = NUM_TYPES
#         n_methods = len(METHODS_ORDER)
#         bar_width = 0.14
#         x = np.arange(n_types)

#         for j, method_name in enumerate(METHODS_ORDER):
#             offsets = x + (j - n_methods / 2 + 0.5) * bar_width
#             values = [data[method_name].get(t, 0) for t in range(n_types)]
#             bars = ax.bar(offsets, values, bar_width,
#                           label=method_name,
#                           color=METHOD_COLORS[method_name],
#                           hatch=METHOD_HATCHES[method_name],
#                           edgecolor='black', linewidth=0.5, alpha=0.85)

#         ax.set_xlabel('Flow Type', fontsize=11)
#         ax.set_ylabel(ylabel, fontsize=11)
#         ax.set_title(title, fontsize=12, fontweight='bold')
#         ax.set_xticks(x)
#         ax.set_xticklabels([f'Type {t+1}' for t in range(n_types)], fontsize=10)
#         ax.legend(fontsize=8, loc='upper right')
#         ax.grid(True, axis='y', alpha=0.3)
#         ax.tick_params(labelsize=9)

#         path = os.path.join(save_dir, f'bar_{metric}.png')
#         fig.savefig(path, dpi=200, bbox_inches='tight')
#         plt.close(fig)
#         print(f"Saved: {path}")


# # ============================================================
# # Figure 4: TensorBoard-style time series (individual metric plots)
# # Each metric+scenario as a separate figure, PPO vs MAPPO overlay per type.
# # Dark background, smooth lines, like TensorBoard.
# # ============================================================
# def plot_tensorboard_style(log_dir, save_dir, window=500):
#     """
#     Individual plots for each (metric, scenario, type) — TensorBoard look.
#     Also combined per (metric, scenario) with all types.
#     """
#     tb_bg = '#1E1E2E'
#     tb_grid = '#333355'
#     tb_text = '#E0E0E0'
#     tb_colors_ppo = {0: '#FF6B6B', 1: '#6BC5FF', 2: '#6BFF8A', 3: '#FFD76B'}
#     tb_colors_mappo = {0: '#FF3333', 1: '#3399FF', 2: '#33CC55', 3: '#FFAA00'}

#     metrics = ['delay', 'throughput', 'loss', 'stretch']
#     metric_labels = {'delay': 'Latency (ms)', 'throughput': 'Throughput Ratio',
#                      'loss': 'Packet Loss Rate', 'stretch': 'Stretch'}

#     tb_dir = os.path.join(save_dir, 'tensorboard_style')
#     os.makedirs(tb_dir, exist_ok=True)

#     for metric in metrics:
#         for scenario in SCENARIOS:
#             # Combined plot (all types, PPO vs MAPPO)
#             fig, ax = plt.subplots(figsize=(10, 5), facecolor=tb_bg)
#             ax.set_facecolor(tb_bg)

#             for method, mcolors, ls in [('ppo', tb_colors_ppo, '--'),
#                                          ('mappo', tb_colors_mappo, '-')]:
#                 log_path = os.path.join(log_dir, f'{method}_{scenario}')
#                 for t in range(NUM_TYPES):
#                     if metric == 'stretch':
#                         data = load_log(os.path.join(log_path, f'dist_type{t}.log'))
#                         if len(data) > 0:
#                             data = compute_stretch(data)
#                     else:
#                         data = load_log(os.path.join(log_path, f'{metric}_type{t}.log'))
#                     if len(data) == 0:
#                         continue
#                     s = smooth(data, window)
#                     x = np.arange(len(s)) / 1000.0
#                     method_label = 'PPO' if method == 'ppo' else 'MAPPO'
#                     ax.plot(x, s, color=mcolors[t], linewidth=1.0, linestyle=ls,
#                             label=f'{method_label} {TYPE_SHORT[t]}', alpha=0.85)

#             ax.set_xlabel('Timeslot (×10³)', fontsize=10, color=tb_text)
#             ax.set_ylabel(metric_labels[metric], fontsize=10, color=tb_text)
#             ax.set_title(f'{metric_labels[metric]} — {SCENARIO_LABELS[scenario]}',
#                          fontsize=12, fontweight='bold', color=tb_text)
#             ax.tick_params(colors=tb_text, labelsize=8)
#             ax.grid(True, color=tb_grid, alpha=0.5)
#             ax.legend(fontsize=7, facecolor='#2A2A44', edgecolor='#444466',
#                       labelcolor=tb_text, loc='upper right', ncol=2)
#             for spine in ax.spines.values():
#                 spine.set_color(tb_grid)

#             path = os.path.join(tb_dir, f'tb_{metric}_{scenario}.png')
#             fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=tb_bg)
#             plt.close(fig)
#             print(f"Saved: {path}")


# # ============================================================
# # Figure 5: Individual separated plots (one per metric × scenario × method)
# # ============================================================
# def plot_individual(log_dir, save_dir, window=500):
#     """One plot per (method, scenario, metric) with all types."""
#     ind_dir = os.path.join(save_dir, 'individual')
#     os.makedirs(ind_dir, exist_ok=True)

#     metrics = ['delay', 'throughput', 'loss', 'stretch', 'globalrwd']
#     metric_labels = {'delay': 'Latency (ms)', 'throughput': 'Throughput Ratio',
#                      'loss': 'Packet Loss Rate', 'stretch': 'Stretch',
#                      'globalrwd': 'Global Reward'}

#     for method in ['ppo', 'mappo']:
#         method_label = 'PPO (DRL-OR)' if method == 'ppo' else 'GAT-MAPPO (Ours)'
#         for scenario in SCENARIOS:
#             log_path = os.path.join(log_dir, f'{method}_{scenario}')
#             for metric in metrics:
#                 fig, ax = plt.subplots(figsize=(8, 4))

#                 if metric == 'globalrwd':
#                     data = load_log(os.path.join(log_path, 'globalrwd.log'))
#                     if len(data) > 0:
#                         s = smooth(data, window)
#                         x = np.arange(len(s)) / 1000.0
#                         ax.plot(x, s, color='#1976D2', linewidth=1.0)
#                 else:
#                     all_data = {}
#                     for t in range(NUM_TYPES):
#                         if metric == 'stretch':
#                             d = load_log(os.path.join(log_path, f'dist_type{t}.log'))
#                             if len(d) > 0:
#                                 d = compute_stretch(d)
#                         else:
#                             d = load_log(os.path.join(log_path, f'{metric}_type{t}.log'))
#                         if len(d) > 0:
#                             all_data[t] = d
#                     if all_data:
#                         min_len = min(len(v) for v in all_data.values())
#                         for t in sorted(all_data.keys()):
#                             s = smooth(all_data[t][:min_len], window)
#                             x = np.arange(len(s)) / 1000.0
#                             ax.plot(x, s, color=TYPE_COLORS[t], linewidth=1.0,
#                                     label=TYPE_SHORT[t])
#                         ax.legend(fontsize=8)

#                 ax.set_title(f'{method_label} — {metric_labels[metric]} — {SCENARIO_LABELS[scenario]}',
#                              fontsize=11, fontweight='bold')
#                 ax.set_xlabel('Timeslot (×10³)', fontsize=9)
#                 ax.set_ylabel(metric_labels[metric], fontsize=9)
#                 ax.grid(True, alpha=0.3)
#                 ax.tick_params(labelsize=8)

#                 path = os.path.join(ind_dir, f'{method}_{scenario}_{metric}.png')
#                 fig.savefig(path, dpi=150, bbox_inches='tight')
#                 plt.close(fig)
#                 print(f"Saved: {path}")


# # ============================================================
# # Figure 6: Throughput in Mbps (actual bandwidth, PPO vs MAPPO)
# # ============================================================
# def plot_throughput_mbps(log_dir, save_dir, window=500):
#     """
#     Actual throughput in Mbps = throughput_ratio × demand_Kbps / 1000.
#     PPO vs MAPPO overlay per type for each scenario.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     fig.suptitle('Actual Throughput (Mbps) — PPO vs GAT-MAPPO',
#                  fontsize=14, fontweight='bold')

#     for i, scenario in enumerate(SCENARIOS):
#         ax = axes[i]
#         for method, ls, alpha_val in [('ppo', '--', 0.6), ('mappo', '-', 0.9)]:
#             log_path = os.path.join(log_dir, f'{method}_{scenario}')
#             method_label = 'PPO' if method == 'ppo' else 'MAPPO'

#             all_data = {}
#             for t in range(NUM_TYPES):
#                 d = load_log(os.path.join(log_path, f'throughput_type{t}.log'))
#                 if len(d) > 0:
#                     all_data[t] = d

#             if not all_data:
#                 continue
#             min_len = min(len(v) for v in all_data.values())

#             for t in sorted(all_data.keys()):
#                 ratio = all_data[t][:min_len]
#                 mbps = ratio * TYPE_DEMANDS_KBPS[t] / 1000.0  # Kbps → Mbps
#                 s = smooth(mbps, window)
#                 x = np.arange(len(s)) / 1000.0
#                 ax.plot(x, s, color=TYPE_COLORS[t], linewidth=1.0,
#                         linestyle=ls, alpha=alpha_val,
#                         label=f'{method_label} {TYPE_SHORT[t]}')

#         ax.set_title(f'Throughput — {SCENARIO_LABELS[scenario]}', fontsize=11)
#         ax.set_xlabel('Timeslot (×10³)', fontsize=9)
#         ax.set_ylabel('Throughput (Mbps)', fontsize=9)
#         ax.legend(fontsize=6, ncol=2)
#         ax.grid(True, alpha=0.3)
#         ax.tick_params(labelsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     path = os.path.join(save_dir, 'throughput_mbps.png')
#     fig.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved: {path}")

#     # Also bar chart of converged Mbps
#     fig, ax = plt.subplots(figsize=(8, 5))
#     methods_bar = ['PPO (DRL-OR)', 'GAT-MAPPO (Ours)']
#     method_dirs = {'PPO (DRL-OR)': 'ppo_initialization',
#                    'GAT-MAPPO (Ours)': 'mappo_initialization'}
#     bar_colors = {'PPO (DRL-OR)': '#1976D2', 'GAT-MAPPO (Ours)': '#D32F2F'}

#     x = np.arange(NUM_TYPES)
#     bar_width = 0.3
#     for j, method_name in enumerate(methods_bar):
#         vals = []
#         for t in range(NUM_TYPES):
#             fpath = os.path.join(log_dir, method_dirs[method_name],
#                                  f'throughput_type{t}.log')
#             d = load_log(fpath)
#             mean_ratio = get_converged_mean(d)
#             mbps = mean_ratio * TYPE_DEMANDS_KBPS[t] / 1000.0
#             vals.append(mbps)
#         ax.bar(x + (j - 0.5) * bar_width, vals, bar_width,
#                label=method_name, color=bar_colors[method_name],
#                edgecolor='black', linewidth=0.5, alpha=0.85)

#     ax.set_xlabel('Flow Type', fontsize=11)
#     ax.set_ylabel('Throughput (Mbps)', fontsize=11)
#     ax.set_title('Converged Throughput (Mbps) — PPO vs GAT-MAPPO', fontsize=12, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels([f'Type {t+1}\n({TYPE_DEMANDS_KBPS[t]} Kbps)' for t in range(NUM_TYPES)],
#                        fontsize=9)
#     ax.legend(fontsize=9)
#     ax.grid(True, axis='y', alpha=0.3)

#     path = os.path.join(save_dir, 'bar_throughput_mbps.png')
#     fig.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved: {path}")


# # ============================================================
# # Figure 7: Avg Packet Loss bar chart (like the uploaded paper Fig.7)
# # Replaces the bottom row of fig5 with a bar chart style comparison.
# # ============================================================
# def plot_avg_loss_bar(log_dir, save_dir):
#     """
#     Bar chart comparing avg packet loss ratio across ALL methods
#     (SPR, LBR, QoSR, PPO, MAPPO) — single bar chart like the uploaded paper.
#     """
#     fig, ax = plt.subplots(figsize=(9, 5.5))

#     # For each method, compute average loss across all types
#     # and also per-type
#     n_types = NUM_TYPES
#     n_methods = len(METHODS_ORDER)
#     bar_width = 0.14
#     x = np.arange(n_types)

#     for j, method_name in enumerate(METHODS_ORDER):
#         log_subdir = METHOD_LOG_MAP[method_name]
#         values = []
#         for t in range(n_types):
#             fpath = os.path.join(log_dir, log_subdir, f'loss_type{t}.log')
#             vals = load_log(fpath)
#             mean_val = get_converged_mean(vals, last_n=2000) * 100  # to %
#             values.append(mean_val)
#         offsets = x + (j - n_methods / 2 + 0.5) * bar_width
#         ax.bar(offsets, values, bar_width,
#                label=method_name, color=METHOD_COLORS[method_name],
#                hatch=METHOD_HATCHES[method_name],
#                edgecolor='black', linewidth=0.5, alpha=0.85)

#     ax.set_xlabel('Flow Type', fontsize=11)
#     ax.set_ylabel('Average Packet Loss Ratio (%)', fontsize=11)
#     ax.set_title('Average Packet Loss Ratio Comparison — Abilene Light Load',
#                  fontsize=12, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels([f'Type {t+1}' for t in range(n_types)], fontsize=10)
#     ax.legend(fontsize=8)
#     ax.grid(True, axis='y', alpha=0.3)

#     path = os.path.join(save_dir, 'bar_avg_loss.png')
#     fig.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved: {path}")


# # ============================================================
# # Figure 8: Fallback Policy Trigger Ratio (from circle.log)
# # ============================================================
# def plot_fallback_ratio(log_dir, save_dir, fb_window=5000):
#     """
#     Fallback policy trigger ratio over time.
#     Reads circle.log (1=loop triggered, 0=safe).
#     Rolling window percentage.
#     """
#     fig, ax = plt.subplots(figsize=(12, 5))

#     combos = [
#         ('mappo_initialization',  'MAPPO Init',        '#D32F2F', '-'),
#         ('mappo_link_failure',    'MAPPO Link Fail',   '#42A5F5', '-'),
#         ('mappo_traffic_change',  'MAPPO Traffic',     '#388E3C', '-'),
#         ('ppo_initialization',    'PPO Init',          '#FFA726', '--'),
#         ('ppo_link_failure',      'PPO Link Fail',     '#7E57C2', '--'),
#         ('ppo_traffic_change',    'PPO Traffic',       '#EC407A', '--'),
#     ]

#     for subdir, label, color, ls in combos:
#         fpath = os.path.join(log_dir, subdir, 'circle.log')
#         data = load_log(fpath)
#         if len(data) == 0:
#             continue
#         # Convert to binary (some logs have float values)
#         binary = (data > 0.5).astype(float)
#         # Rolling mean
#         if len(binary) < fb_window:
#             continue
#         kernel = np.ones(fb_window) / fb_window
#         ratio = np.convolve(binary, kernel, mode='valid') * 100  # percentage
#         x = np.arange(len(ratio))
#         ax.plot(x, ratio, color=color, linewidth=1.2, linestyle=ls,
#                 label=label, alpha=0.85)

#     # DRL-OR reference lines
#     ax.axhline(y=41.85, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
#     ax.text(0, 42.5, 'DRL-OR init beginning (41.85%)', fontsize=7, color='gray')
#     ax.axhline(y=2.30, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
#     ax.text(0, 3.0, 'DRL-OR converged (2.30%)', fontsize=7, color='gray')

#     ax.set_xlabel('Timeslot', fontsize=10)
#     ax.set_ylabel('Fallback Trigger Ratio (%)', fontsize=10)
#     ax.set_title(f'Fallback Policy Trigger Ratio Over Time (window={fb_window:,})',
#                  fontsize=12, fontweight='bold')
#     ax.legend(fontsize=8, loc='upper right')
#     ax.grid(True, alpha=0.3)
#     ax.set_ylim(bottom=-1)

#     # Format x axis with k
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))

#     path = os.path.join(save_dir, 'fallback_ratio.png')
#     fig.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved: {path}")


# # ============================================================
# # Table III style: Print converged values
# # ============================================================
# def print_table3(log_dir):
#     """Print Table III comparison values to console."""
#     print("\n" + "="*80)
#     print("TABLE III — Converged Performance (last 2000 steps)")
#     print("="*80)

#     for metric, unit in [('delay', 'ms'), ('throughput', 'ratio'), ('loss', '%'), ('stretch', 'ratio')]:
#         print(f"\n--- {metric.upper()} ({unit}) ---")
#         header = f"{'Method':<20}"
#         for t in range(NUM_TYPES):
#             header += f"{'Type'+str(t+1):>12}"
#         print(header)
#         print("-" * (20 + 12 * NUM_TYPES))

#         for method_name, log_subdir in METHOD_LOG_MAP.items():
#             row = f"{method_name:<20}"
#             for t in range(NUM_TYPES):
#                 if metric == 'stretch':
#                     fpath = os.path.join(log_dir, log_subdir, f'dist_type{t}.log')
#                     vals = load_log(fpath)
#                     if len(vals) > 0:
#                         vals = compute_stretch(vals)
#                     mean_val = get_converged_mean(vals)
#                 else:
#                     fpath = os.path.join(log_dir, log_subdir, f'{metric}_type{t}.log')
#                     vals = load_log(fpath)
#                     mean_val = get_converged_mean(vals)
#                 if metric == 'loss':
#                     mean_val *= 100
#                 row += f"{mean_val:>12.4f}"
#             print(row)


# # ============================================================
# # Main
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser(description='Plot all thesis figures')
#     parser.add_argument('--log-dir', default='./log', help='Log directory')
#     parser.add_argument('--save-dir', default='./figures', help='Output directory')
#     parser.add_argument('--window', type=int, default=500, help='Smoothing window')
#     parser.add_argument('--fb-window', type=int, default=5000, help='Fallback ratio window')
#     parser.add_argument('--num-types', type=int, default=3,
#                         help='Number of flow types to plot (3 or 4)')
#     args = parser.parse_args()

#     global NUM_TYPES
#     NUM_TYPES = args.num_types

#     os.makedirs(args.save_dir, exist_ok=True)

#     print("="*60)
#     print("Generating all thesis figures...")
#     print(f"  Log dir:  {args.log_dir}")
#     print(f"  Save dir: {args.save_dir}")
#     print(f"  Smoothing window: {args.window}")
#     print(f"  Num types: {NUM_TYPES}")
#     print("="*60)

#     # 1. Fig5 split panels
#     print("\n[1/8] Fig5 split panels (PPO vs MAPPO)...")
#     plot_fig5_split(args.log_dir, args.save_dir, args.window)

#     # 2. Reward comparison
#     print("\n[2/8] Reward comparison...")
#     plot_reward(args.log_dir, args.save_dir, args.window)

#     # 3. Bar charts (all 5 methods)
#     print("\n[3/8] Bar charts (SPR, LBR, QoSR, PPO, MAPPO) — latency, throughput, loss, stretch...")
#     plot_bar_charts(args.log_dir, args.save_dir)

#     # 4. TensorBoard-style plots
#     print("\n[4/8] TensorBoard-style plots...")
#     plot_tensorboard_style(args.log_dir, args.save_dir, args.window)

#     # 5. Individual plots
#     print("\n[5/8] Individual plots...")
#     plot_individual(args.log_dir, args.save_dir, args.window)

#     # 6. Throughput Mbps
#     print("\n[6/8] Throughput Mbps plots...")
#     plot_throughput_mbps(args.log_dir, args.save_dir, args.window)

#     # 7. Avg packet loss bar (paper-style)
#     print("\n[7/8] Avg packet loss bar chart...")
#     plot_avg_loss_bar(args.log_dir, args.save_dir)

#     # 8. Fallback ratio
#     print("\n[8/8] Fallback policy trigger ratio...")
#     plot_fallback_ratio(args.log_dir, args.save_dir, args.fb_window)

#     # Print table
#     print_table3(args.log_dir)

#     print(f"\n{'='*60}")
#     print(f"All figures saved to: {args.save_dir}/")
#     print(f"{'='*60}")


# if __name__ == '__main__':
#     main()




#!/usr/bin/env python3
"""
Comprehensive Plotting Suite for GAT-MAPPO vs PPO (DRL-OR)
===========================================================
Generates all thesis figures from log files.

Usage:
    python3 plot_all.py --log-dir ./log --save-dir ./figures

Expected log structure:
    ./log/
      ppo_initialization/       delay_type0.log, throughput_type0.log, loss_type0.log, ...
      ppo_link_failure/         ...
      ppo_traffic_change/       ...
      mappo_initialization/     ...
      mappo_link_failure/       ...
      mappo_traffic_change/     ...
      Abi_SHR_5000_lightload/   delay_type0.log, throughput_type0.log, loss_type0.log, ...
      Abi_WP_5000_lightload/    ...
      Abi_QoS_5000_lightload/   ...
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
TYPE_NAMES = {0: 'Type 1 (latency)', 1: 'Type 2 (throughput)',
              2: 'Type 3 (lat-thrpt)', 3: 'Type 4 (lat-loss)'}
TYPE_DEMANDS_KBPS = {0: 100, 1: 1500, 2: 1500, 3: 500}  # Kbps
TYPE_COLORS = {0: '#D32F2F', 1: '#1976D2', 2: '#388E3C', 3: '#F57C00'}
TYPE_SHORT = {0: 'type1', 1: 'type2', 2: 'type3', 3: 'type4'}

# Abilene topology: average shortest path in hops (Floyd-Warshall, 11 nodes)
# Used to compute stretch = 1 + delta_dist / avg_shr_dist
ABILENE_AVG_SHR_DIST = 2.4182

SCENARIOS = ['initialization', 'link_failure', 'traffic_change']
SCENARIO_LABELS = {'initialization': 'Initialization',
                   'link_failure': 'Link Failure',
                   'traffic_change': 'Traffic Change'}

# Methods for bar charts
METHODS_ORDER = ['SPR', 'LBR', 'QoSR', 'PPO (DRL-OR)', 'GAT-MAPPO (Ours)']
METHOD_LOG_MAP = {
    'SPR':               'Abi_SHR_5000_lightload',
    'LBR':               'Abi_WP_5000_lightload',
    'QoSR':              'Abi_QoS_5000_lightload',
    'PPO (DRL-OR)':      'ppo_initialization',
    'GAT-MAPPO (Ours)':  'mappo_initialization',
}
# Colors for bar chart methods
METHOD_COLORS = {
    'SPR':               '#5B9BD5',
    'LBR':               '#ED7D31',
    'QoSR':              '#A5A5A5',
    'PPO (DRL-OR)':      '#FFC000',
    'GAT-MAPPO (Ours)':  '#70AD47',
}
METHOD_HATCHES = {
    'SPR':               '///',
    'LBR':               '\\\\\\',
    'QoSR':              'xxx',
    'PPO (DRL-OR)':      '...',
    'GAT-MAPPO (Ours)':  '',
}

NUM_TYPES = 3  # For bar chart and fig5 (types 0,1,2). Change to 4 if type4 desired.

# ============================================================
# Utilities
# ============================================================
def load_log(filepath):
    """Load a log file as numpy array, skip bad lines."""
    vals = []
    if not os.path.exists(filepath):
        return np.array([])
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
    return np.array(vals)


def smooth(data, window=500):
    """Moving average smoothing."""
    if len(data) < window:
        window = max(1, len(data) // 5)
    if window < 2:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def truncate_to_min(*arrays):
    """Truncate all arrays to the minimum length."""
    min_len = min(len(a) for a in arrays if len(a) > 0)
    return [a[:min_len] for a in arrays]


def get_converged_mean(data, last_n=2000):
    """Get mean of last N samples (converged region)."""
    if len(data) == 0:
        return 0.0
    n = min(last_n, len(data))
    return np.mean(data[-n:])


def compute_stretch(delta_dist_array, avg_shr_dist=ABILENE_AVG_SHR_DIST):
    """
    Compute per-flow stretch from delta_dist log.
    stretch_i = 1 + delta_dist_i / avg_shr_dist
    
    delta_dist = (selected path hops) - (shortest path hops)
      = 0 means shortest path was taken (stretch=1.0, ideal)
      > 0 means path is longer than shortest (stretch>1.0)
      < 0 can happen with fallback/loop detection (clamp to 0)
    """
    dd = np.array(delta_dist_array, dtype=float)
    dd = np.maximum(dd, 0)  # clamp negatives
    stretch = 1.0 + dd / avg_shr_dist
    return stretch


# ============================================================
# Figure 1: Split-Panel Fig5 (PPO vs MAPPO, 3 metrics × 3 scenarios)
# ============================================================
def plot_fig5_split(log_dir, save_dir, window=500):
    """
    3 rows (Latency, Throughput, Packet Loss) × 3 cols (Init, Link Fail, Traffic Change).
    Each cell: top=PPO, bottom=MAPPO.
    Truncates x-axis to min length across all types within each method+scenario.
    """
    fig = plt.figure(figsize=(16, 22))
    fig.suptitle('PPO (DRL-OR) vs GAT-MAPPO (Ours) — Abilene Light Load',
                 fontsize=16, fontweight='bold', y=0.98)

    outer = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.28,
                              top=0.93, bottom=0.05, left=0.07, right=0.95)

    metrics = ['delay', 'throughput', 'loss', 'stretch']
    metric_labels = ['Latency (ms)', 'Throughput Ratio', 'Packet Loss Rate', 'Stretch']
    panel_letters = [['(a)', '(b)', '(c)'],
                     ['(d)', '(e)', '(f)'],
                     ['(g)', '(h)', '(i)'],
                     ['(j)', '(k)', '(l)']]

    methods = [('ppo', 'PPO (DRL-OR)'), ('mappo', 'GAT-MAPPO (Ours)')]
    method_bg = {'ppo': '#FFF8E1', 'mappo': '#E3F2FD'}
    method_label_bg = {'ppo': '#FFC107', 'mappo': '#42A5F5'}

    for row, metric in enumerate(metrics):
        for col, scenario in enumerate(SCENARIOS):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col],
                                                     hspace=0.08)
            for mi, (method_key, method_name) in enumerate(methods):
                ax = fig.add_subplot(inner[mi])

                log_path = os.path.join(log_dir, f'{method_key}_{scenario}')

                # Load all types and find min length
                all_data = {}
                for t in range(NUM_TYPES):
                    # stretch is computed from dist (delta_dist) logs
                    if metric == 'stretch':
                        fname = f'dist_type{t}.log'
                    else:
                        fname = f'{metric}_type{t}.log'
                    d = load_log(os.path.join(log_path, fname))
                    if len(d) > 0:
                        if metric == 'stretch':
                            d = compute_stretch(d)
                        all_data[t] = d

                if not all_data:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8, color='gray')
                    continue

                # Truncate to min length across types
                min_len = min(len(v) for v in all_data.values())
                for t in all_data:
                    all_data[t] = all_data[t][:min_len]

                # Plot each type
                for t in sorted(all_data.keys()):
                    s = smooth(all_data[t], window)
                    x = np.arange(len(s))
                    x_scaled = x / 1000.0  # ×10³
                    ax.plot(x_scaled, s, color=TYPE_COLORS[t],
                            linewidth=0.8, label=TYPE_SHORT[t], alpha=0.9)

                ax.set_facecolor(method_bg[method_key])

                # Method label badge
                bbox_color = method_label_bg[method_key]
                ax.text(0.02, 0.92, method_name, transform=ax.transAxes,
                        fontsize=6.5, fontweight='bold', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color,
                                  alpha=0.8, edgecolor='none'),
                        color='white' if method_key == 'mappo' else 'black')

                # Legend (top-right, small)
                ax.legend(fontsize=5.5, loc='upper right', ncol=NUM_TYPES,
                          framealpha=0.7, handlelength=1.2, columnspacing=0.5)

                # Axis labels
                if mi == 1:  # bottom sub-panel
                    ax.set_xlabel(f'Timeslot (×10³)', fontsize=7)
                else:
                    ax.set_xticklabels([])

                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)

            # Panel title
            fig.text(outer[row, col].get_position(fig).x0 +
                     outer[row, col].get_position(fig).width / 2,
                     outer[row, col].get_position(fig).y1 + 0.005,
                     f'{panel_letters[row][col]} {metric_labels[row]} — {SCENARIO_LABELS[scenario]}',
                     ha='center', fontsize=8.5, fontweight='bold')

    # Bottom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=TYPE_COLORS[t], lw=2, label=TYPE_NAMES[t])
                       for t in range(NUM_TYPES)]
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    legend_elements.append(Line2D([0], [0], color='gray', lw=6, alpha=0.3,
                                  label='Top = PPO (DRL-OR)'))
    legend_elements.append(Line2D([0], [0], color='#90CAF9', lw=6, alpha=0.5,
                                  label='Bottom = GAT-MAPPO (Ours)'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.005))

    path = os.path.join(save_dir, 'fig5_split.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 2: Reward comparison (PPO vs MAPPO, 1 row × 3 scenarios)
# ============================================================
def plot_reward(log_dir, save_dir, window=500):
    """Global reward: PPO vs MAPPO overlay for each scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle('Global Reward — PPO (DRL-OR) vs GAT-MAPPO (Ours)',
                 fontsize=14, fontweight='bold')

    colors = {'ppo': '#1976D2', 'mappo': '#D32F2F'}
    labels = {'ppo': 'PPO (DRL-OR)', 'mappo': 'GAT-MAPPO (Ours)'}

    for i, scenario in enumerate(SCENARIOS):
        ax = axes[i]
        for method in ['ppo', 'mappo']:
            fpath = os.path.join(log_dir, f'{method}_{scenario}', 'globalrwd.log')
            data = load_log(fpath)
            if len(data) == 0:
                continue
            s = smooth(data, window)
            x = np.arange(len(s)) / 1000.0
            ax.plot(x, s, color=colors[method], linewidth=1.2,
                    label=labels[method], alpha=0.9)

        ax.set_title(f'Reward — {SCENARIO_LABELS[scenario]}', fontsize=11)
        ax.set_xlabel('Timeslot (×10³)', fontsize=9)
        ax.set_ylabel('Global Reward', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, 'fig5_reward.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 3: Bar Charts (SPR, LBR, QoSR, PPO, MAPPO)
# Like the uploaded paper style — grouped bars per type.
# Uses CONVERGED values (last 2000 steps).
# ============================================================
def plot_bar_charts(log_dir, save_dir):
    """
    3 bar charts: Latency, Throughput Ratio, Avg Packet Loss Ratio.
    Each has groups per flow type, bars per method.
    """
    metrics = {
        'delay':      ('Average Latency (ms)', 'Latency Comparison — Abilene Light Load'),
        'throughput':  ('Throughput Ratio', 'Throughput Ratio Comparison — Abilene Light Load'),
        'loss':        ('Average Packet Loss Ratio (%)', 'Avg. Packet Loss Ratio Comparison — Abilene Light Load'),
        'stretch':     ('Average Stretch', 'Average Stretch Comparison — Abilene Light Load'),
    }

    for metric, (ylabel, title) in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        # Collect converged means
        data = {}  # data[method][type] = mean
        for method_name, log_subdir in METHOD_LOG_MAP.items():
            data[method_name] = {}
            for t in range(NUM_TYPES):
                if metric == 'stretch':
                    fpath = os.path.join(log_dir, log_subdir, f'dist_type{t}.log')
                    vals = load_log(fpath)
                    if len(vals) > 0:
                        vals = compute_stretch(vals)
                    mean_val = get_converged_mean(vals, last_n=2000)
                else:
                    fpath = os.path.join(log_dir, log_subdir, f'{metric}_type{t}.log')
                    vals = load_log(fpath)
                    mean_val = get_converged_mean(vals, last_n=2000)
                if metric == 'loss':
                    mean_val *= 100  # Convert to percentage
                data[method_name][t] = mean_val

        # Plot grouped bars
        n_types = NUM_TYPES
        n_methods = len(METHODS_ORDER)
        bar_width = 0.14
        x = np.arange(n_types)

        for j, method_name in enumerate(METHODS_ORDER):
            offsets = x + (j - n_methods / 2 + 0.5) * bar_width
            values = [data[method_name].get(t, 0) for t in range(n_types)]
            bars = ax.bar(offsets, values, bar_width,
                          label=method_name,
                          color=METHOD_COLORS[method_name],
                          hatch=METHOD_HATCHES[method_name],
                          edgecolor='black', linewidth=0.5, alpha=0.85)

        ax.set_xlabel('Flow Type', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Type {t+1}' for t in range(n_types)], fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=9)

        path = os.path.join(save_dir, f'bar_{metric}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")


# ============================================================
# Figure 4: TensorBoard-style time series (individual metric plots)
# Each metric+scenario as a separate figure, PPO vs MAPPO overlay per type.
# Dark background, smooth lines, like TensorBoard.
# ============================================================
def plot_tensorboard_style(log_dir, save_dir, window=500):
    """
    Individual plots for each (metric, scenario, type) — TensorBoard look.
    Also combined per (metric, scenario) with all types.
    """
    tb_bg = '#1E1E2E'
    tb_grid = '#333355'
    tb_text = '#E0E0E0'
    tb_colors_ppo = {0: '#FF6B6B', 1: '#6BC5FF', 2: '#6BFF8A', 3: '#FFD76B'}
    tb_colors_mappo = {0: '#FF3333', 1: '#3399FF', 2: '#33CC55', 3: '#FFAA00'}

    metrics = ['delay', 'throughput', 'loss', 'stretch']
    metric_labels = {'delay': 'Latency (ms)', 'throughput': 'Throughput Ratio',
                     'loss': 'Packet Loss Rate', 'stretch': 'Stretch'}

    tb_dir = os.path.join(save_dir, 'tensorboard_style')
    os.makedirs(tb_dir, exist_ok=True)

    for metric in metrics:
        for scenario in SCENARIOS:
            # Combined plot (all types, PPO vs MAPPO)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor=tb_bg)
            ax.set_facecolor(tb_bg)

            for method, mcolors, ls in [('ppo', tb_colors_ppo, '--'),
                                         ('mappo', tb_colors_mappo, '-')]:
                log_path = os.path.join(log_dir, f'{method}_{scenario}')
                for t in range(NUM_TYPES):
                    if metric == 'stretch':
                        data = load_log(os.path.join(log_path, f'dist_type{t}.log'))
                        if len(data) > 0:
                            data = compute_stretch(data)
                    else:
                        data = load_log(os.path.join(log_path, f'{metric}_type{t}.log'))
                    if len(data) == 0:
                        continue
                    s = smooth(data, window)
                    x = np.arange(len(s)) / 1000.0
                    method_label = 'PPO' if method == 'ppo' else 'MAPPO'
                    ax.plot(x, s, color=mcolors[t], linewidth=1.0, linestyle=ls,
                            label=f'{method_label} {TYPE_SHORT[t]}', alpha=0.85)

            ax.set_xlabel('Timeslot (×10³)', fontsize=10, color=tb_text)
            ax.set_ylabel(metric_labels[metric], fontsize=10, color=tb_text)
            ax.set_title(f'{metric_labels[metric]} — {SCENARIO_LABELS[scenario]}',
                         fontsize=12, fontweight='bold', color=tb_text)
            ax.tick_params(colors=tb_text, labelsize=8)
            ax.grid(True, color=tb_grid, alpha=0.5)
            ax.legend(fontsize=7, facecolor='#2A2A44', edgecolor='#444466',
                      labelcolor=tb_text, loc='upper right', ncol=2)
            for spine in ax.spines.values():
                spine.set_color(tb_grid)

            path = os.path.join(tb_dir, f'tb_{metric}_{scenario}.png')
            fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=tb_bg)
            plt.close(fig)
            print(f"Saved: {path}")


# ============================================================
# Figure 5: Individual separated plots (one per metric × scenario × method)
# ============================================================
def plot_individual(log_dir, save_dir, window=500):
    """One plot per (method, scenario, metric) with all types."""
    ind_dir = os.path.join(save_dir, 'individual')
    os.makedirs(ind_dir, exist_ok=True)

    metrics = ['delay', 'throughput', 'loss', 'stretch', 'globalrwd']
    metric_labels = {'delay': 'Latency (ms)', 'throughput': 'Throughput Ratio',
                     'loss': 'Packet Loss Rate', 'stretch': 'Stretch',
                     'globalrwd': 'Global Reward'}

    for method in ['ppo', 'mappo']:
        method_label = 'PPO (DRL-OR)' if method == 'ppo' else 'GAT-MAPPO (Ours)'
        for scenario in SCENARIOS:
            log_path = os.path.join(log_dir, f'{method}_{scenario}')
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(8, 4))

                if metric == 'globalrwd':
                    data = load_log(os.path.join(log_path, 'globalrwd.log'))
                    if len(data) > 0:
                        s = smooth(data, window)
                        x = np.arange(len(s)) / 1000.0
                        ax.plot(x, s, color='#1976D2', linewidth=1.0)
                else:
                    all_data = {}
                    for t in range(NUM_TYPES):
                        if metric == 'stretch':
                            d = load_log(os.path.join(log_path, f'dist_type{t}.log'))
                            if len(d) > 0:
                                d = compute_stretch(d)
                        else:
                            d = load_log(os.path.join(log_path, f'{metric}_type{t}.log'))
                        if len(d) > 0:
                            all_data[t] = d
                    if all_data:
                        min_len = min(len(v) for v in all_data.values())
                        for t in sorted(all_data.keys()):
                            s = smooth(all_data[t][:min_len], window)
                            x = np.arange(len(s)) / 1000.0
                            ax.plot(x, s, color=TYPE_COLORS[t], linewidth=1.0,
                                    label=TYPE_SHORT[t])
                        ax.legend(fontsize=8)

                ax.set_title(f'{method_label} — {metric_labels[metric]} — {SCENARIO_LABELS[scenario]}',
                             fontsize=11, fontweight='bold')
                ax.set_xlabel('Timeslot (×10³)', fontsize=9)
                ax.set_ylabel(metric_labels[metric], fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

                path = os.path.join(ind_dir, f'{method}_{scenario}_{metric}.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved: {path}")


# ============================================================
# Figure 6: Throughput in Mbps (actual bandwidth, PPO vs MAPPO)
# ============================================================
def plot_throughput_mbps(log_dir, save_dir, window=500):
    """
    Actual throughput in Mbps = throughput_ratio × demand_Kbps / 1000.
    PPO vs MAPPO overlay per type for each scenario.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Actual Throughput (Mbps) — PPO vs GAT-MAPPO',
                 fontsize=14, fontweight='bold')

    for i, scenario in enumerate(SCENARIOS):
        ax = axes[i]
        for method, ls, alpha_val in [('ppo', '--', 0.6), ('mappo', '-', 0.9)]:
            log_path = os.path.join(log_dir, f'{method}_{scenario}')
            method_label = 'PPO' if method == 'ppo' else 'MAPPO'

            all_data = {}
            for t in range(NUM_TYPES):
                d = load_log(os.path.join(log_path, f'throughput_type{t}.log'))
                if len(d) > 0:
                    all_data[t] = d

            if not all_data:
                continue
            min_len = min(len(v) for v in all_data.values())

            for t in sorted(all_data.keys()):
                ratio = all_data[t][:min_len]
                mbps = ratio * TYPE_DEMANDS_KBPS[t] / 1000.0  # Kbps → Mbps
                s = smooth(mbps, window)
                x = np.arange(len(s)) / 1000.0
                ax.plot(x, s, color=TYPE_COLORS[t], linewidth=1.0,
                        linestyle=ls, alpha=alpha_val,
                        label=f'{method_label} {TYPE_SHORT[t]}')

        ax.set_title(f'Throughput — {SCENARIO_LABELS[scenario]}', fontsize=11)
        ax.set_xlabel('Timeslot (×10³)', fontsize=9)
        ax.set_ylabel('Throughput (Mbps)', fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, 'throughput_mbps.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # Also bar chart of converged Mbps
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_bar = ['PPO (DRL-OR)', 'GAT-MAPPO (Ours)']
    method_dirs = {'PPO (DRL-OR)': 'ppo_initialization',
                   'GAT-MAPPO (Ours)': 'mappo_initialization'}
    bar_colors = {'PPO (DRL-OR)': '#1976D2', 'GAT-MAPPO (Ours)': '#D32F2F'}

    x = np.arange(NUM_TYPES)
    bar_width = 0.3
    for j, method_name in enumerate(methods_bar):
        vals = []
        for t in range(NUM_TYPES):
            fpath = os.path.join(log_dir, method_dirs[method_name],
                                 f'throughput_type{t}.log')
            d = load_log(fpath)
            mean_ratio = get_converged_mean(d)
            mbps = mean_ratio * TYPE_DEMANDS_KBPS[t] / 1000.0
            vals.append(mbps)
        ax.bar(x + (j - 0.5) * bar_width, vals, bar_width,
               label=method_name, color=bar_colors[method_name],
               edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Flow Type', fontsize=11)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Converged Throughput (Mbps) — PPO vs GAT-MAPPO', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Type {t+1}\n({TYPE_DEMANDS_KBPS[t]} Kbps)' for t in range(NUM_TYPES)],
                       fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    path = os.path.join(save_dir, 'bar_throughput_mbps.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 7: Avg Packet Loss bar chart (like the uploaded paper Fig.7)
# Replaces the bottom row of fig5 with a bar chart style comparison.
# ============================================================
def plot_avg_loss_bar(log_dir, save_dir):
    """
    Bar chart comparing avg packet loss ratio across ALL methods
    (SPR, LBR, QoSR, PPO, MAPPO) — single bar chart like the uploaded paper.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # For each method, compute average loss across all types
    # and also per-type
    n_types = NUM_TYPES
    n_methods = len(METHODS_ORDER)
    bar_width = 0.14
    x = np.arange(n_types)

    for j, method_name in enumerate(METHODS_ORDER):
        log_subdir = METHOD_LOG_MAP[method_name]
        values = []
        for t in range(n_types):
            fpath = os.path.join(log_dir, log_subdir, f'loss_type{t}.log')
            vals = load_log(fpath)
            mean_val = get_converged_mean(vals, last_n=2000) * 100  # to %
            values.append(mean_val)
        offsets = x + (j - n_methods / 2 + 0.5) * bar_width
        ax.bar(offsets, values, bar_width,
               label=method_name, color=METHOD_COLORS[method_name],
               hatch=METHOD_HATCHES[method_name],
               edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Flow Type', fontsize=11)
    ax.set_ylabel('Average Packet Loss Ratio (%)', fontsize=11)
    ax.set_title('Average Packet Loss Ratio Comparison — Abilene Light Load',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Type {t+1}' for t in range(n_types)], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    path = os.path.join(save_dir, 'bar_avg_loss.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 8: Fallback Policy Trigger Ratio (from circle.log)
# ============================================================
def plot_fallback_ratio(log_dir, save_dir, fb_window=5000):
    """
    Fallback policy trigger ratio over time.
    Reads circle.log (1=loop triggered, 0=safe).
    Rolling window percentage.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    combos = [
        ('mappo_initialization',  'MAPPO Init',        '#D32F2F', '-'),
        ('mappo_link_failure',    'MAPPO Link Fail',   '#42A5F5', '-'),
        ('mappo_traffic_change',  'MAPPO Traffic',     '#388E3C', '-'),
        ('ppo_initialization',    'PPO Init',          '#FFA726', '--'),
        ('ppo_link_failure',      'PPO Link Fail',     '#7E57C2', '--'),
        ('ppo_traffic_change',    'PPO Traffic',       '#EC407A', '--'),
    ]

    # First pass: compute all ratios and find min length
    all_ratios = {}
    for subdir, label, color, ls in combos:
        fpath = os.path.join(log_dir, subdir, 'circle.log')
        data = load_log(fpath)
        if len(data) == 0:
            continue
        binary = (data > 0.5).astype(float)
        if len(binary) < fb_window:
            continue
        kernel = np.ones(fb_window) / fb_window
        ratio = np.convolve(binary, kernel, mode='valid') * 100
        all_ratios[subdir] = ratio

    if not all_ratios:
        print("  No fallback data found, skipping.")
        plt.close(fig)
        return

    # Truncate all to min length
    min_len = min(len(r) for r in all_ratios.values())
    
    for subdir, label, color, ls in combos:
        if subdir not in all_ratios:
            continue
        ratio = all_ratios[subdir][:min_len]
        x = np.arange(len(ratio))
        ax.plot(x, ratio, color=color, linewidth=1.2, linestyle=ls,
                label=label, alpha=0.85)

    # DRL-OR reference lines
    ax.axhline(y=41.85, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    # ax.text(0, 42.5, 'DRL-OR init beginning (41.85%)', fontsize=7, color='gray')
    ax.axhline(y=2.30, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    # ax.text(0, 3.0, 'DRL-OR converged (2.30%)', fontsize=7, color='gray')

    ax.set_xlabel('Timeslot', fontsize=10)
    ax.set_ylabel('Fallback Trigger Ratio (%)', fontsize=10)
    ax.set_title(f'Fallback Policy Trigger Ratio Over Time (window={fb_window:,})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-1)

    # Format x axis with k
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))

    path = os.path.join(save_dir, 'fallback_ratio.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Table III style: Print converged values
# ============================================================
def print_table3(log_dir):
    """Print Table III comparison values to console."""
    print("\n" + "="*80)
    print("TABLE III — Converged Performance (last 2000 steps)")
    print("="*80)

    for metric, unit in [('delay', 'ms'), ('throughput', 'ratio'), ('loss', '%'), ('stretch', 'ratio')]:
        print(f"\n--- {metric.upper()} ({unit}) ---")
        header = f"{'Method':<20}"
        for t in range(NUM_TYPES):
            header += f"{'Type'+str(t+1):>12}"
        print(header)
        print("-" * (20 + 12 * NUM_TYPES))

        for method_name, log_subdir in METHOD_LOG_MAP.items():
            row = f"{method_name:<20}"
            for t in range(NUM_TYPES):
                if metric == 'stretch':
                    fpath = os.path.join(log_dir, log_subdir, f'dist_type{t}.log')
                    vals = load_log(fpath)
                    if len(vals) > 0:
                        vals = compute_stretch(vals)
                    mean_val = get_converged_mean(vals)
                else:
                    fpath = os.path.join(log_dir, log_subdir, f'{metric}_type{t}.log')
                    vals = load_log(fpath)
                    mean_val = get_converged_mean(vals)
                if metric == 'loss':
                    mean_val *= 100
                row += f"{mean_val:>12.4f}"
            print(row)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Plot all thesis figures')
    parser.add_argument('--log-dir', default='./log', help='Log directory')
    parser.add_argument('--save-dir', default='./figures', help='Output directory')
    parser.add_argument('--window', type=int, default=500, help='Smoothing window')
    parser.add_argument('--fb-window', type=int, default=5000, help='Fallback ratio window')
    parser.add_argument('--num-types', type=int, default=3,
                        help='Number of flow types to plot (3 or 4)')
    args = parser.parse_args()

    global NUM_TYPES
    NUM_TYPES = args.num_types

    os.makedirs(args.save_dir, exist_ok=True)

    print("="*60)
    print("Generating all thesis figures...")
    print(f"  Log dir:  {args.log_dir}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Smoothing window: {args.window}")
    print(f"  Num types: {NUM_TYPES}")
    print("="*60)

    # 1. Fig5 split panels
    print("\n[1/8] Fig5 split panels (PPO vs MAPPO)...")
    plot_fig5_split(args.log_dir, args.save_dir, args.window)

    # 2. Reward comparison
    print("\n[2/8] Reward comparison...")
    plot_reward(args.log_dir, args.save_dir, args.window)

    # 3. Bar charts (all 5 methods)
    print("\n[3/8] Bar charts (SPR, LBR, QoSR, PPO, MAPPO) — latency, throughput, loss, stretch...")
    plot_bar_charts(args.log_dir, args.save_dir)

    # 4. TensorBoard-style plots
    print("\n[4/8] TensorBoard-style plots...")
    plot_tensorboard_style(args.log_dir, args.save_dir, args.window)

    # 5. Individual plots
    print("\n[5/8] Individual plots...")
    plot_individual(args.log_dir, args.save_dir, args.window)

    # 6. Throughput Mbps
    print("\n[6/8] Throughput Mbps plots...")
    plot_throughput_mbps(args.log_dir, args.save_dir, args.window)

    # 7. Avg packet loss bar (paper-style)
    print("\n[7/8] Avg packet loss bar chart...")
    plot_avg_loss_bar(args.log_dir, args.save_dir)

    # 8. Fallback ratio
    print("\n[8/8] Fallback policy trigger ratio...")
    plot_fallback_ratio(args.log_dir, args.save_dir, args.fb_window)

    # Print table
    print_table3(args.log_dir)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {args.save_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()