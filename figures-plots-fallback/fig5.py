# """
# fig5.py — Reproduce DRL-OR Fig.5 layout exactly.
# ==================================================
# Each cell has TWO sub-panels stacked vertically:
#   Top:    PPO (DRL-OR) — type1, type2, type3 curves
#   Bottom: GAT-MAPPO (Ours) — type1, type2, type3 curves

# Layout:
#   (a) Latency - init     (b) Latency - link_fail    (c) Latency - traffic_change
#   (d) Thrpt - init       (e) Thrpt - link_fail      (f) Thrpt - traffic_change

# Usage:
#     python3 fig5.py                          # default
#     python3 fig5.py --all                    # + supplementary + reward
#     python3 fig5.py --window 500 --save-path fig5.pdf
# """

# import os, argparse
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.gridspec as gridspec


# def read_log(filepath):
#     if not os.path.exists(filepath):
#         return None
#     vals = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             s = line.strip()
#             if s:
#                 try: vals.append(float(s))
#                 except ValueError: pass
#     return np.array(vals) if vals else None


# def smooth(data, window):
#     if data is None or len(data) < window:
#         return data
#     return np.convolve(data, np.ones(window)/window, mode='valid')


# # DRL-OR paper colors for types
# TYPE_COLORS = ['#7b2d8e', '#2ca02c', '#17becf']  # purple, green, cyan
# TYPE_STYLES = ['-', '--', '-.']
# TYPE_LABELS = ['type1', 'type2', 'type3']


# def plot_fig5_drlor(log_dir, save_path, window):
#     """
#     Exact DRL-OR Fig.5 layout: 2 rows x 3 cols, each cell has 2 sub-panels.
#     Top sub-panel: PPO (IPPO/DRL-OR baseline)
#     Bottom sub-panel: GAT-MAPPO (ours)
#     """
#     phases = [
#         ('initialization', 'initialization'),
#         ('link_failure', 'link failure'),
#         ('traffic_change', 'traffic change'),
#     ]
    
#     row_configs = [
#         ('delay', 'Latency (ms)'),
#         ('throughput', 'Throughput Ratio'),
#     ]
    
#     labels_grid = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
    
#     fig = plt.figure(figsize=(18, 14))
#     fig.suptitle('Figure 5: PPO (DRL-OR) vs GAT-MAPPO (Ours) on Abilene',
#                  fontsize=15, fontweight='bold', y=0.98)
    
#     # Create outer grid: 2 rows x 3 cols
#     outer = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.25,
#                                top=0.93, bottom=0.06, left=0.06, right=0.97)
    
#     num_types = 3  # DRL-OR Fig.5 uses 3 types (first case)
#     methods = ['ppo', 'mappo']
#     method_titles = ['PPO (DRL-OR)', 'GAT-MAPPO (Ours)']
    
#     for row_idx, (metric, ylabel) in enumerate(row_configs):
#         for col_idx, (phase_key, phase_title) in enumerate(phases):
#             # Each cell gets 2 sub-axes (top=PPO, bottom=MAPPO)
#             inner = gridspec.GridSpecFromSubplotSpec(
#                 2, 1, subplot_spec=outer[row_idx, col_idx],
#                 hspace=0.08)
            
#             for m_idx, (method_key, method_title) in enumerate(zip(methods, method_titles)):
#                 ax = fig.add_subplot(inner[m_idx])
#                 prefix = f"{method_key}_{phase_key}"
#                 has_data = False
                
#                 for t in range(num_types):
#                     fpath = os.path.join(log_dir, prefix, f"{metric}_type{t}.log")
#                     data = read_log(fpath)
                    
#                     if data is not None and len(data) > window:
#                         sm = smooth(data, window)
#                         # X-axis in thousands
#                         x = np.arange(len(sm))
#                         ax.plot(x, sm, color=TYPE_COLORS[t], linestyle=TYPE_STYLES[t],
#                                 linewidth=1.3, alpha=0.9)
#                         has_data = True
                
#                 # Formatting
#                 if has_data:
#                     ax.legend([f'{method_key}-{TYPE_LABELS[t]}' for t in range(num_types)],
#                              fontsize=7, loc='upper right', framealpha=0.7,
#                              handlelength=1.5)
                
#                 ax.grid(True, alpha=0.2, linewidth=0.5)
#                 ax.tick_params(labelsize=8)
                
#                 # X-axis: show in ×10³
#                 ax.xaxis.set_major_formatter(
#                     plt.FuncFormatter(lambda x, p: f'{int(x/1000)}' if x >= 1000 else f'{x:.0f}'))
                
#                 if m_idx == 0:
#                     # Top sub-panel: title, no x-label
#                     lbl = labels_grid[row_idx][col_idx]
#                     ax.set_title(f'{lbl} {ylabel} under {phase_title}',
#                                 fontsize=10, fontweight='bold')
#                     ax.tick_params(labelbottom=False)
#                 else:
#                     # Bottom sub-panel: x-label
#                     ax.set_xlabel('Timeslot (×10³)', fontsize=9)
                
#                 # Y-label on left column only
#                 if col_idx == 0:
#                     ax.set_ylabel(ylabel, fontsize=9)
                
#                 if not has_data:
#                     ax.text(0.5, 0.5, f'{method_title}\n(no data)',
#                             transform=ax.transAxes, ha='center', va='center',
#                             fontsize=9, color='gray', style='italic')
    
#     # Overall legend at bottom
#     legend_elements = []
#     for t in range(num_types):
#         legend_elements.append(
#             Line2D([0], [0], color=TYPE_COLORS[t], linestyle=TYPE_STYLES[t],
#                    linewidth=2, label=TYPE_LABELS[t]))
#     legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=1,
#                                    label='Top=PPO, Bottom=MAPPO'))
    
#     fig.legend(handles=legend_elements, loc='lower center', ncol=4,
#                fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))
    
#     fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Figure 5 saved to: {save_path}")
#     plt.close(fig)


# def plot_supplementary(log_dir, save_path, window):
#     """Supplementary: Loss per type + Path Stretch + Reward"""
#     phases = [
#         ('initialization', 'Initialization'),
#         ('link_failure', 'Link Failure'),
#         ('traffic_change', 'Traffic Change'),
#     ]
    
#     fig = plt.figure(figsize=(18, 14))
#     fig.suptitle('Supplementary: Loss, Path Stretch, Reward — PPO vs GAT-MAPPO',
#                  fontsize=14, fontweight='bold', y=0.98)
    
#     outer = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.25,
#                                top=0.93, bottom=0.05, left=0.06, right=0.97)
    
#     row_configs = [
#         ('loss', 'Packet Loss Rate', 3, True),     # per type, split panels
#         ('dist', 'Path Stretch', 0, False),          # aggregated, overlay
#         ('globalrwd', 'Global Reward', 0, False),    # single, overlay
#     ]
    
#     for row_idx, (metric, ylabel, n_types, split) in enumerate(row_configs):
#         for col_idx, (phase_key, phase_title) in enumerate(phases):
#             if split:
#                 # Split panel like fig5
#                 inner = gridspec.GridSpecFromSubplotSpec(
#                     2, 1, subplot_spec=outer[row_idx, col_idx], hspace=0.08)
#                 for m_idx, mk in enumerate(['ppo', 'mappo']):
#                     ax = fig.add_subplot(inner[m_idx])
#                     prefix = f"{mk}_{phase_key}"
#                     for t in range(n_types):
#                         data = read_log(os.path.join(log_dir, prefix, f"{metric}_type{t}.log"))
#                         if data is not None and len(data) > window:
#                             sm = smooth(data, window)
#                             ax.plot(np.arange(len(sm)), sm, color=TYPE_COLORS[t],
#                                     linestyle=TYPE_STYLES[t], linewidth=1.2, alpha=0.85)
#                     ax.grid(True, alpha=0.2); ax.tick_params(labelsize=8)
#                     ax.xaxis.set_major_formatter(
#                         plt.FuncFormatter(lambda x, p: f'{int(x/1000)}' if x >= 1000 else f'{x:.0f}'))
#                     if m_idx == 0:
#                         ax.set_title(f'{ylabel} — {phase_title}', fontsize=10)
#                         ax.tick_params(labelbottom=False)
#                         ax.legend([f'{mk}-t{t}' for t in range(n_types)], fontsize=6, loc='upper right')
#                     else:
#                         ax.set_xlabel('Timeslot (×10³)', fontsize=9)
#                         ax.legend([f'{mk}-t{t}' for t in range(n_types)], fontsize=6, loc='upper right')
#                     if col_idx == 0: ax.set_ylabel(ylabel, fontsize=9)
#             else:
#                 # Overlay PPO vs MAPPO
#                 ax = fig.add_subplot(outer[row_idx, col_idx])
#                 for mk, color, label in [('ppo', '#2196F3', 'PPO'), ('mappo', '#E91E63', 'MAPPO')]:
#                     prefix = f"{mk}_{phase_key}"
#                     if metric == 'globalrwd':
#                         data = read_log(os.path.join(log_dir, prefix, f"{metric}.log"))
#                     else:
#                         # Aggregate across types
#                         all_t = []
#                         for t in range(4):
#                             d = read_log(os.path.join(log_dir, prefix, f"{metric}_type{t}.log"))
#                             if d is not None: all_t.append(d)
#                         if all_t:
#                             mn = min(len(d) for d in all_t)
#                             data = np.mean([d[:mn] for d in all_t], axis=0)
#                         else:
#                             data = None
#                     if data is not None and len(data) > window:
#                         sm = smooth(data, window)
#                         ax.plot(np.arange(len(sm)), sm, color=color, linewidth=1.3,
#                                 alpha=0.9, label=label)
#                 ax.set_title(f'{ylabel} — {phase_title}', fontsize=10)
#                 ax.set_xlabel('Timeslot (×10³)', fontsize=9)
#                 ax.xaxis.set_major_formatter(
#                     plt.FuncFormatter(lambda x, p: f'{int(x/1000)}' if x >= 1000 else f'{x:.0f}'))
#                 if col_idx == 0: ax.set_ylabel(ylabel, fontsize=9)
#                 ax.grid(True, alpha=0.2); ax.legend(fontsize=9); ax.tick_params(labelsize=8)
    
#     fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Supplementary saved to: {save_path}")
#     plt.close(fig)


# def main():
#     parser = argparse.ArgumentParser(description='Reproduce DRL-OR Figure 5')
#     parser.add_argument('--log-dir', default='./log/', help='Log directory')
#     parser.add_argument('--save-path', default='./fig5.png', help='Output path')
#     parser.add_argument('--window', type=int, default=500, help='Smoothing window')
#     parser.add_argument('--all', action='store_true', help='Also generate supplementary')
#     args = parser.parse_args()
    
#     print(f"Reading logs from: {args.log_dir}")
#     plot_fig5_drlor(args.log_dir, args.save_path, args.window)
    
#     if args.all:
#         base = os.path.splitext(args.save_path)[0]
#         plot_supplementary(args.log_dir, f"{base}_supp.png", args.window)


# if __name__ == "__main__":
#     main()




















# """
# fig5.py — PPO vs GAT-MAPPO comparison across 5 scenarios.
# =============================================================
# Layout (2 rows x 5 cols, each cell has PPO top / MAPPO bottom):

#   (a) Latency - init     (b) Latency - link_fail   (c) Latency - traffic   (d) Latency - cascade   (e) Latency - degrade
#   (f) Thrpt   - init     (g) Thrpt   - link_fail   (h) Thrpt   - traffic   (i) Thrpt   - cascade   (j) Thrpt   - degrade

# Event markers:  vertical dashed lines at scenario event timesteps.

# Usage:
#     python3 fig5.py                          # main figure only
#     python3 fig5.py --all                    # + supplementary + reward
#     python3 fig5.py --window 500 --save-path fig5.pdf
#     python3 fig5.py --original-only          # only 3 original scenarios (like DRL-OR paper)
# """

# import os, argparse
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.gridspec as gridspec


# # =============================================================================
# # SCENARIO DEFINITIONS
# # =============================================================================
# ALL_SCENARIOS = [
#     ('initialization',    'Initialization',      []),
#     ('link_failure',      'Link Failure',        [10000]),
#     ('traffic_change',    'Traffic Change',       [10000]),
#     ('cascading_failure', 'Cascading Failure',    [10000, 50000, 100000, 150000]),
#     ('link_degradation',  'Link Degradation',     [10000, 40000, 80000, 120000, 150000]),
# ]

# ORIGINAL_SCENARIOS = ALL_SCENARIOS[:3]  # DRL-OR paper Fig.5

# # Colors, styles
# TYPE_COLORS = ['#7b2d8e', '#2ca02c', '#17becf']  # purple, green, cyan
# TYPE_STYLES = ['-', '--', '-.']
# TYPE_LABELS = ['type1', 'type2', 'type3']

# EVENT_COLOR = '#E53935'   # red dashed lines for events
# EVENT_ALPHA = 0.5


# # =============================================================================
# # UTILITIES
# # =============================================================================
# def read_log(filepath):
#     if not os.path.exists(filepath):
#         return None
#     vals = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             s = line.strip()
#             if s:
#                 try: vals.append(float(s))
#                 except ValueError: pass
#     return np.array(vals) if vals else None


# def smooth(data, window):
#     if data is None or len(data) < window:
#         return data
#     return np.convolve(data, np.ones(window)/window, mode='valid')


# # =============================================================================
# # MAIN FIGURE — Fig.5
# # =============================================================================
# def plot_fig5(log_dir, save_path, window, scenarios):
#     """
#     Each cell has 2 sub-panels stacked vertically:
#       Top:    PPO (DRL-OR baseline)
#       Bottom: GAT-MAPPO (Ours)
#     """
#     n_scenarios = len(scenarios)

#     row_configs = [
#         ('delay',      'Latency (ms)'),
#         ('throughput',  'Throughput Ratio'),
#     ]

#     # Labels (a)-(j) dynamically
#     labels = [chr(ord('a') + i) for i in range(2 * n_scenarios)]
#     labels_grid = [labels[:n_scenarios], labels[n_scenarios:]]

#     fig_w = max(18, 4.2 * n_scenarios)
#     fig = plt.figure(figsize=(fig_w, 14))
#     fig.suptitle('Figure 5: PPO (DRL-OR) vs GAT-MAPPO (Ours) on Abilene',
#                  fontsize=15, fontweight='bold', y=0.98)

#     outer = gridspec.GridSpec(2, n_scenarios, hspace=0.35, wspace=0.25,
#                               top=0.93, bottom=0.06, left=0.05, right=0.97)

#     num_types = 3
#     methods = ['ppo', 'mappo']
#     method_titles = ['PPO (DRL-OR)', 'GAT-MAPPO (Ours)']

#     for row_idx, (metric, ylabel) in enumerate(row_configs):
#         for col_idx, (phase_key, phase_title, event_steps) in enumerate(scenarios):
#             inner = gridspec.GridSpecFromSubplotSpec(
#                 2, 1, subplot_spec=outer[row_idx, col_idx], hspace=0.08)

#             for m_idx, (method_key, method_title) in enumerate(zip(methods, method_titles)):
#                 ax = fig.add_subplot(inner[m_idx])
#                 prefix = f"{method_key}_{phase_key}"
#                 has_data = False

#                 for t in range(num_types):
#                     fpath = os.path.join(log_dir, prefix, f"{metric}_type{t}.log")
#                     data = read_log(fpath)
#                     if data is not None and len(data) > window:
#                         sm = smooth(data, window)
#                         ax.plot(np.arange(len(sm)), sm, color=TYPE_COLORS[t],
#                                 linestyle=TYPE_STYLES[t], linewidth=1.3, alpha=0.9)
#                         has_data = True

#                 # Draw event markers
#                 if event_steps:
#                     for ev_t in event_steps:
#                         ax.axvline(x=ev_t, color=EVENT_COLOR, linestyle=':',
#                                    linewidth=1.0, alpha=EVENT_ALPHA)

#                 # Formatting
#                 if has_data:
#                     ax.legend([f'{method_key}-{TYPE_LABELS[t]}' for t in range(num_types)],
#                               fontsize=6, loc='upper right', framealpha=0.7, handlelength=1.5)

#                 ax.grid(True, alpha=0.2, linewidth=0.5)
#                 ax.tick_params(labelsize=7)
#                 ax.xaxis.set_major_formatter(
#                     plt.FuncFormatter(lambda x, p: f'{int(x/1000)}' if x >= 1000 else f'{x:.0f}'))

#                 if m_idx == 0:
#                     lbl = f'({labels_grid[row_idx][col_idx]})'
#                     ax.set_title(f'{lbl} {ylabel} — {phase_title}',
#                                  fontsize=9, fontweight='bold')
#                     ax.tick_params(labelbottom=False)
#                 else:
#                     ax.set_xlabel('Timeslot (×10³)', fontsize=8)

#                 if col_idx == 0:
#                     ax.set_ylabel(ylabel, fontsize=9)

#                 if not has_data:
#                     ax.text(0.5, 0.5, f'{method_title}\n(no data)',
#                             transform=ax.transAxes, ha='center', va='center',
#                             fontsize=9, color='gray', style='italic')

#     # Legend at bottom
#     legend_elements = []
#     for t in range(num_types):
#         legend_elements.append(
#             Line2D([0], [0], color=TYPE_COLORS[t], linestyle=TYPE_STYLES[t],
#                    linewidth=2, label=TYPE_LABELS[t]))
#     legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=1,
#                                   label='Top=PPO, Bottom=MAPPO'))
#     # Event marker legend entry
#     has_events = any(ev for _, _, ev in scenarios)
#     if has_events:
#         legend_elements.append(Line2D([0], [0], color=EVENT_COLOR, linestyle=':',
#                                       linewidth=1.5, label='Event trigger'))

#     fig.legend(handles=legend_elements, loc='lower center',
#                ncol=len(legend_elements), fontsize=10, frameon=True,
#                bbox_to_anchor=(0.5, 0.0))

#     fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Figure 5 saved to: {save_path}")
#     plt.close(fig)


# # =============================================================================
# # SUPPLEMENTARY FIGURE
# # =============================================================================
# def plot_supplementary(log_dir, save_path, window, scenarios):
#     """Loss per type + Path Stretch + Global Reward for all scenarios."""
#     n_scenarios = len(scenarios)

#     fig = plt.figure(figsize=(max(18, 4.2 * n_scenarios), 16))
#     fig.suptitle('Supplementary: Loss, Path Stretch, Reward — PPO vs GAT-MAPPO',
#                  fontsize=14, fontweight='bold', y=0.98)

#     outer = gridspec.GridSpec(3, n_scenarios, hspace=0.35, wspace=0.25,
#                               top=0.93, bottom=0.05, left=0.06, right=0.97)

#     row_configs = [
#         ('loss',       'Packet Loss Rate',  3,  True),    # per type, split panels
#         ('dist',       'Path Stretch',       0,  False),   # aggregated, overlay
#         ('globalrwd',  'Global Reward',      0,  False),   # single, overlay
#     ]

#     for row_idx, (metric, ylabel, n_types, split) in enumerate(row_configs):
#         for col_idx, (phase_key, phase_title, event_steps) in enumerate(scenarios):
#             if split:
#                 inner = gridspec.GridSpecFromSubplotSpec(
#                     2, 1, subplot_spec=outer[row_idx, col_idx], hspace=0.08)
#                 for m_idx, mk in enumerate(['ppo', 'mappo']):
#                     ax = fig.add_subplot(inner[m_idx])
#                     prefix = f"{mk}_{phase_key}"
#                     for t in range(n_types):
#                         data = read_log(os.path.join(log_dir, prefix, f"{metric}_type{t}.log"))
#                         if data is not None and len(data) > window:
#                             sm = smooth(data, window)
#                             ax.plot(np.arange(len(sm)), sm, color=TYPE_COLORS[t],
#                                     linestyle=TYPE_STYLES[t], linewidth=1.2, alpha=0.85)
#                     # Event markers
#                     for ev_t in event_steps:
#                         ax.axvline(x=ev_t, color=EVENT_COLOR, linestyle=':', linewidth=0.8, alpha=EVENT_ALPHA)
#                     ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
#                     ax.xaxis.set_major_formatter(
#                         plt.FuncFormatter(lambda x, p: f'{int(x/1000)}' if x >= 1000 else f'{x:.0f}'))
#                     if m_idx == 0:
#                         ax.set_title(f'{ylabel} — {phase_title}', fontsize=9)
#                         ax.tick_params(labelbottom=False)
#                         ax.legend([f'{mk}-t{t}' for t in range(n_types)], fontsize=5, loc='upper right')
#                     else:
#                         ax.set_xlabel('Timeslot (×10³)', fontsize=8)
#                         ax.legend([f'{mk}-t{t}' for t in range(n_types)], fontsize=5, loc='upper right')
#                     if col_idx == 0: ax.set_ylabel(ylabel, fontsize=9)
#             else:
#                 ax = fig.add_subplot(outer[row_idx, col_idx])
#                 for mk, color, label in [('ppo', '#2196F3', 'PPO'), ('mappo', '#E91E63', 'MAPPO')]:
#                     prefix = f"{mk}_{phase_key}"
#                     if metric == 'globalrwd':
#                         data = read_log(os.path.join(log_dir, prefix, f"{metric}.log"))
#                     else:
#                         all_t = []
#                         for t in range(4):
#                             d = read_log(os.path.join(log_dir, prefix, f"{metric}_type{t}.log"))
#                             if d is not None: all_t.append(d)
#                         if all_t:
#                             mn = min(len(d) for d in all_t)
#                             data = np.mean([d[:mn] for d in all_t], axis=0)
#                         else:
#                             data = None
#                     if data is not None and len(data) > window:
#                         sm = smooth(data, window)
#                         ax.plot(np.arange(len(sm)), sm, color=color, linewidth=1.3,
#                                 alpha=0.9, label=label)
#                 # Event markers
#                 for ev_t in event_steps:
#                     ax.axvline(x=ev_t, color=EVENT_COLOR, linestyle=':', linewidth=0.8, alpha=EVENT_ALPHA)
#                 ax.set_title(f'{ylabel} — {phase_title}', fontsize=9)
#                 ax.set_xlabel('Timeslot (×10³)', fontsize=8)
#                 ax.xaxis.set_major_formatter(
#                     plt.FuncFormatter(lambda x, p: f'{int(x/1000)}' if x >= 1000 else f'{x:.0f}'))
#                 if col_idx == 0: ax.set_ylabel(ylabel, fontsize=9)
#                 ax.grid(True, alpha=0.2); ax.legend(fontsize=8); ax.tick_params(labelsize=7)

#     fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Supplementary saved to: {save_path}")
#     plt.close(fig)


# # =============================================================================
# # NEW SCENARIOS DETAIL FIGURE
# # =============================================================================
# def plot_new_scenarios_detail(log_dir, save_path, window):
#     """
#     Detailed view of the 2 new scenarios only:
#     cascading_failure and link_degradation.
#     Shows event annotations with labels.
#     """
#     new_scenarios = [
#         ('cascading_failure', 'Cascading Failure', [
#             (10000,  'Link 0-4\nfails'),
#             (50000,  'Link 1-3\nfails'),
#             (100000, 'Link 4-7\nfails'),
#             (150000, 'Link 0-4\nrestored'),
#         ]),
#         ('link_degradation', 'Link Degradation', [
#             (10000,  '60%'),
#             (40000,  '20%'),
#             (80000,  '5%'),
#             (120000, '100%\nrecovery'),
#             (150000, '20%\nagain'),
#         ]),
#     ]

#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle('New Scenarios: Cascading Failure & Link Degradation\nPPO (blue) vs GAT-MAPPO (red)',
#                  fontsize=14, fontweight='bold')

#     metrics = [
#         ('delay',      'Latency (ms)'),
#         ('throughput',  'Throughput Ratio'),
#     ]

#     for col_idx, (phase_key, phase_title, event_annotations) in enumerate(new_scenarios):
#         for row_idx, (metric, ylabel) in enumerate(metrics):
#             ax = axes[row_idx][col_idx]

#             for mk, color, ls, label in [
#                 ('ppo',   '#2196F3', '-',  'PPO'),
#                 ('mappo', '#E91E63', '-',  'GAT-MAPPO'),
#             ]:
#                 prefix = f"{mk}_{phase_key}"
#                 # Average across types
#                 all_t = []
#                 for t in range(3):
#                     d = read_log(os.path.join(log_dir, prefix, f"{metric}_type{t}.log"))
#                     if d is not None: all_t.append(d)
#                 if all_t:
#                     mn = min(len(d) for d in all_t)
#                     data = np.mean([d[:mn] for d in all_t], axis=0)
#                     if len(data) > window:
#                         sm = smooth(data, window)
#                         ax.plot(np.arange(len(sm)), sm, color=color, linestyle=ls,
#                                 linewidth=1.5, alpha=0.9, label=label)

#             # Event annotations with labels
#             ymin, ymax = ax.get_ylim()
#             for ev_t, ev_label in event_annotations:
#                 ax.axvline(x=ev_t, color=EVENT_COLOR, linestyle=':', linewidth=1.0, alpha=0.6)
#                 ax.annotate(ev_label, xy=(ev_t, ymax), xytext=(ev_t + 2000, ymax * 0.95),
#                             fontsize=6, color=EVENT_COLOR, ha='left', va='top',
#                             arrowprops=dict(arrowstyle='->', color=EVENT_COLOR, lw=0.5))

#             ax.set_title(f'{ylabel} — {phase_title}', fontsize=11, fontweight='bold')
#             ax.set_xlabel('Timeslot (×10³)', fontsize=9)
#             ax.set_ylabel(ylabel, fontsize=9)
#             ax.xaxis.set_major_formatter(
#                 plt.FuncFormatter(lambda x, p: f'{int(x/1000)}' if x >= 1000 else f'{x:.0f}'))
#             ax.grid(True, alpha=0.2)
#             ax.legend(fontsize=9)
#             ax.tick_params(labelsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"New scenarios detail saved to: {save_path}")
#     plt.close(fig)


# # =============================================================================
# # MAIN
# # =============================================================================
# def main():
#     parser = argparse.ArgumentParser(description='Figure 5: PPO vs GAT-MAPPO')
#     parser.add_argument('--log-dir', default='./log/', help='Log directory')
#     parser.add_argument('--save-path', default='./fig5.png', help='Output path')
#     parser.add_argument('--window', type=int, default=500, help='Smoothing window')
#     parser.add_argument('--all', action='store_true', help='Generate all figures')
#     parser.add_argument('--original-only', action='store_true',
#                         help='Only 3 original scenarios (like DRL-OR paper)')
#     args = parser.parse_args()

#     scenarios = ORIGINAL_SCENARIOS if args.original_only else ALL_SCENARIOS
#     n = len(scenarios)
#     print(f"Plotting {n} scenarios from: {args.log_dir}")
#     for key, title, evs in scenarios:
#         print(f"  {key}: events at {evs if evs else 'none'}")

#     plot_fig5(args.log_dir, args.save_path, args.window, scenarios)

#     if args.all:
#         base = os.path.splitext(args.save_path)[0]
#         plot_supplementary(args.log_dir, f"{base}_supp.png", args.window, scenarios)
#         plot_new_scenarios_detail(args.log_dir, f"{base}_new_scenarios.png", args.window)


# if __name__ == "__main__":
#     main()




















"""
fig5.py — PPO vs GAT-MAPPO comparison
=======================================
Layout: 3 rows x 3 cols, each cell has 2 sub-panels (PPO top, MAPPO bottom)

         Initialization         Link Failure          Traffic Change
Row 1:  (a) Latency             (b) Latency           (c) Latency
Row 2:  (d) Throughput Ratio    (e) Throughput Ratio   (f) Throughput Ratio
Row 3:  (g) Packet Loss Rate   (h) Packet Loss Rate   (i) Packet Loss Rate

Each sub-panel shows 3 types:
  type1 (red), type2 (blue), type3 (green)

Top sub-panel:   PPO (DRL-OR baseline)
Bottom sub-panel: GAT-MAPPO (Ours)

Note: DRL-OR Fig.5 uses Case 1 (3 flow types only, no type4).
      Type4 (latency-loss-sensitive) is only used in Table III (Case 2).

Usage:
    python3 fig5.py                                    # main figure
    python3 fig5.py --log-dir ./log --save-path fig5.pdf
    python3 fig5.py --window 500
    python3 fig5.py --reward                           # also plot reward comparison
"""

import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


# =============================================================================
# CONFIG
# =============================================================================
SCENARIOS = [
    ('initialization', 'Initialization'),
    ('link_failure',   'Link Failure'),
    ('traffic_change', 'Traffic Change'),
]

METRICS = [
    ('delay',      'Latency (ms)'),
    ('throughput', 'Throughput Ratio'),
    ('loss',       'Packet Loss Rate'),
]

# Type colors: red, blue, green (as you requested)
TYPE_COLORS = ['#D32F2F', '#1976D2', '#388E3C']   # red, blue, green
TYPE_STYLES = ['-', '-', '-']                       # all solid (cleaner)
TYPE_LABELS = ['Type 1 (latency)', 'Type 2 (throughput)', 'Type 3 (lat-thrpt)']
NUM_TYPES = 3

METHODS = [
    ('ppo',   'PPO (DRL-OR)'),
    ('mappo', 'GAT-MAPPO (Ours)'),
]

LABELS = [
    ['(a)', '(b)', '(c)'],
    ['(d)', '(e)', '(f)'],
    ['(g)', '(h)', '(i)'],
]


# =============================================================================
# UTILITIES
# =============================================================================
def read_log(filepath):
    """Read a .log file with one float per line."""
    if not os.path.exists(filepath):
        return None
    vals = []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    vals.append(float(s))
                except ValueError:
                    pass
    return np.array(vals) if vals else None


def smooth(data, window):
    """Moving average smoothing."""
    if data is None or len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


# =============================================================================
# MAIN FIGURE
# =============================================================================
def plot_fig5(log_dir, save_path, window):
    """
    3 rows (metrics) x 3 cols (scenarios)
    Each cell: 2 sub-panels (PPO top, MAPPO bottom)
    Each sub-panel: 3 type curves (red, blue, green)
    """
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        'PPO (DRL-OR) vs GAT-MAPPO (Ours) — Abilene Light Load',
        fontsize=16, fontweight='bold', y=0.98
    )

    outer = gridspec.GridSpec(
        3, 3, hspace=0.30, wspace=0.22,
        top=0.94, bottom=0.05, left=0.06, right=0.97
    )

    for row_idx, (metric_key, metric_label) in enumerate(METRICS):
        for col_idx, (scenario_key, scenario_label) in enumerate(SCENARIOS):

            # Each cell gets 2 sub-axes
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[row_idx, col_idx], hspace=0.08
            )

            for m_idx, (method_key, method_title) in enumerate(METHODS):
                ax = fig.add_subplot(inner[m_idx])
                prefix = f"{method_key}_{scenario_key}"
                has_data = False

                for t in range(NUM_TYPES):
                    fpath = os.path.join(log_dir, prefix, f"{metric_key}_type{t}.log")
                    data = read_log(fpath)

                    if data is not None and len(data) > window:
                        sm = smooth(data, window)
                        x = np.arange(len(sm))
                        ax.plot(
                            x, sm,
                            color=TYPE_COLORS[t],
                            linestyle=TYPE_STYLES[t],
                            linewidth=1.4, alpha=0.9
                        )
                        has_data = True

                # --- Formatting ---
                ax.grid(True, alpha=0.2, linewidth=0.5)
                ax.tick_params(labelsize=8)

                # X-axis in ×10³
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda x, p: f'{int(x / 1000)}' if x >= 1000 else f'{x:.0f}'
                    )
                )

                if m_idx == 0:
                    # Top sub-panel: title + method label, no x-label
                    lbl = LABELS[row_idx][col_idx]
                    ax.set_title(
                        f'{lbl} {metric_label} — {scenario_label}',
                        fontsize=10, fontweight='bold'
                    )
                    ax.tick_params(labelbottom=False)

                    # Method label top-left
                    ax.text(
                        0.02, 0.95, method_title,
                        transform=ax.transAxes, fontsize=8,
                        fontweight='bold', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8)
                    )
                else:
                    # Bottom sub-panel: x-label
                    ax.set_xlabel('Timeslot (×10³)', fontsize=9)

                    # Method label top-left
                    ax.text(
                        0.02, 0.95, method_title,
                        transform=ax.transAxes, fontsize=8,
                        fontweight='bold', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1F5FE', alpha=0.8)
                    )

                # Y-label on left column only
                if col_idx == 0:
                    ax.set_ylabel(metric_label, fontsize=9)

                # Legend inside each sub-panel (compact)
                if has_data:
                    ax.legend(
                        [f'type{t + 1}' for t in range(NUM_TYPES)],
                        fontsize=6, loc='upper right', framealpha=0.7,
                        handlelength=1.5, ncol=1
                    )

                if not has_data:
                    ax.text(
                        0.5, 0.5, f'{method_title}\n(no data yet)',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='gray', style='italic'
                    )

    # --- Bottom legend ---
    legend_elements = [
        Line2D([0], [0], color=TYPE_COLORS[0], linewidth=2.5, label=TYPE_LABELS[0]),
        Line2D([0], [0], color=TYPE_COLORS[1], linewidth=2.5, label=TYPE_LABELS[1]),
        Line2D([0], [0], color=TYPE_COLORS[2], linewidth=2.5, label=TYPE_LABELS[2]),
        Line2D([0], [0], color='gray', linewidth=0, marker='s', markersize=8,
               markerfacecolor='#FFF9C4', label='Top = PPO (DRL-OR)'),
        Line2D([0], [0], color='gray', linewidth=0, marker='s', markersize=8,
               markerfacecolor='#E1F5FE', label='Bottom = GAT-MAPPO (Ours)'),
    ]

    fig.legend(
        handles=legend_elements, loc='lower center', ncol=5,
        fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0)
    )

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 5 saved to: {save_path}")
    plt.close(fig)


# =============================================================================
# REWARD COMPARISON FIGURE
# =============================================================================
def plot_reward(log_dir, save_path, window):
    """
    1 row x 3 cols: Global reward PPO vs MAPPO overlaid
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Global Reward — PPO (DRL-OR) vs GAT-MAPPO (Ours)',
        fontsize=14, fontweight='bold'
    )

    method_colors = {'ppo': '#1976D2', 'mappo': '#D32F2F'}
    method_labels = {'ppo': 'PPO (DRL-OR)', 'mappo': 'GAT-MAPPO (Ours)'}

    for col_idx, (scenario_key, scenario_label) in enumerate(SCENARIOS):
        ax = axes[col_idx]

        for method_key in ['ppo', 'mappo']:
            prefix = f"{method_key}_{scenario_key}"
            fpath = os.path.join(log_dir, prefix, 'globalrwd.log')
            data = read_log(fpath)

            if data is not None and len(data) > window:
                sm = smooth(data, window)
                ax.plot(
                    np.arange(len(sm)), sm,
                    color=method_colors[method_key],
                    linewidth=1.5, alpha=0.9,
                    label=method_labels[method_key]
                )

        ax.set_title(f'Reward — {scenario_label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timeslot (×10³)', fontsize=10)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, p: f'{int(x / 1000)}' if x >= 1000 else f'{x:.0f}'
            )
        )
        if col_idx == 0:
            ax.set_ylabel('Global Reward', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Reward figure saved to: {save_path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Figure 5: PPO vs GAT-MAPPO')
    parser.add_argument('--log-dir', default='./log/', help='Log directory')
    parser.add_argument('--save-path', default='./fig5.png', help='Output path for main figure')
    parser.add_argument('--window', type=int, default=500, help='Smoothing window')
    parser.add_argument('--reward', action='store_true', help='Also plot reward comparison')
    args = parser.parse_args()

    print(f"Reading logs from: {args.log_dir}")
    print(f"Expected log folders:")
    for mk, _ in METHODS:
        for sk, _ in SCENARIOS:
            d = os.path.join(args.log_dir, f"{mk}_{sk}")
            exists = '✓' if os.path.isdir(d) else '✗'
            print(f"  {exists} {d}")

    plot_fig5(args.log_dir, args.save_path, args.window)

    if args.reward:
        base = os.path.splitext(args.save_path)[0]
        plot_reward(args.log_dir, f"{base}_reward.png", args.window)


if __name__ == "__main__":
    main()