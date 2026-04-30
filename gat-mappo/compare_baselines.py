#!/usr/bin/env python3
"""
Compare Baseline Results with DRL-OR/PPO/MAPPO
==============================================
This script reads log files from baseline evaluations and DRL-OR runs,
then generates a comparison table similar to Table III in the DRL-OR paper.

Usage:
------
python3 compare_baselines.py --log-dir ./log

Or specify specific directories:
python3 compare_baselines.py \
    --spr-dir ./log/baseline_SPR \
    --lbr-dir ./log/baseline_LBR \
    --qosr-dir ./log/baseline_QoSR \
    --drl-dir ./log/drl_or_experiment
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict


def read_log_file(filepath):
    """Read a log file and return list of float values"""
    if not os.path.exists(filepath):
        return None
    
    values = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        values.append(float(line))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None
    
    return values if values else None


def load_results(log_dir, method_name):
    """Load all results from a log directory"""
    results = {
        'method': method_name,
        'delay': {},
        'throughput': {},
        'loss': {},
        'count': {}
    }
    
    if not os.path.exists(log_dir):
        print(f"Warning: Directory not found: {log_dir}")
        return None
    
    # Flow types in DRL-OR paper:
    # Type 0: Latency-sensitive (Type I)
    # Type 1: Throughput-sensitive (Type II)
    # Type 2: Latency-throughput-sensitive (Type III)
    # Type 3: Latency-loss-sensitive (Type IV)
    
    for type_id in range(4):
        # Read delay
        delay_data = read_log_file(os.path.join(log_dir, f"delay_type{type_id}.log"))
        if delay_data:
            results['delay'][type_id] = delay_data
            results['count'][type_id] = len(delay_data)
        
        # Read throughput
        throughput_data = read_log_file(os.path.join(log_dir, f"throughput_type{type_id}.log"))
        if throughput_data:
            results['throughput'][type_id] = throughput_data
        
        # Read loss
        loss_data = read_log_file(os.path.join(log_dir, f"loss_type{type_id}.log"))
        if loss_data:
            results['loss'][type_id] = loss_data
    
    return results


def print_table_iii_format(all_results):
    """Print results in Table III format from the DRL-OR paper"""
    
    # Table header
    print("\n" + "=" * 120)
    print("RESULTS COMPARISON (Table III Format)")
    print("=" * 120)
    
    # Column headers
    print("\n{:<12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Method",
        "Latency I", "Latency III", "Latency IV",
        "Thrpt II", "Thrpt III",
        "Loss IV"
    ))
    print("-" * 120)
    
    for results in all_results:
        if results is None:
            continue
        
        method = results['method']
        
        # Get averages
        lat_i = np.mean(results['delay'].get(0, [0])) if results['delay'].get(0) else 0
        lat_iii = np.mean(results['delay'].get(2, [0])) if results['delay'].get(2) else 0
        lat_iv = np.mean(results['delay'].get(3, [0])) if results['delay'].get(3) else 0
        
        thrpt_ii = np.mean(results['throughput'].get(1, [0])) * 100 if results['throughput'].get(1) else 0
        thrpt_iii = np.mean(results['throughput'].get(2, [0])) * 100 if results['throughput'].get(2) else 0
        
        loss_iv = np.mean(results['loss'].get(3, [0])) * 100 if results['loss'].get(3) else 0
        
        print("{:<12} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            method,
            lat_i, lat_iii, lat_iv,
            thrpt_ii, thrpt_iii,
            loss_iv
        ))
    
    print("-" * 120)
    print("\nNote: Latency in ms, Throughput and Loss in %")


def print_detailed_comparison(all_results):
    """Print detailed comparison for all flow types"""
    
    type_names = {
        0: "Type I (Latency-sensitive)",
        1: "Type II (Throughput-sensitive)",
        2: "Type III (Latency-throughput)",
        3: "Type IV (Latency-loss)"
    }
    
    print("\n" + "=" * 100)
    print("DETAILED RESULTS BY FLOW TYPE")
    print("=" * 100)
    
    for type_id in range(4):
        print(f"\n--- {type_names[type_id]} ---")
        print("{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "Method", "Count", "Avg Delay", "Std Delay", "Avg Thrpt", "Avg Loss"
        ))
        print("-" * 72)
        
        for results in all_results:
            if results is None:
                continue
            
            method = results['method']
            count = results['count'].get(type_id, 0)
            
            if count > 0:
                delay = results['delay'].get(type_id, [])
                throughput = results['throughput'].get(type_id, [])
                loss = results['loss'].get(type_id, [])
                
                avg_delay = np.mean(delay) if delay else 0
                std_delay = np.std(delay) if delay else 0
                avg_thrpt = np.mean(throughput) if throughput else 0
                avg_loss = np.mean(loss) if loss else 0
                
                print("{:<12} {:>12} {:>12.2f} {:>12.2f} {:>12.4f} {:>12.4f}".format(
                    method, count, avg_delay, std_delay, avg_thrpt, avg_loss
                ))
            else:
                print("{:<12} {:>12}".format(method, "N/A"))


def save_comparison_csv(all_results, output_file):
    """Save comparison results to CSV file"""
    
    with open(output_file, 'w') as f:
        # Header
        f.write("Method,FlowType,Count,AvgDelay,StdDelay,AvgThroughput,StdThroughput,AvgLoss,StdLoss\n")
        
        type_names = ["TypeI", "TypeII", "TypeIII", "TypeIV"]
        
        for results in all_results:
            if results is None:
                continue
            
            method = results['method']
            
            for type_id in range(4):
                count = results['count'].get(type_id, 0)
                
                if count > 0:
                    delay = results['delay'].get(type_id, [])
                    throughput = results['throughput'].get(type_id, [])
                    loss = results['loss'].get(type_id, [])
                    
                    f.write(f"{method},{type_names[type_id]},{count},"
                           f"{np.mean(delay) if delay else 0:.4f},"
                           f"{np.std(delay) if delay else 0:.4f},"
                           f"{np.mean(throughput) if throughput else 0:.4f},"
                           f"{np.std(throughput) if throughput else 0:.4f},"
                           f"{np.mean(loss) if loss else 0:.4f},"
                           f"{np.std(loss) if loss else 0:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")


def find_log_dirs(base_dir):
    """Auto-detect log directories"""
    dirs = {}
    
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue
        
        name_lower = name.lower()
        if 'spr' in name_lower or 'shr' in name_lower:
            dirs['SPR'] = full_path
        elif 'lbr' in name_lower or 'wp' in name_lower:
            dirs['LBR'] = full_path
        elif 'qosr' in name_lower or 'qos' in name_lower:
            dirs['QoSR'] = full_path
        elif 'drl' in name_lower or 'ppo' in name_lower or 'mappo' in name_lower:
            dirs['DRL-OR'] = full_path
    
    return dirs


def main():
    parser = argparse.ArgumentParser(description='Compare baseline routing results')
    parser.add_argument('--log-dir', type=str, default='./log',
                        help='Base log directory to search for results')
    parser.add_argument('--spr-dir', type=str, default=None,
                        help='SPR results directory')
    parser.add_argument('--lbr-dir', type=str, default=None,
                        help='LBR results directory')
    parser.add_argument('--qosr-dir', type=str, default=None,
                        help='QoSR results directory')
    parser.add_argument('--drl-dir', type=str, default=None,
                        help='DRL-OR results directory')
    parser.add_argument('--output', type=str, default='comparison_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    
    # Auto-detect or use specified directories
    if args.spr_dir is None and args.lbr_dir is None and args.qosr_dir is None:
        print(f"Auto-detecting log directories in {args.log_dir}...")
        detected = find_log_dirs(args.log_dir)
        print(f"Found: {list(detected.keys())}")
        
        args.spr_dir = detected.get('SPR')
        args.lbr_dir = detected.get('LBR')
        args.qosr_dir = detected.get('QoSR')
        if args.drl_dir is None:
            args.drl_dir = detected.get('DRL-OR')
    
    # Load results
    all_results = []
    
    if args.spr_dir:
        print(f"Loading SPR results from {args.spr_dir}")
        all_results.append(load_results(args.spr_dir, 'SPR'))
    
    if args.lbr_dir:
        print(f"Loading LBR results from {args.lbr_dir}")
        all_results.append(load_results(args.lbr_dir, 'LBR'))
    
    if args.qosr_dir:
        print(f"Loading QoSR results from {args.qosr_dir}")
        all_results.append(load_results(args.qosr_dir, 'QoSR'))
    
    if args.drl_dir:
        print(f"Loading DRL-OR results from {args.drl_dir}")
        all_results.append(load_results(args.drl_dir, 'DRL-OR'))
    
    # Filter out None results
    all_results = [r for r in all_results if r is not None]
    
    if not all_results:
        print("\nERROR: No valid results found!")
        print("\nMake sure you have run the baseline evaluations first:")
        print("  ./run_baseline_spr.sh")
        print("  ./run_baseline_lbr.sh")
        print("  ./run_baseline_qosr.sh")
        sys.exit(1)
    
    # Print comparisons
    print_table_iii_format(all_results)
    print_detailed_comparison(all_results)
    
    # Save to CSV
    save_comparison_csv(all_results, args.output)


if __name__ == "__main__":
    main()