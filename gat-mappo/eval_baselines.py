#!/usr/bin/env python3
"""
Baseline Evaluation Script for DRL-OR
=====================================
This script runs SPR (Shortest Path Routing), LBR (Load Balancing Routing), 
and QoSR (QoS Routing) baselines with the Mininet testbed.

Usage:
------
1. Start testbed:     cd testbed && sudo ./run.sh
2. Start controller:  cd ryu-controller && ./run.sh  
3. Run this script:   python3 eval_baselines.py --method SHR --topo Abi --num-steps 10000

Methods:
--------
- SHR: Shortest Hop Routing (SPR in paper)
- WP:  Widest Path / Load Balancing Routing (LBR in paper)
- QoS: QoS Routing (QoSR in paper)
- DS:  Differentiated Services

Author: Based on DRL-OR codebase
"""

import os
import sys
import glob
import argparse
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net_env.simenv import NetEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate baseline routing algorithms')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['SHR', 'WP', 'QoS', 'DS'],
                        help='Routing method: SHR (SPR), WP (LBR), QoS (QoSR), DS')
    parser.add_argument('--topo', type=str, default='Abi',
                        choices=['Abi', 'GEA'],
                        help='Topology: Abi (Abilene) or GEA (GEANT)')
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='Number of flow requests to evaluate')
    parser.add_argument('--load', type=str, default='light',
                        choices=['light', 'heavy'],
                        help='Traffic load scenario')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Custom log directory (default: ./log/<topo>_<method>_<steps>)')
    return parser.parse_args()


def setup_log_directory(log_dir):
    """Create log directory (does NOT delete existing files)"""
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create directory {log_dir}: {e}")
    
    # Check if directory already has log files (warn user but don't delete)
    existing_files = glob.glob(os.path.join(log_dir, '*.log'))
    if existing_files:
        print(f"Warning: Directory {log_dir} already contains {len(existing_files)} log files.")
        print("         New results will overwrite files with the same names.")
    
    return log_dir


def run_baseline_evaluation(args):
    """Run baseline evaluation with specified parameters"""
    
    # Setup topology and demand matrix
    if args.topo == "Abi":
        demand_matrix = "Abi_500.txt"
    else:
        demand_matrix = "GEA_500.txt"
    
    # Setup log directory
    if args.log_dir is None:
        log_dir = f"./log/{args.topo}_{args.method}_{args.num_steps}_{args.load}load"
    else:
        log_dir = args.log_dir
    
    log_dir = setup_log_directory(log_dir)
    print(f"\n{'='*60}")
    print(f"Baseline Evaluation: {args.method}")
    print(f"{'='*60}")
    print(f"Topology:    {args.topo}")
    print(f"Method:      {args.method}")
    print(f"Num Steps:   {args.num_steps}")
    print(f"Load:        {args.load}")
    print(f"Log Dir:     {log_dir}")
    print(f"{'='*60}\n")
    
    # Initialize environment
    print("Initializing network environment...")
    env_args = None
    envs = NetEnv(env_args)
    num_agent, num_node, observation_spaces, action_spaces, num_type = envs.setup(args.topo, demand_matrix)
    envs.reset()
    print(f"Environment initialized: {num_node} nodes, {num_agent} agents, {num_type} flow types")
    
    # Open log files
    log_dist_files = []
    log_demand_files = []
    log_delay_files = []
    log_throughput_files = []
    log_loss_files = []
    
    for i in range(num_type):
        log_dist_files.append(open(f"{log_dir}/dist_type{i}.log", "w", 1))
        log_demand_files.append(open(f"{log_dir}/demand_type{i}.log", "w", 1))
        log_delay_files.append(open(f"{log_dir}/delay_type{i}.log", "w", 1))
        log_throughput_files.append(open(f"{log_dir}/throughput_type{i}.log", "w", 1))
        log_loss_files.append(open(f"{log_dir}/loss_type{i}.log", "w", 1))
    
    # Additional summary logs
    log_summary = open(f"{log_dir}/summary.log", "w", 1)
    
    # Statistics tracking
    stats = {
        'delay': {i: [] for i in range(num_type)},
        'throughput': {i: [] for i in range(num_type)},
        'loss': {i: [] for i in range(num_type)},
        'type_count': {i: 0 for i in range(num_type)}
    }
    
    # Run evaluation
    start_time = time.time()
    print(f"\nStarting evaluation of {args.num_steps} flow requests...")
    print("-" * 60)
    
    for step in range(args.num_steps):
        if step % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (args.num_steps - step - 1) if step > 0 else 0
            print(f"Step {step}/{args.num_steps} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        try:
            rtype, delta_dist, delta_demand, delay, throughput_rate, loss_rate = envs.step_baseline(args.method)
            
            # Log to files
            print(delta_dist, file=log_dist_files[rtype])
            print(delta_demand, file=log_demand_files[rtype])
            print(delay, file=log_delay_files[rtype])
            print(throughput_rate, file=log_throughput_files[rtype])
            print(loss_rate, file=log_loss_files[rtype])
            
            # Track statistics
            stats['delay'][rtype].append(delay)
            stats['throughput'][rtype].append(throughput_rate)
            stats['loss'][rtype].append(loss_rate)
            stats['type_count'][rtype] += 1
            
        except Exception as e:
            print(f"Error at step {step}: {e}")
            continue
    
    # Close log files
    for f in log_dist_files + log_demand_files + log_delay_files + log_throughput_files + log_loss_files:
        f.close()
    
    # Calculate and print summary statistics
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per step: {total_time/args.num_steps*1000:.2f} ms")
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY - {args.method}")
    print(f"{'='*60}")
    
    # Flow type names
    type_names = {
        0: "Type I   (Latency-sensitive)",
        1: "Type II  (Throughput-sensitive)", 
        2: "Type III (Latency-throughput)",
        3: "Type IV  (Latency-loss)"
    }
    
    summary_lines = []
    summary_lines.append(f"Method: {args.method}")
    summary_lines.append(f"Topology: {args.topo}")
    summary_lines.append(f"Load: {args.load}")
    summary_lines.append(f"Total Steps: {args.num_steps}")
    summary_lines.append(f"Total Time: {total_time:.1f}s")
    summary_lines.append("")
    
    print("\n{:<35} {:>10} {:>15} {:>15} {:>15}".format(
        "Flow Type", "Count", "Avg Latency", "Avg Thrpt", "Avg Loss"))
    print("-" * 90)
    
    for i in range(num_type):
        count = stats['type_count'][i]
        if count > 0:
            avg_delay = np.mean(stats['delay'][i])
            avg_throughput = np.mean(stats['throughput'][i])
            avg_loss = np.mean(stats['loss'][i])
            
            print("{:<35} {:>10} {:>15.2f} {:>15.4f} {:>15.4f}".format(
                type_names.get(i, f"Type {i}"), count, avg_delay, avg_throughput, avg_loss))
            
            summary_lines.append(f"{type_names.get(i, f'Type {i}')}: count={count}, "
                               f"delay={avg_delay:.2f}ms, thrpt={avg_throughput:.4f}, loss={avg_loss:.4f}")
    
    print("-" * 90)
    
    # Write summary
    for line in summary_lines:
        print(line, file=log_summary)
    log_summary.close()
    
    print(f"\nLogs saved to: {log_dir}/")
    print(f"  - delay_type[0-3].log")
    print(f"  - throughput_type[0-3].log")
    print(f"  - loss_type[0-3].log")
    print(f"  - dist_type[0-3].log")
    print(f"  - demand_type[0-3].log")
    print(f"  - summary.log")
    
    return stats


def main():
    args = parse_args()
    run_baseline_evaluation(args)


if __name__ == "__main__":
    main()