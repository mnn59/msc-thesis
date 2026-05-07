import os
import numpy as np

def mean_log(path):
    if not os.path.exists(path):
        return None
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(float(line))
    return np.mean(vals) if vals else None

def summarize(log_dir):
    result = {}

    # Latency: Type I, III, IV = 0, 2, 3
    result["latency_type_I"] = mean_log(os.path.join(log_dir, "delay_type0.log"))
    result["latency_type_III"] = mean_log(os.path.join(log_dir, "delay_type2.log"))
    result["latency_type_IV"] = mean_log(os.path.join(log_dir, "delay_type3.log"))

    # Throughput ratio: Type II, III = 1, 2
    t2 = mean_log(os.path.join(log_dir, "throughput_type1.log"))
    t3 = mean_log(os.path.join(log_dir, "throughput_type2.log"))
    result["thrpt_type_II_percent"] = None if t2 is None else t2 * 100
    result["thrpt_type_III_percent"] = None if t3 is None else t3 * 100

    # Loss ratio: Type IV = 3
    l4 = mean_log(os.path.join(log_dir, "loss_type3.log"))
    result["loss_type_IV_percent"] = None if l4 is None else l4 * 100

    return result

for d in [
    "./log/Abi_SHR_5000_lightload",
    "./log/Abi_WP_5000_lightload",
    "./log/Abi_QoS_5000_lightload",
    # "./log/ppo_initi",
]:
    print("\n", d)
    print(summarize(d))