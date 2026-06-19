# MAPPO-OR: Enhanced Multi-Agent Deep Reinforcement Learning for Online Routing with Multi-Type Service Requirements

This is the official implementation of our paper "MAPPO-OR: Enhanced Multi-Agent Deep Reinforcement Learning for Online Routing with Multi-Type Service Requirements", by Niknejad. Mahdi and Darmani. Mohammad. 

# Requirements
- Python 3.9/0
- Pytorch 2.6.0
- Mininet 2.3.0
- Ryu 4.34

# Running MAPPO-OR
- Run testbed

```
cd testbed
sudo ./run.sh
```

- Run ryu controller
```
cd ryu-controller
./run.sh
```

- Run MAPPO-OR algorithm
```
cd gat-mappo
./run.sh
```

# Plots
- Fig 5.2
```
python fignew4n.py   --root ./gat-mappo/log   --load light   --methods DRL-OR MAPPO-OR   --metrics delay throughput loss   --scenarios initialization link_failure traffic_change   --event link_failure:10000   --event traffic_change:10000   --out ./figures_ch5_3x3/combined_3x3_fixed_event_split_by_type.png   --moving-window 100   --verbose
```

- Fig 5.3
```
python fignew3.py   --root ./gat-mappo/log   --out-dir ./figures_ch5__bars3   --metrics delay throughput loss   --methods SPR QoSR LBR DRL-OR MAPPO-OR   --tail 5000 
```

- Fig 5.4
```
python fignew5.py   --root ./gat-mappo/log   --load heavy   --methods DRL-OR MAPPO-OR   --scenarios initialization link_failure traffic_change   --event link_failure:10000   --event traffic_change:10000   --out ./figures_ch5_global_reward/global_reward_combined_heavy.png   --moving-window 100   --verbose
```

# Compare with baselines
Used to construct Table 5.5.
```
python3 eval_baselines.py --method SHR --topo Abi --load light --num-steps 5000

python3 eval_baselines.py --method WP --topo Abi --load light --num-steps 5000

python3 eval_baselines.py --method QoS --topo Abi --load light --num-steps 5000

python3 eval_baselines.py --method SHR --topo Abi --load heavy --num-steps 5000

python3 eval_baselines.py --method WP --topo Abi --load heavy --num-steps 5000

python3 eval_baselines.py --method QoS --topo Abi --load heavy --num-steps 5000
```

# Evaluate fallback policy
Used to construct Table 5.4.
```
python fallback2.py --log-base ./gat-mappo/log --window 5000 --out ./figures/fallback_ratio.png
```

# Represent topology
Used to construct Fig 5.1.
```
gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r300 -sOutputFile=Abi-topo.png ./gat-mappo/net_env/inputs/Abi/Abi-topo.eps
```