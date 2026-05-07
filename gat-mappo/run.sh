# !/bin/bash
# =================================================================
# Run experiments: PPO (IPPO baseline) vs GAT-MAPPO-CTDE
# =================================================================
# PPO:  main.py      — original DRL-OR IPPO (DTDE)
# MAPPO: main_mappo.py — MAPPO-CTDE with GAT + centralized critic
# =================================================================

# #############  PPO (IPPO BASELINE)  ##############################

# Initialization (Fig.5 a,d)
# python3 main.py \
#     --env-name Abi --demand-matrix Abi_500.txt \
#     --num-env-steps 300000 --num-steps 512 --num-mini-batch 32 \
#     --num-pretrain-epochs 30 --num-pretrain-steps 128 \
#     --lr 2.5e-5 --ppo-epoch 4 --clip-param 0.1 --max-grad-norm 0.5 \
#     --use-gae --use-linear-lr-decay --no-gat \
#     --log-dir ./log/ppo_initialization_heavy \
#     --model-save-path ./model/ppo_initialization_heavy \
#     --seed 1

# Link failure (Fig.5 b,e)
# python3 main.py \
#     --env-name Abi --demand-matrix Abi_500.txt \
#     --num-env-steps 180000 --use-gae --no-gat \
#     --log-dir ./log/ppo_link_failure \
#     --model-load-path ./model/ppo_initialization \
#     --model-save-path ./model/ppo_link_failure

# Traffic change (Fig.5 c,f)
# python3 main.py \
#     --env-name Abi --demand-matrix Abi_500.txt \
#     --num-env-steps 180000 --use-gae --no-gat \
#     --log-dir ./log/ppo_traffic_change \
#     --model-load-path ./model/ppo_initialization \
#     --model-save-path ./model/ppo_traffic_change


# #############  GAT-MAPPO-CTDE  ###################################

# Initialization (Fig.5 a,d)
python3 main_mappo.py \
    --env-name Abi --demand-matrix Abi_500.txt \
    --num-env-steps 300000 --num-steps 512 --num-mini-batch 1 \
    --num-pretrain-epochs 30 --num-pretrain-steps 128 \
    --clip-param 0.2 --ppo-epoch 15 \
    --actor-lr 5e-4 --critic-lr 5e-4 \
    --use-gat --use-linear-lr-decay \
    --log-dir ./log/mappo_initialization_heavy \
    --model-save-path ./model/mappo_initialization_heavy \
    --seed 1

# Link failure (Fig.5 b,e)
# python3 main_mappo.py \
#     --env-name Abi --demand-matrix Abi_500.txt \
#     --num-env-steps 180000 --use-gat \
#     --log-dir ./log/mappo_link_failure \
#     --model-load-path ./model/mappo_initialization \
#     --model-save-path ./model/mappo_link_failure

# Traffic change (Fig.5 c,f)
# python3 main_mappo.py \
#     --env-name Abi --demand-matrix Abi_500.txt \
#     --num-env-steps 180000 --use-gat \
#     --log-dir ./log/mappo_traffic_change \
#     --model-load-path ./model/mappo_initialization \
#     --model-save-path ./model/mappo_traffic_change


# Link degradation scenario
# python3 main_mappo.py \
#     --scenario link_degradation \
#     --env-name Abi --demand-matrix Abi_500.txt \
#     --num-env-steps 180000 --use-gat \
#     --log-dir ./log/mappo_link_degradation \
#     --model-load-path ./model/mappo_initialization \
#     --model-save-path ./model/mappo_link_degradation

# #############  GENERATE PLOTS  ###################################
# python3 fig5.py --log-dir ./log/ --save-path ./fig5.png --all

























#!/bin/bash
# =====================================================================
# GAT-MAPPO-CTDE vs PPO Experiments
# 5 scenarios, optional heavy/light load
# =====================================================================

# ---- CHANGE THIS FOR HIGH LOAD ----
# LOAD="light"              # "light" or "heavy" or "default"
# DEMAND="Abi_500.txt"      # demand matrix (traffic distribution)
# # ------------------------------------

# SEED=1

# ==================================================================
# MAPPO — 5 SCENARIOS
# ==================================================================

# Common MAPPO hyperparameters (from MAPPO paper Table 7)
# MAPPO_ARGS="--clip-param 0.2 --ppo-epoch 15 --actor-lr 5e-4 --critic-lr 5e-4 \
#     --use-gat --use-linear-lr-decay \
#     --num-steps 512 --num-mini-batch 4"

# # 1. Initialization (300k steps, trains from scratch)
# python3 main_mappo.py \
#     --scenario initialization \
#     --demand-matrix $DEMAND --load $LOAD \
#     $MAPPO_ARGS \
#     --num-pretrain-epochs 30 --num-pretrain-steps 128 \
#     --log-dir ./log/mappo_initialization \
#     --model-save-path ./model/mappo_initialization \
#     --seed $SEED

# 2. Link failure (180k steps, loads trained model)
# python3 main_mappo.py \
#     --scenario link_failure \
#     --demand-matrix $DEMAND --load $LOAD \
#     $MAPPO_ARGS \
#     --log-dir ./log/mappo_link_failure \
#     --model-load-path ./model/mappo_initialization \
#     --model-save-path ./model/mappo_link_failure \
#     --seed $SEED

# 3. Traffic change (180k steps)
# python3 main_mappo.py \
#     --scenario traffic_change \
#     --demand-matrix $DEMAND --load $LOAD \
#     $MAPPO_ARGS \
#     --log-dir ./log/mappo_traffic_change \
#     --model-load-path ./model/mappo_initialization \
#     --model-save-path ./model/mappo_traffic_change \
#     --seed $SEED

# 4. Cascading failure — NEW (180k steps)
# python3 main_mappo.py \
#     --scenario cascading_failure \
#     --demand-matrix $DEMAND --load $LOAD \
#     $MAPPO_ARGS \
#     --log-dir ./log/mappo_cascading_failure \
#     --model-load-path ./model/mappo_initialization \
#     --model-save-path ./model/mappo_cascading_failure \
#     --seed $SEED

# 5. Link degradation — NEW (180k steps)
# python3 main_mappo.py \
#     --scenario link_degradation \
#     --demand-matrix $DEMAND --load $LOAD \
#     $MAPPO_ARGS \
#     --log-dir ./log/mappo_link_degradation \
#     --model-load-path ./model/mappo_initialization \
#     --model-save-path ./model/mappo_link_degradation \
#     --seed $SEED


# ==================================================================
# PPO BASELINE — 3 original scenarios (for comparison)
# ==================================================================

# 1. Initialization
# python3 main.py \
#     --env-name Abi --demand-matrix $DEMAND \
#     --num-env-steps 300000 --num-steps 512 --num-mini-batch 4 \
#     --num-pretrain-epochs 30 --num-pretrain-steps 128 \
#     --lr 2.5e-5 --use-gae --use-linear-lr-decay --no-gat \
#     --log-dir ./log/ppo_initialization \
#     --model-save-path ./model/ppo_initialization \
#     --seed $SEED

# 2. Link failure
# python3 main.py \
#     --env-name Abi --demand-matrix $DEMAND \
#     --num-env-steps 180000 --use-gae --no-gat \
#     --log-dir ./log/ppo_link_failure \
#     --model-load-path ./model/ppo_initialization \
#     --model-save-path ./model/ppo_link_failure \
#     --seed $SEED

# 3. Traffic change
# python3 main.py \
#     --env-name Abi --demand-matrix $DEMAND \
#     --num-env-steps 180000 --use-gae --no-gat \
#     --log-dir ./log/ppo_traffic_change \
#     --model-load-path ./model/ppo_initialization \
#     --model-save-path ./model/ppo_traffic_change \
#     --seed $SEED


# ==================================================================
# TensorBoard — monitor all runs live
# ==================================================================
# echo ""
# echo "To monitor all experiments live:"
# echo "  tensorboard --logdir ./log"
# echo ""
# echo "Or specific experiment:"
# echo "  tensorboard --logdir ./log/mappo_cascading_failure/tb"