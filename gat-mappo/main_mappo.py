# """
# GAT-MAPPO-CTDE Training Script
# ================================
# MAPPO with Centralized Training, Decentralized Execution.
# Based on drl-or-fake/main_mappo.py (proven to work).

# Architecture:
#   - Actors (Decentralized): Policy networks with optional GAT, use LOCAL obs
#   - Critic (Centralized): Separate network using GLOBAL state (link caps, usage, losses)
  
# Key differences from main.py (PPO/IPPO baseline):
#   1. Separate CentralizedCritic network
#   2. Global state constructed from envs internals at each step
#   3. Centralized value used for ALL agents (replaces actor's value)
#   4. Separate actor/critic optimizers with own learning rates
#   5. PopArt / Huber loss / clipped value loss

# Usage:
#     python3 main_mappo.py --env-name Abi --demand-matrix Abi_500.txt \
#         --log-dir ./log/mappo_initialization --model-save-path ./model/mappo_initialization
# """

# import copy, glob, os, time
# from collections import deque

# import gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import utils
# from arguments import get_mappo_args
# from model import Policy
# from storage import RolloutStorage
# from algo.mappo import (
#     MAPPO_CTDE,
#     CentralizedCritic,
#     construct_global_state,
#     get_global_state_dim,
# )

# from net_env.simenv import NetEnv
# from net_env.env_utils import extract_adjacency_matrix


# def main():
#     args = get_mappo_args()
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     np.random.seed(args.seed)
#     if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True

#     log_dir = os.path.expanduser(args.log_dir)
#     utils.cleanup_log_dir(log_dir)
#     model_save_path = args.model_save_path
#     model_load_path = args.model_load_path
#     ckpt_step = args.ckpt_steps
#     torch.set_num_threads(1)
#     device = torch.device("cuda:0" if args.cuda else "cpu")
#     print("device: ", device)


#     # === TensorBoard ===
#     writer = None
#     if args.use_tensorboard:
#         try:
#             from torch.utils.tensorboard import SummaryWriter
#             tb_dir = os.path.join(log_dir, 'tb')
#             writer = SummaryWriter(tb_dir)
#             print(f"TensorBoard: {tb_dir}")
#         except ImportError:
#             print("TensorBoard not installed, continuing without it.")

#     print("\n" + "="*70)
#     print("GAT-MAPPO-CTDE Training for SDN Routing")
#     print("="*70)

#     # === Environment ===
#     envs = NetEnv(args)
#     num_agent, num_node, observation_spaces, action_spaces, num_type = \
#         envs.setup(args.env_name, args.demand_matrix)
#     request, obses = envs.reset()

#     global_state_dim = get_global_state_dim(num_node, num_type)
#     print(f"Env: {args.env_name} | Agents: {num_agent} | Nodes: {num_node} | Types: {num_type}")
#     print(f"Obs dim: {observation_spaces[0].shape} | Global state dim: {global_state_dim}")

#     # === GAT adjacency matrix ===
#     adj_matrix = None
#     if args.use_gat:
#         adj_matrix, edge_list = extract_adjacency_matrix(envs, num_node)
#         adj_matrix = adj_matrix.to(device)
#         print(f"GAT enabled: {num_node} nodes, {len(edge_list)} directed edges")

#     # === Log files ===
#     log_dist_files, log_demand_files, log_delay_files = [], [], []
#     log_throughput_files, log_loss_files = [], []
#     for i in range(num_type):
#         log_dist_files.append(open(f"{log_dir}/dist_type{i}.log", "w", 1))
#         log_demand_files.append(open(f"{log_dir}/demand_type{i}.log", "w", 1))
#         log_delay_files.append(open(f"{log_dir}/delay_type{i}.log", "w", 1))
#         log_throughput_files.append(open(f"{log_dir}/throughput_type{i}.log", "w", 1))
#         log_loss_files.append(open(f"{log_dir}/loss_type{i}.log", "w", 1))
#     log_globalrwd_file = open(f"{log_dir}/globalrwd.log", "w", 1)
#     log_circle_file = open(f"{log_dir}/circle.log", "w", 1)

#     # === Build actor networks (decentralized, with optional GAT) ===
#     actor_critics = []
#     rollouts = []
#     for i in range(num_agent):
#         actor_critic = Policy(
#             observation_spaces[i].shape, action_spaces[i], num_node,
#             node_num=num_node, type_num=num_type,
#             adj_matrix=adj_matrix, num_nodes=num_node if args.use_gat else None,
#             base_kwargs={'recurrent': args.recurrent_policy})
#         if model_load_path is not None:
#             mf = os.path.join(model_load_path, f'agent{i}.pth')
#             if os.path.exists(mf):
#                 actor_critic.load_state_dict(torch.load(mf, map_location=device))
#                 print(f"  Loaded actor {i} from {mf}")
#         actor_critic.to(device)
#         actor_critics.append(actor_critic)

#         rollout = RolloutStorage(
#             args.num_pretrain_steps, observation_spaces[i].shape,
#             action_spaces[i], actor_critic.recurrent_hidden_state_size, num_node)
#         rollouts.append(rollout)
#         rollouts[i].obs[0].copy_(obses[i])
#         rollouts[i].to(device)

#     # === Build centralized critic (separate network, GLOBAL state input) ===
#     centralized_critic = CentralizedCritic(
#         global_state_dim=global_state_dim,
#         hidden_size=args.critic_hidden_size,
#         num_layers=args.critic_num_layers,
#         use_feature_normalization=args.use_feature_normalization,
#         use_orthogonal=True,
#         use_popart=args.use_popart,
#         device=device)
#     if model_load_path is not None:
#         cf = os.path.join(model_load_path, 'critic.pth')
#         if os.path.exists(cf):
#             centralized_critic.load_state_dict(torch.load(cf, map_location=device))
#             print(f"  Loaded centralized critic from {cf}")

#     # === Create MAPPO-CTDE agent ===
#     mappo_agent = MAPPO_CTDE(
#         actor_critics=actor_critics,
#         centralized_critic=centralized_critic,
#         clip_param=args.clip_param,
#         ppo_epoch=args.ppo_epoch,
#         num_mini_batch=args.num_mini_batch,
#         value_loss_coef=args.value_loss_coef,
#         entropy_coef=args.entropy_coef,
#         actor_lr=args.actor_lr,
#         critic_lr=args.critic_lr,
#         eps=args.eps,
#         max_grad_norm=args.max_grad_norm,
#         use_huber_loss=args.use_huber_loss,
#         huber_delta=args.huber_delta,
#         use_clipped_value_loss=args.use_clipped_value_loss,
#         use_popart=args.use_popart,
#         use_valuenorm=args.use_valuenorm,
#         use_linear_lr_decay=args.use_linear_lr_decay,
#         device=device)

#     print(f"\n[CTDE Config]")
#     print(f"  Actor LR={args.actor_lr}, Critic LR={args.critic_lr}")
#     print(f"  Clip={args.clip_param}, PPO epochs={args.ppo_epoch}, Mini-batch={args.num_mini_batch}")
#     print(f"  PopArt={args.use_popart}, Huber={args.use_huber_loss}, GAT={args.use_gat}")
#     print(f"  Critic: {args.critic_num_layers}x{args.critic_hidden_size}")

#     # ==================== PRE-TRAINING ====================
#     print(f"\n{'='*70}")
#     print(f"PRE-TRAINING: {args.num_pretrain_epochs} epochs x {args.num_pretrain_steps} steps")
#     print(f"{'='*70}")

#     mappo_agent.prep_training()
#     pretrain_start = time.time()

#     for epoch in range(args.num_pretrain_epochs):
#         epoch_start = time.time()
#         global_states_buffer = []

#         for step in range(args.num_pretrain_steps):
#             with torch.no_grad():
#                 values = [None]*num_agent; actions = [None]*num_agent
#                 action_log_probs = [None]*num_agent
#                 recurrent_hidden_states = [None]*num_agent
#                 condition_states = [None]*num_agent

#                 curr_path = [0]*num_node; agents_flag = [0]*num_agent
#                 curr_agent, path = envs.first_agent()

#                 # Construct GLOBAL STATE for centralized critic
#                 global_state = construct_global_state(
#                     envs._link_capa, envs._link_usage, envs._link_losses,
#                     envs._request.s, envs._request.t,
#                     envs._request.rtype, envs._request.demand,
#                     num_node, num_type, device)
#                 global_states_buffer.append(global_state)

#                 # Centralized value (GLOBAL state -> single V(s))
#                 central_value = mappo_agent.get_values(global_state.unsqueeze(0))

#                 # Hop-by-hop actor decisions (LOCAL obs + condition)
#                 while curr_agent is not None and agents_flag[curr_agent] != 1:
#                     for k in path: curr_path[k] = 1
#                     agents_flag[curr_agent] = 1
#                     cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
#                     _, a, alp, rhs = actor_critics[curr_agent].act(
#                         rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
#                         rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
#                         cs.unsqueeze(0))
#                     values[curr_agent] = central_value  # Use CENTRALIZED value
#                     actions[curr_agent] = a
#                     action_log_probs[curr_agent] = alp
#                     recurrent_hidden_states[curr_agent] = rhs
#                     condition_states[curr_agent] = cs
#                     curr_agent, path = envs.next_agent(curr_agent, a)

#                 # Off-path agents
#                 cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
#                 for k in range(num_agent):
#                     if agents_flag[k] != 1:
#                         _, a, alp, rhs = actor_critics[k].act(
#                             rollouts[k].obs[rollouts[k].step].unsqueeze(0),
#                             rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
#                             cs.unsqueeze(0))
#                         values[k] = central_value
#                         actions[k] = a; action_log_probs[k] = alp
#                         recurrent_hidden_states[k] = rhs; condition_states[k] = cs

#             gfactors = [0.]*num_agent
#             obses, rewards, path, dd, dm, cf, rt, grwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
#             print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
#             print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)

#             for k in range(num_agent):
#                 masks = torch.tensor([1.])
#                 rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
#                     condition_states[k], actions[k].squeeze(0),
#                     action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

#         # Compute returns with centralized critic
#         for k in range(num_agent):
#             with torch.no_grad():
#                 final_gs = construct_global_state(
#                     envs._link_capa, envs._link_usage, envs._link_losses,
#                     envs._request.s, envs._request.t,
#                     envs._request.rtype, envs._request.demand,
#                     num_node, num_type, device)
#                 next_value = mappo_agent.get_values(final_gs.unsqueeze(0)).detach()
#                 rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

#         # MAPPO-CTDE update
#         gs_tensor = torch.stack(global_states_buffer)
#         agent_masks = torch.ones(num_agent)
#         value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
#             rollouts, gs_tensor, agent_masks)

#         for k in range(num_agent):
#             rollouts[k].after_update()

#         if epoch % 5 == 0:
#             print(f"Epoch {epoch:3d}/{args.num_pretrain_epochs} | "
#                   f"V-Loss: {value_loss:.4f} | A-Loss: {action_loss:.4f} | "
#                   f"Entropy: {dist_entropy:.4f} | Time: {time.time()-epoch_start:.1f}s")

#     print(f"\nPre-training done in {(time.time()-pretrain_start)/60:.1f} min\n")

#     # ==================== ONLINE TRAINING ====================
#     print(f"{'='*70}")
#     print(f"ONLINE TRAINING: {args.num_env_steps} steps")
#     print(f"{'='*70}")

#     request, obses = envs.reset()
#     rollouts = []
#     for i in range(num_agent):
#         r = RolloutStorage(args.num_steps, observation_spaces[i].shape,
#             action_spaces[i], actor_critics[i].recurrent_hidden_state_size, num_node)
#         rollouts.append(r)
#         rollouts[i].obs[0].copy_(obses[i])
#         rollouts[i].to(device)

#     mappo_agent.reset_optimizers()
#     episode_rewards = deque(maxlen=100)
#     rwd_buf = deque(maxlen=200)
#     delay_bufs = {t: deque(maxlen=200) for t in range(num_type)}
#     thr_bufs = {t: deque(maxlen=200) for t in range(num_type)}
#     loss_bufs = {t: deque(maxlen=200) for t in range(num_type)}
#     start_time = time.time()
#     global_states_buffer = []

#     for step in range(args.num_env_steps):
#         if args.use_linear_lr_decay:
#             mappo_agent.lr_decay(step, args.num_env_steps)

#         with torch.no_grad():
#             values = [None]*num_agent; actions = [None]*num_agent
#             action_log_probs = [None]*num_agent
#             recurrent_hidden_states = [None]*num_agent
#             condition_states = [None]*num_agent

#             curr_path = [0]*num_node; agents_flag = [0]*num_agent
#             curr_agent, path = envs.first_agent()

#             # Global state for centralized critic
#             global_state = construct_global_state(
#                 envs._link_capa, envs._link_usage, envs._link_losses,
#                 envs._request.s, envs._request.t,
#                 envs._request.rtype, envs._request.demand,
#                 num_node, num_type, device)
#             global_states_buffer.append(global_state)

#             central_value = mappo_agent.get_values(global_state.unsqueeze(0))

#             while curr_agent is not None and agents_flag[curr_agent] != 1:
#                 for k in path: curr_path[k] = 1
#                 agents_flag[curr_agent] = 1
#                 cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
#                 _, a, alp, rhs = actor_critics[curr_agent].act(
#                     rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
#                     rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
#                     cs.unsqueeze(0))
#                 values[curr_agent] = central_value
#                 actions[curr_agent] = a; action_log_probs[curr_agent] = alp
#                 recurrent_hidden_states[curr_agent] = rhs; condition_states[curr_agent] = cs
#                 curr_agent, path = envs.next_agent(curr_agent, a)

#             cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
#             for k in range(num_agent):
#                 if agents_flag[k] != 1:
#                     _, a, alp, rhs = actor_critics[k].act(
#                         rollouts[k].obs[rollouts[k].step].unsqueeze(0),
#                         rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
#                         cs.unsqueeze(0))
#                     values[k] = central_value
#                     actions[k] = a; action_log_probs[k] = alp
#                     recurrent_hidden_states[k] = rhs; condition_states[k] = cs

#         gfactors = [1.]*num_agent
#         obses, rewards, path, dd, dm, cf, rt, grwd, delay, thr, lr = envs.step(actions, gfactors)
#         print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
#         print(delay, file=log_delay_files[rt]); print(thr, file=log_throughput_files[rt])
#         print(lr, file=log_loss_files[rt])
#         print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)

#         episode_rewards.append(grwd)
#         rwd_buf.append(grwd)
#         delay_bufs[rt].append(delay); thr_bufs[rt].append(thr); loss_bufs[rt].append(lr)

#         # TensorBoard
#         if writer and step % 200 == 0 and step > 0:
#             writer.add_scalar('reward/global', np.mean(rwd_buf), step)
#             for t in range(num_type):
#                 if delay_bufs[t]: writer.add_scalar(f'delay/type{t}', np.mean(delay_bufs[t]), step)
#                 if thr_bufs[t]: writer.add_scalar(f'throughput/type{t}', np.mean(thr_bufs[t]), step)
#                 if loss_bufs[t]: writer.add_scalar(f'loss/type{t}', np.mean(loss_bufs[t]), step)

#         # Log training losses when update happens
#         if rollouts[0].step == 0 and writer and step > 0:
#             writer.add_scalar('train/value_loss', value_loss, step)
#             writer.add_scalar('train/action_loss', action_loss, step)
#             writer.add_scalar('train/entropy', dist_entropy, step)

#         # Console
#         if step % 5000 == 0 and step > 0:
#             elapsed = time.time() - start_time
#             sps = step / elapsed
#             avg_rwd = np.mean(rwd_buf) if rwd_buf else 0
#             print(f"Step {step:6d}/{args.num_env_steps} | Reward: {avg_rwd:.3f} | "
#                   f"Speed: {sps:.0f} steps/s | ETA: {(args.num_env_steps-step)/max(sps,1)/60:.1f}min")

#         agent_masks_tensor = torch.tensor(agents_flag, dtype=torch.float32)

#         for k in range(num_agent):
#             masks = torch.tensor([1.] if agents_flag[k]==1 else [0.])
#             rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
#                 condition_states[k], actions[k].squeeze(0),
#                 action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

#             if rollouts[k].step == 0:
#                 with torch.no_grad():
#                     final_gs = construct_global_state(
#                         envs._link_capa, envs._link_usage, envs._link_losses,
#                         envs._request.s, envs._request.t,
#                         envs._request.rtype, envs._request.demand,
#                         num_node, num_type, device)
#                     next_value = mappo_agent.get_values(final_gs.unsqueeze(0)).detach()
#                     rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

#         # Update when rollout is full
#         if rollouts[0].step == 0:
#             gs_tensor = torch.stack(global_states_buffer)
#             value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
#                 rollouts, gs_tensor, agent_masks_tensor)
#             global_states_buffer = []
#             for k in range(num_agent):
#                 rollouts[k].after_update()

#         # Checkpoint
#         if step % ckpt_step == 0 and step > 0 and model_save_path:
#             os.makedirs(model_save_path, exist_ok=True)
#             for i in range(num_agent):
#                 torch.save(actor_critics[i].state_dict(), f"{model_save_path}/agent{i}.pth")
#             torch.save(centralized_critic.state_dict(), f"{model_save_path}/critic.pth")

#     # Final save
#     if model_save_path:
#         os.makedirs(model_save_path, exist_ok=True)
#         for i in range(num_agent):
#             torch.save(actor_critics[i].state_dict(), f"{model_save_path}/agent{i}.pth")
#         torch.save(centralized_critic.state_dict(), f"{model_save_path}/critic.pth")
#         print(f"Model saved to {model_save_path}")

#     # Close logs
#     for f in log_dist_files + log_demand_files + log_delay_files + log_throughput_files + log_loss_files:
#         f.close()
#     log_globalrwd_file.close(); log_circle_file.close()
#     if writer: writer.close()

#     total_time = time.time() - start_time
#     print(f"\nTraining complete in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")


# if __name__ == "__main__":
#     main()









# fix pre-training bug
"""
GAT-MAPPO-CTDE Training Script
================================
MAPPO with Centralized Training, Decentralized Execution.
Based on drl-or-fake/main_mappo.py (proven to work).

Architecture:
  - Actors (Decentralized): Policy networks with optional GAT, use LOCAL obs
  - Critic (Centralized): Separate network using GLOBAL state (link caps, usage, losses)
  
Key differences from main.py (PPO/IPPO baseline):
  1. Separate CentralizedCritic network
  2. Global state constructed from envs internals at each step
  3. Centralized value used for ALL agents (replaces actor's value)
  4. Separate actor/critic optimizers with own learning rates
  5. PopArt / Huber loss / clipped value loss

Usage:
    python3 main_mappo.py --env-name Abi --demand-matrix Abi_500.txt \
        --log-dir ./log/mappo_initialization --model-save-path ./model/mappo_initialization
"""

import copy, glob, os, time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from arguments import get_mappo_args
from model import Policy
from storage import RolloutStorage
from algo.mappo import (
    MAPPO_CTDE,
    CentralizedCritic,
    construct_global_state,
    get_global_state_dim,
)

from net_env.simenv import NetEnv
from net_env.env_utils import extract_adjacency_matrix



# =====================================================================
# SCENARIO CONFIGURATIONS
# =====================================================================
SCENARIO_CONFIGS = {
    'initialization': {
        'description': 'Training from scratch (300k steps)',
        'total_steps': 300000,
        'events': []
    },
    'link_failure': {
        'description': 'Single link failure at t=10k (180k steps)',
        'total_steps': 180000,
        'events': [
            {'timestep': 10000, 'action': 'link_failure',
             'description': 'Link 0-4 fails'}
        ]
    },
    'traffic_change': {
        'description': 'Traffic demand change at t=10k (180k steps)',
        'total_steps': 180000,
        'events': [
            {'timestep': 10000, 'action': 'demand_change',
             'description': 'Switch to mid load (request_times=30)'}
        ]
    },
    'link_degradation': {
        'description': 'Gradual bandwidth degradation on bottleneck link (180k steps)',
        'total_steps': 180000,
        'events': [
            {'timestep': 10000,  'action': 'link_degradation_stage1',
             'description': 'Link 0-4 at 60% capacity (1488 Kbps)'},
            {'timestep': 40000,  'action': 'link_degradation_stage2',
             'description': 'Link 0-4 at 20% capacity (496 Kbps)'},
            {'timestep': 80000,  'action': 'link_degradation_stage3',
             'description': 'Link 0-4 at 5% capacity (124 Kbps)'},
            {'timestep': 120000, 'action': 'link_recovery',
             'description': 'Full recovery to 100% (2480 Kbps)'},
            {'timestep': 150000, 'action': 'link_degradation_stage2',
             'description': 'Second degradation cycle: 20% (test adaptation)'},
        ]
    },
}


def main():
    args = get_mappo_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # --- Scenario selection ---
    scenario = getattr(args, 'scenario', 'initialization')
    sc = SCENARIO_CONFIGS.get(scenario, SCENARIO_CONFIGS['initialization'])
    if scenario != 'initialization':
        args.num_env_steps = sc['total_steps']
    scenario_events = sc['events']
    event_idx = 0
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario} — {sc['description']}")
    if scenario_events:
        for ev in scenario_events:
            print(f"  t={ev['timestep']:>7,}: {ev['description']}")
    print(f"{'='*60}\n")

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    ckpt_step = args.ckpt_steps
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print("device: ", device)


    # === TensorBoard ===
    writer = None
    if args.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(log_dir, 'tb')
            writer = SummaryWriter(tb_dir)
            print(f"TensorBoard: {tb_dir}")
        except ImportError:
            print("TensorBoard not installed, continuing without it.")

    print("\n" + "="*70)
    print("GAT-MAPPO-CTDE Training for SDN Routing")
    print("="*70)

    # === Environment ===
    envs = NetEnv(args)
    num_agent, num_node, observation_spaces, action_spaces, num_type = \
        envs.setup(args.env_name, args.demand_matrix)
    request, obses = envs.reset()

    global_state_dim = get_global_state_dim(num_node, num_type)
    print(f"Env: {args.env_name} | Agents: {num_agent} | Nodes: {num_node} | Types: {num_type}")
    print(f"Obs dim: {observation_spaces[0].shape} | Global state dim: {global_state_dim}")

    # === GAT adjacency matrix ===
    adj_matrix = None
    if args.use_gat:
        adj_matrix, edge_list = extract_adjacency_matrix(envs, num_node)
        adj_matrix = adj_matrix.to(device)
        print(f"GAT enabled: {num_node} nodes, {len(edge_list)} directed edges")

    # === Log files ===
    log_dist_files, log_demand_files, log_delay_files = [], [], []
    log_throughput_files, log_loss_files = [], []
    for i in range(num_type):
        log_dist_files.append(open(f"{log_dir}/dist_type{i}.log", "w", 1))
        log_demand_files.append(open(f"{log_dir}/demand_type{i}.log", "w", 1))
        log_delay_files.append(open(f"{log_dir}/delay_type{i}.log", "w", 1))
        log_throughput_files.append(open(f"{log_dir}/throughput_type{i}.log", "w", 1))
        log_loss_files.append(open(f"{log_dir}/loss_type{i}.log", "w", 1))
    log_globalrwd_file = open(f"{log_dir}/globalrwd.log", "w", 1)
    log_circle_file = open(f"{log_dir}/circle.log", "w", 1)

    # === Build actor networks (decentralized, with optional GAT) ===
    actor_critics = []
    rollouts = []
    for i in range(num_agent):
        actor_critic = Policy(
            observation_spaces[i].shape, action_spaces[i], num_node,
            node_num=num_node, type_num=num_type,
            adj_matrix=adj_matrix, num_nodes=num_node if args.use_gat else None,
            base_kwargs={'recurrent': args.recurrent_policy})
        if model_load_path is not None:
            mf = os.path.join(model_load_path, f'agent{i}.pth')
            if os.path.exists(mf):
                actor_critic.load_state_dict(torch.load(mf, map_location=device))
                print(f"  Loaded actor {i} from {mf}")
        actor_critic.to(device)
        actor_critics.append(actor_critic)

        rollout = RolloutStorage(
            args.num_pretrain_steps, observation_spaces[i].shape,
            action_spaces[i], actor_critic.recurrent_hidden_state_size, num_node)
        rollouts.append(rollout)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)

    # === Build centralized critic (separate network, GLOBAL state input) ===
    centralized_critic = CentralizedCritic(
        global_state_dim=global_state_dim,
        hidden_size=args.critic_hidden_size,
        num_layers=args.critic_num_layers,
        use_feature_normalization=args.use_feature_normalization,
        use_orthogonal=True,
        use_popart=args.use_popart,
        device=device)
    if model_load_path is not None:
        cf = os.path.join(model_load_path, 'critic.pth')
        if os.path.exists(cf):
            centralized_critic.load_state_dict(torch.load(cf, map_location=device))
            print(f"  Loaded centralized critic from {cf}")

    # === Create MAPPO-CTDE agent ===
    mappo_agent = MAPPO_CTDE(
        actor_critics=actor_critics,
        centralized_critic=centralized_critic,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_huber_loss=args.use_huber_loss,
        huber_delta=args.huber_delta,
        use_clipped_value_loss=args.use_clipped_value_loss,
        use_popart=args.use_popart,
        use_valuenorm=args.use_valuenorm,
        use_linear_lr_decay=args.use_linear_lr_decay,
        device=device)

    print(f"\n[CTDE Config]")
    print(f"  Actor LR={args.actor_lr}, Critic LR={args.critic_lr}")
    print(f"  Clip={args.clip_param}, PPO epochs={args.ppo_epoch}, Mini-batch={args.num_mini_batch}")
    print(f"  PopArt={args.use_popart}, Huber={args.use_huber_loss}, GAT={args.use_gat}")
    print(f"  Critic: {args.critic_num_layers}x{args.critic_hidden_size}")

    # ==================== PRE-TRAINING ====================
    # Only pretrain from scratch. Skip if loading a saved model.
    if model_load_path:
        print(f"\nSkipping pre-training (loaded model from {model_load_path})")
    else:
        print(f"\n{'='*70}")
        print(f"PRE-TRAINING: {args.num_pretrain_epochs} epochs x {args.num_pretrain_steps} steps")
        print(f"{'='*70}")

        mappo_agent.prep_training()
        pretrain_start = time.time()

        for epoch in range(args.num_pretrain_epochs):
            epoch_start = time.time()
            global_states_buffer = []

            for step in range(args.num_pretrain_steps):
                with torch.no_grad():
                    values = [None]*num_agent; actions = [None]*num_agent
                    action_log_probs = [None]*num_agent
                    recurrent_hidden_states = [None]*num_agent
                    condition_states = [None]*num_agent

                    curr_path = [0]*num_node; agents_flag = [0]*num_agent
                    curr_agent, path = envs.first_agent()

                    # Construct GLOBAL STATE for centralized critic
                    global_state = construct_global_state(
                        envs._link_capa, envs._link_usage, envs._link_losses,
                        envs._request.s, envs._request.t,
                        envs._request.rtype, envs._request.demand,
                        num_node, num_type, device)
                    global_states_buffer.append(global_state)

                    # Centralized value (GLOBAL state -> single V(s))
                    central_value = mappo_agent.get_values(global_state.unsqueeze(0))

                    # Hop-by-hop actor decisions (LOCAL obs + condition)
                    while curr_agent is not None and agents_flag[curr_agent] != 1:
                        for k in path: curr_path[k] = 1
                        agents_flag[curr_agent] = 1
                        cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
                        _, a, alp, rhs = actor_critics[curr_agent].act(
                            rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                            rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
                            cs.unsqueeze(0))
                        values[curr_agent] = central_value  # Use CENTRALIZED value
                        actions[curr_agent] = a
                        action_log_probs[curr_agent] = alp
                        recurrent_hidden_states[curr_agent] = rhs
                        condition_states[curr_agent] = cs
                        curr_agent, path = envs.next_agent(curr_agent, a)

                    # Off-path agents
                    cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
                    for k in range(num_agent):
                        if agents_flag[k] != 1:
                            _, a, alp, rhs = actor_critics[k].act(
                                rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                                rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
                                cs.unsqueeze(0))
                            values[k] = central_value
                            actions[k] = a; action_log_probs[k] = alp
                            recurrent_hidden_states[k] = rhs; condition_states[k] = cs

                gfactors = [0.]*num_agent
                obses, rewards, path, dd, dm, cf, rt, grwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
                print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
                print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)

                for k in range(num_agent):
                    masks = torch.tensor([1.])
                    rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
                        condition_states[k], actions[k].squeeze(0),
                        action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

            # Compute returns with centralized critic
            for k in range(num_agent):
                with torch.no_grad():
                    final_gs = construct_global_state(
                        envs._link_capa, envs._link_usage, envs._link_losses,
                        envs._request.s, envs._request.t,
                        envs._request.rtype, envs._request.demand,
                        num_node, num_type, device)
                    next_value = mappo_agent.get_values(final_gs.unsqueeze(0)).detach()
                    rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

            # MAPPO-CTDE update
            gs_tensor = torch.stack(global_states_buffer)
            agent_masks = torch.ones(num_agent)
            value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
                rollouts, gs_tensor, agent_masks)

            for k in range(num_agent):
                rollouts[k].after_update()

            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}/{args.num_pretrain_epochs} | "
                      f"V-Loss: {value_loss:.4f} | A-Loss: {action_loss:.4f} | "
                      f"Entropy: {dist_entropy:.4f} | Time: {time.time()-epoch_start:.1f}s")

        print(f"\nPre-training done in {(time.time()-pretrain_start)/60:.1f} min\n")

    # ==================== ONLINE TRAINING ====================
    print(f"{'='*70}")
    print(f"ONLINE TRAINING: {args.num_env_steps} steps")
    print(f"{'='*70}")

    request, obses = envs.reset()
    rollouts = []
    for i in range(num_agent):
        r = RolloutStorage(args.num_steps, observation_spaces[i].shape,
            action_spaces[i], actor_critics[i].recurrent_hidden_state_size, num_node)
        rollouts.append(r)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)

    mappo_agent.reset_optimizers()
    episode_rewards = deque(maxlen=100)
    rwd_buf = deque(maxlen=200)
    delay_bufs = {t: deque(maxlen=200) for t in range(num_type)}
    thr_bufs = {t: deque(maxlen=200) for t in range(num_type)}
    loss_bufs = {t: deque(maxlen=200) for t in range(num_type)}
    start_time = time.time()
    global_states_buffer = []

    # Initialize value_loss etc for TensorBoard (before first update)
    value_loss = 0.0; action_loss = 0.0; dist_entropy = 0.0

    for step in range(args.num_env_steps):
        if args.use_linear_lr_decay:
            mappo_agent.lr_decay(step, args.num_env_steps)

        # === Check for scheduled scenario events ===
        while event_idx < len(scenario_events) and step >= scenario_events[event_idx]['timestep']:
            ev = scenario_events[event_idx]
            print(f"\n{'='*50}")
            print(f">>> Step {step:,}: EVENT — {ev['description']}")
            print(f"{'='*50}")
            envs.change_env(ev['action'])
            event_idx += 1

        with torch.no_grad():
            values = [None]*num_agent; actions = [None]*num_agent
            action_log_probs = [None]*num_agent
            recurrent_hidden_states = [None]*num_agent
            condition_states = [None]*num_agent

            curr_path = [0]*num_node; agents_flag = [0]*num_agent
            curr_agent, path = envs.first_agent()

            # Global state for centralized critic
            global_state = construct_global_state(
                envs._link_capa, envs._link_usage, envs._link_losses,
                envs._request.s, envs._request.t,
                envs._request.rtype, envs._request.demand,
                num_node, num_type, device)
            global_states_buffer.append(global_state)

            central_value = mappo_agent.get_values(global_state.unsqueeze(0))

            while curr_agent is not None and agents_flag[curr_agent] != 1:
                for k in path: curr_path[k] = 1
                agents_flag[curr_agent] = 1
                cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
                _, a, alp, rhs = actor_critics[curr_agent].act(
                    rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                    rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
                    cs.unsqueeze(0))
                values[curr_agent] = central_value
                actions[curr_agent] = a; action_log_probs[curr_agent] = alp
                recurrent_hidden_states[curr_agent] = rhs; condition_states[curr_agent] = cs
                curr_agent, path = envs.next_agent(curr_agent, a)

            cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
            for k in range(num_agent):
                if agents_flag[k] != 1:
                    _, a, alp, rhs = actor_critics[k].act(
                        rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                        rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
                        cs.unsqueeze(0))
                    values[k] = central_value
                    actions[k] = a; action_log_probs[k] = alp
                    recurrent_hidden_states[k] = rhs; condition_states[k] = cs

        gfactors = [1.]*num_agent
        obses, rewards, path, dd, dm, cf, rt, grwd, delay, thr, lr = envs.step(actions, gfactors)
        print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
        print(delay, file=log_delay_files[rt]); print(thr, file=log_throughput_files[rt])
        print(lr, file=log_loss_files[rt])
        print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)

        episode_rewards.append(grwd)
        rwd_buf.append(grwd)
        delay_bufs[rt].append(delay); thr_bufs[rt].append(thr); loss_bufs[rt].append(lr)

        # TensorBoard
        if writer and step % 200 == 0 and step > 0:
            writer.add_scalar('reward/global', np.mean(rwd_buf), step)
            for t in range(num_type):
                if delay_bufs[t]: writer.add_scalar(f'delay/type{t}', np.mean(delay_bufs[t]), step)
                if thr_bufs[t]: writer.add_scalar(f'throughput/type{t}', np.mean(thr_bufs[t]), step)
                if loss_bufs[t]: writer.add_scalar(f'loss/type{t}', np.mean(loss_bufs[t]), step)

        # Log training losses when update happens
        if rollouts[0].step == 0 and writer and step > 0:
            writer.add_scalar('train/value_loss', value_loss, step)
            writer.add_scalar('train/action_loss', action_loss, step)
            writer.add_scalar('train/entropy', dist_entropy, step)

        # Console
        if step % 5000 == 0 and step > 0:
            elapsed = time.time() - start_time
            sps = step / elapsed
            avg_rwd = np.mean(rwd_buf) if rwd_buf else 0
            print(f"Step {step:6d}/{args.num_env_steps} | Reward: {avg_rwd:.3f} | "
                  f"Speed: {sps:.0f} steps/s | ETA: {(args.num_env_steps-step)/max(sps,1)/60:.1f}min")

        agent_masks_tensor = torch.tensor(agents_flag, dtype=torch.float32)

        for k in range(num_agent):
            masks = torch.tensor([1.] if agents_flag[k]==1 else [0.])
            rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
                condition_states[k], actions[k].squeeze(0),
                action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

            if rollouts[k].step == 0:
                with torch.no_grad():
                    final_gs = construct_global_state(
                        envs._link_capa, envs._link_usage, envs._link_losses,
                        envs._request.s, envs._request.t,
                        envs._request.rtype, envs._request.demand,
                        num_node, num_type, device)
                    next_value = mappo_agent.get_values(final_gs.unsqueeze(0)).detach()
                    rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

        # Update when rollout is full
        if rollouts[0].step == 0:
            gs_tensor = torch.stack(global_states_buffer)
            value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
                rollouts, gs_tensor, agent_masks_tensor)
            global_states_buffer = []
            for k in range(num_agent):
                rollouts[k].after_update()

        # Checkpoint
        if step % ckpt_step == 0 and step > 0 and model_save_path:
            os.makedirs(model_save_path, exist_ok=True)
            for i in range(num_agent):
                torch.save(actor_critics[i].state_dict(), f"{model_save_path}/agent{i}.pth")
            torch.save(centralized_critic.state_dict(), f"{model_save_path}/critic.pth")

    # Final save
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
        for i in range(num_agent):
            torch.save(actor_critics[i].state_dict(), f"{model_save_path}/agent{i}.pth")
        torch.save(centralized_critic.state_dict(), f"{model_save_path}/critic.pth")
        print(f"Model saved to {model_save_path}")

    # Close logs
    for f in log_dist_files + log_demand_files + log_delay_files + log_throughput_files + log_loss_files:
        f.close()
    log_globalrwd_file.close(); log_circle_file.close()
    if writer: writer.close()

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")


if __name__ == "__main__":
    main()









# """
# GAT-MAPPO-CTDE Training Script
# ================================
# Supports 5 scenarios:
#   initialization, link_failure, traffic_change, cascading_failure, link_degradation

# Usage:
#   python3 main_mappo.py --scenario initialization --demand-matrix Abi_500.txt
#   python3 main_mappo.py --scenario cascading_failure --model-load-path ./model/mappo_init
#   python3 main_mappo.py --scenario link_degradation --load heavy
# """

# import copy, glob, os, time
# from collections import deque

# import gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import utils
# from arguments import get_mappo_args
# from model import Policy
# from storage import RolloutStorage
# from algo.mappo import (
#     MAPPO_CTDE,
#     CentralizedCritic,
#     construct_global_state,
#     get_global_state_dim,
# )

# from net_env.simenv import NetEnv
# from net_env.env_utils import extract_adjacency_matrix


# # =====================================================================
# # SCENARIO CONFIGURATIONS
# # =====================================================================
# SCENARIO_CONFIGS = {
#     'initialization': {
#         'description': 'Training from scratch (DRL-OR Fig.5 a,d)',
#         'total_steps': 300000,
#         'events': []
#     },
#     'link_failure': {
#         'description': 'Single link failure at t=10k (DRL-OR Fig.5 b,e)',
#         'total_steps': 180000,
#         'events': [
#             {'timestep': 10000, 'action': 'link_failure',
#              'description': 'Link 0-4 fails'}
#         ]
#     },
#     'traffic_change': {
#         'description': 'Traffic demand change at t=10k (DRL-OR Fig.5 c,f)',
#         'total_steps': 180000,
#         'events': [
#             {'timestep': 10000, 'action': 'demand_change',
#              'description': 'Switch to mid load (request_times=30)'}
#         ]
#     },
#     'cascading_failure': {
#         'description': 'Multiple cascading link failures (180k steps)',
#         'total_steps': 180000,
#         'events': [
#             {'timestep': 10000,  'action': 'cascade_failure_1',
#              'description': 'Link 0-4 fails (bottleneck)'},
#             {'timestep': 50000,  'action': 'cascade_failure_2',
#              'description': 'Link 1-3 fails (rerouting stress)'},
#             {'timestep': 100000, 'action': 'cascade_failure_3',
#              'description': 'Link 4-7 fails (alt path)'},
#             {'timestep': 150000, 'action': 'partial_recovery',
#              'description': 'Link 0-4 restored'},
#         ]
#     },
#     'link_degradation': {
#         'description': 'Gradual bandwidth degradation (180k steps)',
#         'total_steps': 180000,
#         'events': [
#             {'timestep': 10000,  'action': 'link_degradation_stage1',
#              'description': 'Link 0-4 at 60% capacity'},
#             {'timestep': 40000,  'action': 'link_degradation_stage2',
#              'description': 'Link 0-4 at 20% capacity'},
#             {'timestep': 80000,  'action': 'link_degradation_stage3',
#              'description': 'Link 0-4 at 5% capacity'},
#             {'timestep': 120000, 'action': 'link_recovery',
#              'description': 'Full recovery to 100%'},
#             {'timestep': 150000, 'action': 'link_degradation_stage2',
#              'description': 'Second cycle: 20% (test learned adaptation)'},
#         ]
#     },
# }


# # =====================================================================
# # ENVIRONMENT CHANGE — works with UNMODIFIED simenv.py
# # =====================================================================
# def apply_env_change(envs, action, num_node):
#     """Apply environment changes by directly manipulating env internals."""

#     # --- Original DRL-OR scenarios (same as simenv.change_env) ---
#     if action == 'link_failure':
#         envs._link_capa[0][4] = 0
#         envs._link_capa[4][0] = 0
#         _recalc_shortest_paths(envs, num_node)
#         print(f"  [ENV] Link 0-4 failed")

#     elif action == 'demand_change':
#         envs._request_times = [[30], [30], [30], [30]]
#         print(f"  [ENV] Demand changed to mid load (request_times=30)")

#     # --- NEW: Cascading failures ---
#     elif action == 'cascade_failure_1':
#         # Save original capacity for potential recovery
#         if not hasattr(envs, '_original_capa_04'):
#             envs._original_capa_04 = envs._link_capa[0][4]
#         envs._link_capa[0][4] = 0
#         envs._link_capa[4][0] = 0
#         _recalc_shortest_paths(envs, num_node)
#         print(f"  [ENV] CASCADE 1: Link 0-4 failed")

#     elif action == 'cascade_failure_2':
#         envs._link_capa[1][3] = 0
#         envs._link_capa[3][1] = 0
#         _recalc_shortest_paths(envs, num_node)
#         print(f"  [ENV] CASCADE 2: Link 1-3 failed")

#     elif action == 'cascade_failure_3':
#         envs._link_capa[4][7] = 0
#         envs._link_capa[7][4] = 0
#         _recalc_shortest_paths(envs, num_node)
#         print(f"  [ENV] CASCADE 3: Link 4-7 failed")

#     # --- NEW: Link degradation ---
#     elif action == 'link_degradation_stage1':
#         if not hasattr(envs, '_original_capa_04'):
#             envs._original_capa_04 = envs._link_capa[0][4]
#         cap = int(envs._original_capa_04 * 0.6)
#         envs._link_capa[0][4] = cap
#         envs._link_capa[4][0] = cap
#         print(f"  [ENV] DEGRADE stage1: Link 0-4 at 60% ({cap} Kbps)")

#     elif action == 'link_degradation_stage2':
#         if not hasattr(envs, '_original_capa_04'):
#             envs._original_capa_04 = envs._link_capa[0][4]
#         cap = int(envs._original_capa_04 * 0.2)
#         envs._link_capa[0][4] = cap
#         envs._link_capa[4][0] = cap
#         print(f"  [ENV] DEGRADE stage2: Link 0-4 at 20% ({cap} Kbps)")

#     elif action == 'link_degradation_stage3':
#         if not hasattr(envs, '_original_capa_04'):
#             envs._original_capa_04 = envs._link_capa[0][4]
#         cap = max(1, int(envs._original_capa_04 * 0.05))
#         envs._link_capa[0][4] = cap
#         envs._link_capa[4][0] = cap
#         print(f"  [ENV] DEGRADE stage3: Link 0-4 at 5% ({cap} Kbps)")

#     # --- Recovery ---
#     elif action == 'link_recovery':
#         if hasattr(envs, '_original_capa_04'):
#             envs._link_capa[0][4] = envs._original_capa_04
#             envs._link_capa[4][0] = envs._original_capa_04
#             print(f"  [ENV] RECOVERY: Link 0-4 restored to 100% ({envs._original_capa_04} Kbps)")
#         _recalc_shortest_paths(envs, num_node)

#     elif action == 'partial_recovery':
#         if hasattr(envs, '_original_capa_04'):
#             envs._link_capa[0][4] = envs._original_capa_04
#             envs._link_capa[4][0] = envs._original_capa_04
#             print(f"  [ENV] PARTIAL RECOVERY: Link 0-4 restored ({envs._original_capa_04} Kbps)")
#         _recalc_shortest_paths(envs, num_node)

#     else:
#         raise ValueError(f"Unknown env action: {action}")


# def _recalc_shortest_paths(envs, num_node):
#     """Recalculate shortest paths (Floyd-Warshall) after topology change."""
#     envs._shr_dist = []
#     for i in range(num_node):
#         envs._shr_dist.append([])
#         for j in range(num_node):
#             if j == i:
#                 envs._shr_dist[i].append(0)
#             elif (j in envs._link_lists[i]) and (envs._link_capa[i][j] > 0):
#                 envs._shr_dist[i].append(1)
#             else:
#                 envs._shr_dist[i].append(1e6)
#     for k in range(num_node):
#         for i in range(num_node):
#             for j in range(num_node):
#                 if envs._shr_dist[i][j] > envs._shr_dist[i][k] + envs._shr_dist[k][j]:
#                     envs._shr_dist[i][j] = envs._shr_dist[i][k] + envs._shr_dist[k][j]


# def main():
#     args = get_mappo_args()
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     np.random.seed(args.seed)
#     if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True

#     # --- Scenario selection ---
#     scenario = args.scenario
#     if scenario not in SCENARIO_CONFIGS:
#         print(f"ERROR: Unknown scenario '{scenario}'")
#         print(f"Available: {list(SCENARIO_CONFIGS.keys())}")
#         return
#     sc = SCENARIO_CONFIGS[scenario]
#     args.num_env_steps = sc['total_steps']

#     log_dir = os.path.expanduser(args.log_dir)
#     utils.cleanup_log_dir(log_dir)
#     model_save_path = args.model_save_path
#     model_load_path = args.model_load_path
#     ckpt_step = args.ckpt_steps
#     torch.set_num_threads(1)
#     device = torch.device("cuda:0" if args.cuda else "cpu")

#     # === TensorBoard — FIXED ===
#     writer = None
#     if args.use_tensorboard:
#         try:
#             from torch.utils.tensorboard import SummaryWriter
#             tb_dir = os.path.join(log_dir, 'tb')
#             os.makedirs(tb_dir, exist_ok=True)
#             writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
#             writer.add_scalar('_init/start', 0, 0)
#             writer.flush()
#             print(f"TensorBoard ready: tensorboard --logdir {tb_dir}")
#         except ImportError:
#             print("WARNING: tensorboard not installed (pip install tensorboard)")

#     print("\n" + "="*70)
#     print(f"GAT-MAPPO-CTDE | Scenario: {scenario}")
#     print(f"  {sc['description']}")
#     print(f"  Total steps: {sc['total_steps']:,}")
#     print(f"  Demand matrix: {args.demand_matrix}")
#     print(f"  Load: {args.load}")
#     if sc['events']:
#         print(f"  Events ({len(sc['events'])}):")
#         for ev in sc['events']:
#             print(f"    t={ev['timestep']:>7,}: {ev.get('description', ev['action'])}")
#     print("="*70)

#     # === Environment ===
#     envs = NetEnv(args)
#     num_agent, num_node, observation_spaces, action_spaces, num_type = \
#         envs.setup(args.env_name, args.demand_matrix)
#     request, obses = envs.reset()

#     # --- Apply load setting AFTER setup (overrides simenv defaults) ---
#     if args.load == 'heavy':
#         envs._request_times = [[50], [50], [50], [50]]
#         print(f"Load: HEAVY (request_times=50, more concurrent flows)")
#     elif args.load == 'light':
#         envs._request_times = [[10], [10], [10], [10]]
#         print(f"Load: LIGHT (request_times=10)")
#     # else 'default' — keep whatever simenv.py sets

#     global_state_dim = get_global_state_dim(num_node, num_type)
#     print(f"Env: {args.env_name} | Agents: {num_agent} | Nodes: {num_node} | Types: {num_type}")
#     print(f"Global state dim: {global_state_dim}")

#     # === GAT adjacency matrix ===
#     adj_matrix = None
#     if args.use_gat:
#         adj_matrix, edge_list = extract_adjacency_matrix(envs, num_node)
#         adj_matrix = adj_matrix.to(device)
#         print(f"GAT enabled: {num_node} nodes, {len(edge_list)} edges")

#     # === Log files ===
#     log_dist_files, log_demand_files, log_delay_files = [], [], []
#     log_throughput_files, log_loss_files = [], []
#     for i in range(num_type):
#         log_dist_files.append(open(f"{log_dir}/dist_type{i}.log", "w", 1))
#         log_demand_files.append(open(f"{log_dir}/demand_type{i}.log", "w", 1))
#         log_delay_files.append(open(f"{log_dir}/delay_type{i}.log", "w", 1))
#         log_throughput_files.append(open(f"{log_dir}/throughput_type{i}.log", "w", 1))
#         log_loss_files.append(open(f"{log_dir}/loss_type{i}.log", "w", 1))
#     log_globalrwd_file = open(f"{log_dir}/globalrwd.log", "w", 1)
#     log_circle_file = open(f"{log_dir}/circle.log", "w", 1)
#     log_events_file = open(f"{log_dir}/events.log", "w", 1)

#     # === Build actors ===
#     actor_critics = []
#     rollouts = []
#     for i in range(num_agent):
#         ac = Policy(
#             observation_spaces[i].shape, action_spaces[i], num_node,
#             node_num=num_node, type_num=num_type,
#             adj_matrix=adj_matrix, num_nodes=num_node if args.use_gat else None,
#             base_kwargs={'recurrent': args.recurrent_policy})
#         if model_load_path:
#             mf = os.path.join(model_load_path, f'agent{i}.pth')
#             if os.path.exists(mf):
#                 ac.load_state_dict(torch.load(mf, map_location=device))
#                 print(f"  Loaded actor {i}")
#         ac.to(device)
#         actor_critics.append(ac)
#         r = RolloutStorage(args.num_pretrain_steps, observation_spaces[i].shape,
#             action_spaces[i], ac.recurrent_hidden_state_size, num_node)
#         rollouts.append(r)
#         rollouts[i].obs[0].copy_(obses[i])
#         rollouts[i].to(device)

#     # === Centralized critic ===
#     centralized_critic = CentralizedCritic(
#         global_state_dim=global_state_dim,
#         hidden_size=args.critic_hidden_size,
#         num_layers=args.critic_num_layers,
#         use_feature_normalization=args.use_feature_normalization,
#         use_orthogonal=True, use_popart=args.use_popart, device=device)
#     if model_load_path:
#         cf = os.path.join(model_load_path, 'critic.pth')
#         if os.path.exists(cf):
#             centralized_critic.load_state_dict(torch.load(cf, map_location=device))
#             print(f"  Loaded critic")

#     # === MAPPO agent ===
#     mappo_agent = MAPPO_CTDE(
#         actor_critics=actor_critics, centralized_critic=centralized_critic,
#         clip_param=args.clip_param, ppo_epoch=args.ppo_epoch,
#         num_mini_batch=args.num_mini_batch,
#         value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef,
#         actor_lr=args.actor_lr, critic_lr=args.critic_lr, eps=args.eps,
#         max_grad_norm=args.max_grad_norm, use_huber_loss=args.use_huber_loss,
#         huber_delta=args.huber_delta, use_clipped_value_loss=args.use_clipped_value_loss,
#         use_popart=args.use_popart, use_valuenorm=args.use_valuenorm,
#         use_linear_lr_decay=args.use_linear_lr_decay, device=device)

#     print(f"Actor LR={args.actor_lr}, Critic LR={args.critic_lr}, GAT={args.use_gat}")

#     # ==================== PRE-TRAINING ====================
#     print(f"\nPRE-TRAINING: {args.num_pretrain_epochs} epochs x {args.num_pretrain_steps} steps")
#     mappo_agent.prep_training()
#     pretrain_start = time.time()

#     for epoch in range(args.num_pretrain_epochs):
#         global_states_buffer = []
#         for step in range(args.num_pretrain_steps):
#             with torch.no_grad():
#                 values=[None]*num_agent; actions=[None]*num_agent
#                 action_log_probs=[None]*num_agent
#                 recurrent_hidden_states=[None]*num_agent
#                 condition_states=[None]*num_agent
#                 curr_path=[0]*num_node; agents_flag=[0]*num_agent
#                 curr_agent, path = envs.first_agent()
#                 gs = construct_global_state(
#                     envs._link_capa, envs._link_usage, envs._link_losses,
#                     envs._request.s, envs._request.t,
#                     envs._request.rtype, envs._request.demand,
#                     num_node, num_type, device)
#                 global_states_buffer.append(gs)
#                 cv = mappo_agent.get_values(gs.unsqueeze(0))
#                 while curr_agent is not None and agents_flag[curr_agent] != 1:
#                     for k in path: curr_path[k] = 1
#                     agents_flag[curr_agent] = 1
#                     cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
#                     _, a, alp, rhs = actor_critics[curr_agent].act(
#                         rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
#                         rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
#                         cs.unsqueeze(0))
#                     values[curr_agent]=cv; actions[curr_agent]=a
#                     action_log_probs[curr_agent]=alp
#                     recurrent_hidden_states[curr_agent]=rhs; condition_states[curr_agent]=cs
#                     curr_agent, path = envs.next_agent(curr_agent, a)
#                 cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
#                 for k in range(num_agent):
#                     if agents_flag[k] != 1:
#                         _, a, alp, rhs = actor_critics[k].act(
#                             rollouts[k].obs[rollouts[k].step].unsqueeze(0),
#                             rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
#                             cs.unsqueeze(0))
#                         values[k]=cv; actions[k]=a; action_log_probs[k]=alp
#                         recurrent_hidden_states[k]=rhs; condition_states[k]=cs
#             gfactors = [0.]*num_agent
#             obses, rewards, path, dd, dm, cf, rt, grwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
#             print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
#             print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)
#             for k in range(num_agent):
#                 masks = torch.tensor([1.])
#                 rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
#                     condition_states[k], actions[k].squeeze(0),
#                     action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)
#         for k in range(num_agent):
#             with torch.no_grad():
#                 fgs = construct_global_state(envs._link_capa, envs._link_usage, envs._link_losses,
#                     envs._request.s, envs._request.t, envs._request.rtype, envs._request.demand,
#                     num_node, num_type, device)
#                 nv = mappo_agent.get_values(fgs.unsqueeze(0)).detach()
#                 rollouts[k].compute_returns(nv, args.use_gae, args.gamma, args.gae_lambda)
#         gst = torch.stack(global_states_buffer)
#         vl, al, ent, ratio = mappo_agent.update(rollouts, gst, torch.ones(num_agent))
#         for k in range(num_agent):
#             rollouts[k].after_update()
#         if epoch % 5 == 0:
#             print(f"  Epoch {epoch:3d}/{args.num_pretrain_epochs} | V:{vl:.4f} A:{al:.4f} E:{ent:.4f}")

#     print(f"Pre-training done in {(time.time()-pretrain_start)/60:.1f} min\n")

#     # ==================== ONLINE TRAINING ====================
#     print(f"ONLINE TRAINING: {args.num_env_steps:,} steps | Scenario: {scenario}")
#     request, obses = envs.reset()

#     # Re-apply load after reset (reset may re-init request_times)
#     if args.load == 'heavy':
#         envs._request_times = [[50], [50], [50], [50]]
#     elif args.load == 'light':
#         envs._request_times = [[10], [10], [10], [10]]

#     rollouts = []
#     for i in range(num_agent):
#         r = RolloutStorage(args.num_steps, observation_spaces[i].shape,
#             action_spaces[i], actor_critics[i].recurrent_hidden_state_size, num_node)
#         rollouts.append(r)
#         rollouts[i].obs[0].copy_(obses[i])
#         rollouts[i].to(device)

#     mappo_agent.reset_optimizers()
#     rwd_buf = deque(maxlen=200)
#     delay_bufs = {t: deque(maxlen=200) for t in range(num_type)}
#     thr_bufs = {t: deque(maxlen=200) for t in range(num_type)}
#     loss_bufs = {t: deque(maxlen=200) for t in range(num_type)}
#     start_time = time.time()
#     global_states_buffer = []

#     # Event tracking
#     event_idx = 0
#     events = sc['events']

#     # Initialize last losses for TensorBoard (FIXED: avoids NameError)
#     last_vl, last_al, last_ent = 0.0, 0.0, 0.0

#     for step in range(args.num_env_steps):
#         # === Check for scheduled scenario events ===
#         while event_idx < len(events) and step >= events[event_idx]['timestep']:
#             ev = events[event_idx]
#             desc = ev.get('description', ev['action'])
#             print(f"\n{'='*50}")
#             print(f">>> Step {step:,}: EVENT — {desc}")
#             print(f"{'='*50}")
#             apply_env_change(envs, ev['action'], num_node)
#             log_events_file.write(f"{step},{ev['action']},{desc}\n")
#             log_events_file.flush()
#             if writer:
#                 writer.add_text('events', f"Step {step}: {desc}", step)
#                 writer.flush()
#             event_idx += 1

#         if args.use_linear_lr_decay:
#             mappo_agent.lr_decay(step, args.num_env_steps)

#         with torch.no_grad():
#             values=[None]*num_agent; actions=[None]*num_agent
#             action_log_probs=[None]*num_agent
#             recurrent_hidden_states=[None]*num_agent
#             condition_states=[None]*num_agent
#             curr_path=[0]*num_node; agents_flag=[0]*num_agent
#             curr_agent, path = envs.first_agent()

#             gs = construct_global_state(
#                 envs._link_capa, envs._link_usage, envs._link_losses,
#                 envs._request.s, envs._request.t,
#                 envs._request.rtype, envs._request.demand,
#                 num_node, num_type, device)
#             global_states_buffer.append(gs)
#             cv = mappo_agent.get_values(gs.unsqueeze(0))

#             while curr_agent is not None and agents_flag[curr_agent] != 1:
#                 for k in path: curr_path[k] = 1
#                 agents_flag[curr_agent] = 1
#                 cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
#                 _, a, alp, rhs = actor_critics[curr_agent].act(
#                     rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
#                     rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
#                     cs.unsqueeze(0))
#                 values[curr_agent]=cv; actions[curr_agent]=a
#                 action_log_probs[curr_agent]=alp
#                 recurrent_hidden_states[curr_agent]=rhs; condition_states[curr_agent]=cs
#                 curr_agent, path = envs.next_agent(curr_agent, a)

#             cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
#             for k in range(num_agent):
#                 if agents_flag[k] != 1:
#                     _, a, alp, rhs = actor_critics[k].act(
#                         rollouts[k].obs[rollouts[k].step].unsqueeze(0),
#                         rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
#                         cs.unsqueeze(0))
#                     values[k]=cv; actions[k]=a; action_log_probs[k]=alp
#                     recurrent_hidden_states[k]=rhs; condition_states[k]=cs

#         gfactors = [1.]*num_agent
#         obses, rewards, path, dd, dm, cf, rt, grwd, delay, thr, lr = envs.step(actions, gfactors)
#         print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
#         print(delay, file=log_delay_files[rt]); print(thr, file=log_throughput_files[rt])
#         print(lr, file=log_loss_files[rt])
#         print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)

#         rwd_buf.append(grwd)
#         delay_bufs[rt].append(delay); thr_bufs[rt].append(thr); loss_bufs[rt].append(lr)

#         # === TensorBoard — FIXED: write + flush every 200 steps ===
#         if writer and step % 200 == 0 and step > 0:
#             writer.add_scalar('reward/global', np.mean(rwd_buf), step)
#             for t in range(num_type):
#                 if delay_bufs[t]: writer.add_scalar(f'delay/type{t}', np.mean(delay_bufs[t]), step)
#                 if thr_bufs[t]: writer.add_scalar(f'throughput/type{t}', np.mean(thr_bufs[t]), step)
#                 if loss_bufs[t]: writer.add_scalar(f'loss/type{t}', np.mean(loss_bufs[t]), step)
#             writer.add_scalar('train/value_loss', last_vl, step)
#             writer.add_scalar('train/action_loss', last_al, step)
#             writer.add_scalar('train/entropy', last_ent, step)
#             writer.flush()  # <-- KEY FIX: force write to disk

#         # Console
#         if step % 5000 == 0 and step > 0:
#             elapsed = time.time() - start_time
#             sps = step / elapsed
#             avg_rwd = np.mean(rwd_buf) if rwd_buf else 0
#             print(f"Step {step:6d}/{args.num_env_steps} | Reward: {avg_rwd:.3f} | "
#                   f"Speed: {sps:.0f} steps/s | ETA: {(args.num_env_steps-step)/max(sps,1)/60:.1f}min")

#         agent_masks_tensor = torch.tensor(agents_flag, dtype=torch.float32)
#         for k in range(num_agent):
#             masks = torch.tensor([1.] if agents_flag[k]==1 else [0.])
#             rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
#                 condition_states[k], actions[k].squeeze(0),
#                 action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)
#             if rollouts[k].step == 0:
#                 with torch.no_grad():
#                     fgs = construct_global_state(envs._link_capa, envs._link_usage, envs._link_losses,
#                         envs._request.s, envs._request.t, envs._request.rtype, envs._request.demand,
#                         num_node, num_type, device)
#                     nv = mappo_agent.get_values(fgs.unsqueeze(0)).detach()
#                     rollouts[k].compute_returns(nv, args.use_gae, args.gamma, args.gae_lambda)

#         # === Update when rollout buffer is full ===
#         if rollouts[0].step == 0:
#             gst = torch.stack(global_states_buffer)
#             last_vl, last_al, last_ent, ratio = mappo_agent.update(
#                 rollouts, gst, agent_masks_tensor)
#             global_states_buffer = []
#             for k in range(num_agent):
#                 rollouts[k].after_update()

#         # Checkpoint
#         if step % ckpt_step == 0 and step > 0 and model_save_path:
#             os.makedirs(model_save_path, exist_ok=True)
#             for i in range(num_agent):
#                 torch.save(actor_critics[i].state_dict(), f"{model_save_path}/agent{i}.pth")
#             torch.save(centralized_critic.state_dict(), f"{model_save_path}/critic.pth")

#     # Final save
#     if model_save_path:
#         os.makedirs(model_save_path, exist_ok=True)
#         for i in range(num_agent):
#             torch.save(actor_critics[i].state_dict(), f"{model_save_path}/agent{i}.pth")
#         torch.save(centralized_critic.state_dict(), f"{model_save_path}/critic.pth")
#         print(f"Model saved to {model_save_path}")

#     for f in log_dist_files + log_demand_files + log_delay_files + log_throughput_files + log_loss_files:
#         f.close()
#     log_globalrwd_file.close(); log_circle_file.close(); log_events_file.close()
#     if writer: writer.close()

#     total_time = time.time() - start_time
#     print(f"\nDone in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")


# if __name__ == "__main__":
#     main()