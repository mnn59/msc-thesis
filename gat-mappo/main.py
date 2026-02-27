"""
GAT-MAPPO-DRL-OR Training Script
=================================
Uses original DRL-OR simenv.py WITHOUT modification.
Supports --algo mappo (GAT-MAPPO) and --algo ppo (original baseline).
Includes TensorBoard logging for live monitoring.
"""

import copy, glob, os, time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.ppo import PPO
import utils
from arguments import get_args
from model import Policy
from storage import RolloutStorage

from net_env.simenv import NetEnv
from net_env.env_utils import extract_adjacency_matrix


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    ckpt_step = args.ckpt_steps
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # === TensorBoard setup ===
    writer = None
    if args.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(log_dir, 'tb')
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(tb_dir)
            print(f"TensorBoard logging to: {tb_dir}")
            print(f"  Run: tensorboard --logdir {tb_dir}")
        except ImportError:
            print("TensorBoard not installed. Run: pip install tensorboard")
            print("Continuing without TensorBoard...")
            writer = None

    # === Environment setup (original DRL-OR simenv, unchanged) ===
    envs = NetEnv(args)
    num_agent, num_node, observation_spaces, action_spaces, num_type = \
        envs.setup(args.env_name, args.demand_matrix)
    request, obses = envs.reset()
    print("observation_spaces", observation_spaces)
    print(f"algo={args.algo}, use_gat={args.use_gat}, lr={args.lr}, "
          f"ppo_epoch={args.ppo_epoch}, mini_batch={args.num_mini_batch}, "
          f"clip={args.clip_param}, steps={args.num_steps}, "
          f"grad_norm={args.max_grad_norm}")

    # === Extract adjacency matrix from simenv for GAT ===
    adj_matrix = None
    if args.use_gat:
        adj_matrix, edge_list = extract_adjacency_matrix(envs, num_node)
        adj_matrix = adj_matrix.to(device)
        print(f"GAT enabled: {num_node} nodes, {len(edge_list)} directed edges")

    # === Open log files (same as DRL-OR) ===
    log_dist_files, log_demand_files, log_delay_files = [], [], []
    log_throughput_files, log_loss_files = [], []
    for i in range(num_type):
        log_dist_files.append(open("%s/dist_type%d.log" % (log_dir, i), "w", 1))
        log_demand_files.append(open("%s/demand_type%d.log" % (log_dir, i), "w", 1))
        log_delay_files.append(open("%s/delay_type%d.log" % (log_dir, i), "w", 1))
        log_throughput_files.append(open("%s/throughput_type%d.log" % (log_dir, i), "w", 1))
        log_loss_files.append(open("%s/loss_type%d.log" % (log_dir, i), "w", 1))
    log_globalrwd_file = open("%s/globalrwd.log" % log_dir, "w", 1)
    log_circle_file = open("%s/circle.log" % log_dir, "w", 1)

    # === Build models ===
    actor_critics, agents, rollouts = [], [], []
    for i in range(num_agent):
        actor_critic = Policy(
            observation_spaces[i].shape, action_spaces[i], num_node,
            node_num=num_node, type_num=num_type,
            adj_matrix=adj_matrix, num_nodes=num_node if args.use_gat else None,
            base_kwargs={'recurrent': args.recurrent_policy})

        if model_load_path is not None:
            actor_critic.load_state_dict(torch.load("%s/agent%d.pth" % (model_load_path, i)))
        actor_critic.to(device)
        actor_critics.append(actor_critic)

        agent = PPO(actor_critic, args.clip_param, args.ppo_epoch,
                    args.num_mini_batch, args.value_loss_coef,
                    args.entropy_coef, lr=args.lr, eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
        agents.append(agent)

        rollouts.append(RolloutStorage(args.num_pretrain_steps, observation_spaces[i].shape,
                        action_spaces[i], actor_critic.recurrent_hidden_state_size, num_node))
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)

    # === Pretraining ===
    # Only pretrain from scratch. Skip if loading a saved model.
    if model_load_path:
        print(f"Skipping pre-training (loaded model from {model_load_path})")
    else:
        for ep in range(args.num_pretrain_epochs):
            for _ in range(args.num_pretrain_steps):
                with torch.no_grad():
                    values = [None]*num_agent; actions = [None]*num_agent
                    action_log_probs = [None]*num_agent
                    recurrent_hidden_states = [None]*num_agent
                    condition_states = [None]*num_agent

                    curr_path = [0]*num_node; agents_flag = [0]*num_agent
                    curr_agent, path = envs.first_agent()
                    while curr_agent is not None and agents_flag[curr_agent] != 1:
                        for k in path: curr_path[k] = 1
                        agents_flag[curr_agent] = 1
                        cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
                        v, a, alp, rhs = actor_critics[curr_agent].act(
                            rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                            rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
                            cs.unsqueeze(0))
                        values[curr_agent]=v; actions[curr_agent]=a
                        action_log_probs[curr_agent]=alp
                        recurrent_hidden_states[curr_agent]=rhs; condition_states[curr_agent]=cs
                        curr_agent, path = envs.next_agent(curr_agent, a)

                    cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
                    for k in range(num_agent):
                        if agents_flag[k] != 1:
                            v, a, alp, rhs = actor_critics[k].act(
                                rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                                rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
                                cs.unsqueeze(0))
                            values[k]=v; actions[k]=a; action_log_probs[k]=alp
                            recurrent_hidden_states[k]=rhs; condition_states[k]=cs

                gfactors = [1.]*num_agent
                obses, rewards, path, dd, dm, cf, rt, grwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
                print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
                print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)

                for k in range(num_agent):
                    masks = torch.tensor([1.])
                    rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
                        condition_states[k], actions[k].squeeze(0),
                        action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

            for k in range(num_agent):
                if args.use_linear_lr_decay:
                    utils.update_linear_schedule(agents[k].optimizer, ep, args.num_pretrain_epochs, args.lr*100)
                with torch.no_grad():
                    cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
                    nv = actor_critics[k].get_value(
                        rollouts[k].obs[-1].unsqueeze(0),
                        rollouts[k].recurrent_hidden_states[-1].unsqueeze(0),
                        cs.unsqueeze(0)).detach()
                    rollouts[k].compute_returns(nv, args.use_gae, args.gamma, args.gae_lambda)
                agents[k].update(rollouts[k])

    # === Online training ===
    for i in range(num_agent):
        rollouts[i] = RolloutStorage(args.num_steps, observation_spaces[i].shape,
                        action_spaces[i], actor_critics[i].recurrent_hidden_state_size, num_node)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)

    # Running averages for TensorBoard
    rwd_buf = deque(maxlen=200)
    delay_bufs = {t: deque(maxlen=200) for t in range(num_type)}
    thr_bufs = {t: deque(maxlen=200) for t in range(num_type)}
    loss_bufs = {t: deque(maxlen=200) for t in range(num_type)}

    # Initialize training loss vars for TensorBoard (before first update)
    value_loss = 0.0; action_loss = 0.0; dist_entropy = 0.0

    start_time = time.time()
    for j in range(args.num_env_steps):
        with torch.no_grad():
            values = [None]*num_agent; actions = [None]*num_agent
            action_log_probs = [None]*num_agent
            recurrent_hidden_states = [None]*num_agent
            condition_states = [None]*num_agent

            curr_path = [0]*num_node; agents_flag = [0]*num_agent
            curr_agent, path = envs.first_agent()
            while curr_agent is not None and agents_flag[curr_agent] != 1:
                for k in path: curr_path[k] = 1
                agents_flag[curr_agent] = 1
                cs = torch.tensor(curr_path, dtype=torch.float32).to(device)
                v, a, alp, rhs = actor_critics[curr_agent].act(
                    rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                    rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
                    cs.unsqueeze(0))
                values[curr_agent]=v; actions[curr_agent]=a
                action_log_probs[curr_agent]=alp
                recurrent_hidden_states[curr_agent]=rhs; condition_states[curr_agent]=cs
                curr_agent, path = envs.next_agent(curr_agent, a)

            cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
            for k in range(num_agent):
                if agents_flag[k] != 1:
                    v, a, alp, rhs = actor_critics[k].act(
                        rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                        rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
                        cs.unsqueeze(0))
                    values[k]=v; actions[k]=a; action_log_probs[k]=alp
                    recurrent_hidden_states[k]=rhs; condition_states[k]=cs

        gfactors = [1.]*num_agent
        obses, rewards, path, dd, dm, cf, rt, grwd, delay, thr, lr = envs.step(actions, gfactors)
        print(dd, file=log_dist_files[rt]); print(dm, file=log_demand_files[rt])
        print(delay, file=log_delay_files[rt]); print(thr, file=log_throughput_files[rt])
        print(lr, file=log_loss_files[rt])
        print(grwd, file=log_globalrwd_file); print(cf, file=log_circle_file)

        # === TensorBoard logging ===
        rwd_buf.append(grwd)
        delay_bufs[rt].append(delay)
        thr_bufs[rt].append(thr)
        loss_bufs[rt].append(lr)

        if writer and j % 200 == 0 and j > 0:
            writer.add_scalar('reward/global', np.mean(rwd_buf), j)
            for t in range(num_type):
                if len(delay_bufs[t]) > 0:
                    writer.add_scalar(f'delay/type{t}', np.mean(delay_bufs[t]), j)
                if len(thr_bufs[t]) > 0:
                    writer.add_scalar(f'throughput/type{t}', np.mean(thr_bufs[t]), j)
                if len(loss_bufs[t]) > 0:
                    writer.add_scalar(f'loss/type{t}', np.mean(loss_bufs[t]), j)
            writer.flush()

        # Console progress
        if j % 5000 == 0 and j > 0:
            elapsed = time.time() - start_time
            sps = j / elapsed
            avg_rwd = np.mean(rwd_buf) if rwd_buf else 0
            print(f"Step {j}/{args.num_env_steps} | "
                  f"Reward: {avg_rwd:.3f} | "
                  f"Speed: {sps:.0f} steps/s | "
                  f"ETA: {(args.num_env_steps-j)/max(sps,1)/60:.1f}min")

        for k in range(num_agent):
            masks = torch.tensor([1.]) if agents_flag[k]==1 else torch.tensor([0.])
            rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0),
                condition_states[k], actions[k].squeeze(0),
                action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

            if rollouts[k].step == 0:
                if args.use_linear_lr_decay:
                    utils.update_linear_schedule(agents[k].optimizer, j, args.num_env_steps, args.lr)
                if args.use_linear_clip_decay and hasattr(agents[k], 'clip_param'):
                    agents[k].clip_param = args.clip_param * (1 - j / float(args.num_env_steps))
                with torch.no_grad():
                    cs = torch.tensor([0]*num_node, dtype=torch.float32).to(device)
                    nv = actor_critics[k].get_value(
                        rollouts[k].obs[-1].unsqueeze(0),
                        rollouts[k].recurrent_hidden_states[-1].unsqueeze(0),
                        cs.unsqueeze(0)).detach()
                    rollouts[k].compute_returns(nv, args.use_gae, args.gamma, args.gae_lambda)
                value_loss, action_loss, dist_entropy = agents[k].update(rollouts[k])
                rollouts[k].after_update()

                # Log training losses to TensorBoard (once per rollout, from last agent)
                if k == num_agent - 1 and writer:
                    writer.add_scalar('train/value_loss', value_loss, j)
                    writer.add_scalar('train/action_loss', action_loss, j)
                    writer.add_scalar('train/entropy', dist_entropy, j)
                    writer.flush()

        if j % ckpt_step == 0 and model_save_path:
            os.makedirs(model_save_path, exist_ok=True)
            for i in range(num_agent):
                torch.save(actor_critics[i].state_dict(), "%s/agent%d.pth" % (model_save_path, i))

    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
        for i in range(num_agent):
            torch.save(actor_critics[i].state_dict(), "%s/agent%d.pth" % (model_save_path, i))

    if writer:
        writer.flush()
        writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()