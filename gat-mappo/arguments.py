"""
Arguments for DRL-OR training.
  get_args()       - PPO/IPPO baseline (original DRL-OR)
  get_mappo_args() - MAPPO-CTDE with centralized critic
"""
import argparse
import torch

def get_args():
    """PPO/IPPO baseline arguments (original DRL-OR defaults)."""
    parser = argparse.ArgumentParser(description='GAT-PPO-DRL-OR')
    parser.add_argument('--algo', default='ppo', help='ppo')
    parser.add_argument('--lr', type=float, default=2.5e-5)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--use-gae', action='store_true', default=False)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda-deterministic', action='store_true', default=False)
    parser.add_argument('--num-steps', type=int, default=512)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--num-mini-batch', type=int, default=32)
    parser.add_argument('--clip-param', type=float, default=0.1)
    parser.add_argument('--num-env-steps', type=int, default=300000)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False)
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False)
    parser.add_argument('--log-dir', default='./log/')
    parser.add_argument('--model-save-path', default=None)
    parser.add_argument('--model-load-path', default=None)
    parser.add_argument('--ckpt-steps', type=int, default=10000)
    parser.add_argument('--recurrent-policy', action='store_true', default=False)
    parser.add_argument('--env-name', default='Abi')
    parser.add_argument('--demand-matrix', default='Abi_500.txt')
    parser.add_argument('--num-pretrain-epochs', type=int, default=30)
    parser.add_argument('--num-pretrain-steps', type=int, default=128)
    parser.add_argument('--use-gat', action='store_true', default=False)
    parser.add_argument('--no-gat', action='store_true', default=False)
    parser.add_argument('--use-tensorboard', action='store_true', default=True)
    parser.add_argument('--no-tensorboard', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.no_gat: args.use_gat = False
    if args.no_tensorboard: args.use_tensorboard = False
    return args

def get_mappo_args():
    """MAPPO-CTDE arguments (from drl-or-fake repo, proven to work)."""
    parser = argparse.ArgumentParser(description='MAPPO-CTDE for DRL-OR')
    # MAPPO hyperparameters (from MAPPO paper Table 7)
    parser.add_argument('--actor-lr', type=float, default=5e-4)
    parser.add_argument('--critic-lr', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--use-gae', action='store_true', default=True)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=1.0)
    parser.add_argument('--max-grad-norm', type=float, default=10.0)
    parser.add_argument('--ppo-epoch', type=int, default=15)
    parser.add_argument('--num-mini-batch', type=int, default=1)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--huber-delta', type=float, default=10.0)
    # CTDE critic
    parser.add_argument('--critic-hidden-size', type=int, default=64)
    parser.add_argument('--critic-num-layers', type=int, default=2)
    parser.add_argument('--use-feature-normalization', action='store_true', default=True)
    # MAPPO features
    parser.add_argument('--use-popart', action='store_true', default=True)
    parser.add_argument('--no-popart', action='store_true', default=False)
    parser.add_argument('--use-valuenorm', action='store_true', default=False)
    parser.add_argument('--use-huber-loss', action='store_true', default=True)
    parser.add_argument('--use-clipped-value-loss', action='store_true', default=True)
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=True)
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False)
    parser.add_argument('--recurrent-policy', action='store_true', default=False)
    # Training
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda-deterministic', action='store_true', default=False)
    parser.add_argument('--num-steps', type=int, default=512)
    parser.add_argument('--num-pretrain-epochs', type=int, default=30)
    parser.add_argument('--num-pretrain-steps', type=int, default=128)
    parser.add_argument('--ckpt-steps', type=int, default=10000)
    parser.add_argument('--num-env-steps', type=int, default=300000)
    # Environment
    parser.add_argument('--env-name', default='Abi')
    parser.add_argument('--demand-matrix', default='Abi_500.txt')
    # Logging
    parser.add_argument('--log-dir', default='./log/mappo_initialization')
    parser.add_argument('--model-load-path', default=None)
    parser.add_argument('--model-save-path', default=None)
    # Device
    parser.add_argument('--no-cuda', action='store_true', default=False)
    # GAT
    parser.add_argument('--use-gat', action='store_true', default=True)
    parser.add_argument('--no-gat', action='store_true', default=False)
    # TensorBoard
    parser.add_argument('--use-tensorboard', action='store_true', default=True)
    parser.add_argument('--no-tensorboard', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.no_gat: args.use_gat = False
    if args.no_popart: args.use_popart = False
    if args.no_tensorboard: args.use_tensorboard = False
    return args











# """
# Arguments for DRL-OR training.
#   get_args()       - PPO/IPPO baseline (original DRL-OR)
#   get_mappo_args() - MAPPO-CTDE with centralized critic
# """
# import argparse
# import torch

# def get_args():
#     """PPO/IPPO baseline arguments (original DRL-OR defaults)."""
#     parser = argparse.ArgumentParser(description='GAT-PPO-DRL-OR')
#     parser.add_argument('--algo', default='ppo', help='ppo')
#     parser.add_argument('--lr', type=float, default=2.5e-5)
#     parser.add_argument('--eps', type=float, default=1e-5)
#     parser.add_argument('--alpha', type=float, default=0.99)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--use-gae', action='store_true', default=False)
#     parser.add_argument('--gae-lambda', type=float, default=0.95)
#     parser.add_argument('--entropy-coef', type=float, default=0.01)
#     parser.add_argument('--value-loss-coef', type=float, default=0.5)
#     parser.add_argument('--max-grad-norm', type=float, default=0.5)
#     parser.add_argument('--seed', type=int, default=1)
#     parser.add_argument('--cuda-deterministic', action='store_true', default=False)
#     parser.add_argument('--num-steps', type=int, default=512)
#     parser.add_argument('--ppo-epoch', type=int, default=4)
#     parser.add_argument('--num-mini-batch', type=int, default=32)
#     parser.add_argument('--clip-param', type=float, default=0.1)
#     parser.add_argument('--num-env-steps', type=int, default=300000)
#     parser.add_argument('--no-cuda', action='store_true', default=False)
#     parser.add_argument('--use-linear-lr-decay', action='store_true', default=False)
#     parser.add_argument('--use-linear-clip-decay', action='store_true', default=False)
#     parser.add_argument('--log-dir', default='./log/')
#     parser.add_argument('--model-save-path', default=None)
#     parser.add_argument('--model-load-path', default=None)
#     parser.add_argument('--ckpt-steps', type=int, default=10000)
#     parser.add_argument('--recurrent-policy', action='store_true', default=False)
#     parser.add_argument('--env-name', default='Abi')
#     parser.add_argument('--demand-matrix', default='Abi_500.txt')
#     parser.add_argument('--num-pretrain-epochs', type=int, default=30)
#     parser.add_argument('--num-pretrain-steps', type=int, default=128)
#     parser.add_argument('--use-gat', action='store_true', default=False)
#     parser.add_argument('--no-gat', action='store_true', default=False)
#     parser.add_argument('--use-tensorboard', action='store_true', default=True)
#     parser.add_argument('--no-tensorboard', action='store_true', default=False)
#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     if args.no_gat: args.use_gat = False
#     if args.no_tensorboard: args.use_tensorboard = False
#     return args

# def get_mappo_args():
#     """MAPPO-CTDE arguments (from drl-or-fake repo, proven to work)."""
#     parser = argparse.ArgumentParser(description='MAPPO-CTDE for DRL-OR')
#     # MAPPO hyperparameters (from MAPPO paper Table 7)
#     parser.add_argument('--actor-lr', type=float, default=5e-4)
#     parser.add_argument('--critic-lr', type=float, default=5e-4)
#     parser.add_argument('--lr', type=float, default=5e-4)
#     parser.add_argument('--eps', type=float, default=1e-5)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--use-gae', action='store_true', default=True)
#     parser.add_argument('--gae-lambda', type=float, default=0.95)
#     parser.add_argument('--entropy-coef', type=float, default=0.01)
#     parser.add_argument('--value-loss-coef', type=float, default=1.0)
#     parser.add_argument('--max-grad-norm', type=float, default=10.0)
#     parser.add_argument('--ppo-epoch', type=int, default=15)
#     parser.add_argument('--num-mini-batch', type=int, default=1)
#     parser.add_argument('--clip-param', type=float, default=0.2)
#     parser.add_argument('--huber-delta', type=float, default=10.0)
#     # CTDE critic
#     parser.add_argument('--critic-hidden-size', type=int, default=64)
#     parser.add_argument('--critic-num-layers', type=int, default=2)
#     parser.add_argument('--use-feature-normalization', action='store_true', default=True)
#     # MAPPO features
#     parser.add_argument('--use-popart', action='store_true', default=True)
#     parser.add_argument('--no-popart', action='store_true', default=False)
#     parser.add_argument('--use-valuenorm', action='store_true', default=False)
#     parser.add_argument('--use-huber-loss', action='store_true', default=True)
#     parser.add_argument('--use-clipped-value-loss', action='store_true', default=True)
#     parser.add_argument('--use-linear-lr-decay', action='store_true', default=True)
#     parser.add_argument('--use-linear-clip-decay', action='store_true', default=False)
#     parser.add_argument('--recurrent-policy', action='store_true', default=False)
#     # Training
#     parser.add_argument('--seed', type=int, default=1)
#     parser.add_argument('--cuda-deterministic', action='store_true', default=False)
#     parser.add_argument('--num-steps', type=int, default=512)
#     parser.add_argument('--num-pretrain-epochs', type=int, default=30)
#     parser.add_argument('--num-pretrain-steps', type=int, default=128)
#     parser.add_argument('--ckpt-steps', type=int, default=10000)
#     parser.add_argument('--num-env-steps', type=int, default=300000)
#     # Environment
#     parser.add_argument('--env-name', default='Abi')
#     parser.add_argument('--demand-matrix', default='Abi_500.txt')
#     parser.add_argument('--scenario', type=str, default='initialization',
#                         choices=['initialization', 'link_failure', 'traffic_change',
#                                  'cascading_failure', 'link_degradation'],
#                         help='training scenario')
#     parser.add_argument('--load', type=str, default='default',
#                         choices=['light', 'heavy', 'default'],
#                         help='traffic load: light(10), heavy(50), default(simenv)')
#     # Logging
#     parser.add_argument('--log-dir', default='./log/mappo_initialization')
#     parser.add_argument('--model-load-path', default=None)
#     parser.add_argument('--model-save-path', default=None)
#     # Device
#     parser.add_argument('--no-cuda', action='store_true', default=False)
#     # GAT
#     parser.add_argument('--use-gat', action='store_true', default=True)
#     parser.add_argument('--no-gat', action='store_true', default=False)
#     # TensorBoard
#     parser.add_argument('--use-tensorboard', action='store_true', default=True)
#     parser.add_argument('--no-tensorboard', action='store_true', default=False)
#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     if args.no_gat: args.use_gat = False
#     if args.no_popart: args.use_popart = False
#     if args.no_tensorboard: args.use_tensorboard = False
#     return args