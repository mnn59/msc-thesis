"""
GAT-MAPPO Model for DRL-OR (v2 - Hybrid Architecture)
======================================================
Pure PyTorch GAT (NO torch_geometric dependency).

v2 Architecture (Hybrid MLP+GAT):
  - MLP branch: processes raw obs exactly like DRL-OR (preserves all info)
  - GAT branch: processes link_usage+link_loss as per-node graph features
  - Actor: MLP_features + GAT_features + condition_state -> action
  - Critic (centralized): MLP_features + GAT_features (pooled) -> V(s)

vs DRL-OR-S: they put attention in the distribution layer (AttentionDist);
  we put GAT in shared feature extraction (JGAT-MAPPO approach).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from distributions import Categorical, MultiCategorical, MultiTypeCategorical
from utils import init


class GATLayer(nn.Module):
    """Graph Attention Layer. From JGAT-MAPPO Eq.(11)-(13)."""
    def __init__(self, in_features, out_features, num_heads=4, concat=True,
                 dropout=0.0, leaky_slope=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        nn.init.xavier_uniform_(self.W)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky_relu = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        B, N, F_in = x.shape
        H = self.num_heads
        adj = adj.float()
        Wh = torch.einsum('bnf,hfo->bhno', x, self.W)
        attn_src = torch.einsum('bhno,hol->bhn', Wh, self.a_src).unsqueeze(-1)
        attn_dst = torch.einsum('bhno,hol->bhn', Wh, self.a_dst).unsqueeze(-2)
        e = self.leaky_relu(attn_src + attn_dst)
        if adj.dim() == 2:
            mask = adj.unsqueeze(0).unsqueeze(0)
        else:
            mask = adj.unsqueeze(1)
        e = e.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, Wh)
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * self.out_features)
        else:
            out = out.mean(dim=1)
        return F.elu(out)


class SharedGAT(nn.Module):
    """2-layer GAT. Small and lightweight — supplements MLP."""
    def __init__(self, in_features, hidden_dim=16, num_heads=4):
        super(SharedGAT, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_dim, num_heads=num_heads, concat=True)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads=1, concat=True)
        self.output_dim = hidden_dim

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = self.gat2(x, adj)
        return x


class ObsToNodeFeatures(nn.Module):
    """
    Extracts STRUCTURED per-node features from DRL-OR's flat observation.
    
    DRL-OR state: [..., link_usage(N*N), link_loss(N*N), type(T), src(N), dst(N)]
    Per-node feature = [usage_row(N), loss_row(N)] -> dim = 2*N
    """
    def __init__(self, obs_dim, num_nodes, num_type, adj_matrix):
        super(ObsToNodeFeatures, self).__init__()
        self.num_nodes = num_nodes
        self.register_buffer('adj', adj_matrix.float())
        tail_size = num_type + 2 * num_nodes
        self.link_loss_start = obs_dim - tail_size - num_nodes * num_nodes
        self.link_usage_start = self.link_loss_start - num_nodes * num_nodes
        self.node_feat_dim = 2 * num_nodes
        
    def forward(self, obs):
        leading = obs.shape[:-1]
        N = self.num_nodes
        usage = obs[..., self.link_usage_start:self.link_usage_start + N*N]
        loss = obs[..., self.link_loss_start:self.link_loss_start + N*N]
        usage_mat = usage.reshape(*leading, N, N)
        loss_mat = loss.reshape(*leading, N, N)
        node_feats = torch.cat([usage_mat, loss_mat], dim=-1)
        return node_feats, self.adj


class Policy(nn.Module):
    """
    Hybrid MLP+GAT Policy for DRL-OR with MAPPO CTDE.
    
    MLP branch: raw obs -> features (exactly like DRL-OR MLPBase)
    GAT branch: link_usage+link_loss per-node -> graph attention -> pooled features
    Actor: MLP_feat + GAT_pool + condition -> action distribution
    Critic: MLP_feat + GAT_pool -> V(s) [no condition, same as DRL-OR]
    """
    def __init__(self, obs_shape, action_space, condition_space,
                 node_num=None, type_num=None, adj_matrix=None, num_nodes=None,
                 base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        
        self._node_num = node_num
        self._type_num = type_num
        self._recurrent = base_kwargs.get('recurrent', False)
        
        obs_dim = obs_shape[0]
        hidden_size = 64
        gat_out_dim = 16
        
        use_gat = (adj_matrix is not None and num_nodes is not None)
        self._use_gat = use_gat
        
        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        if use_gat:
            gat_node_feat_dim = 2 * num_nodes
            self.obs2node = ObsToNodeFeatures(obs_dim, num_nodes, type_num or 0, adj_matrix)
            self.shared_gat = SharedGAT(gat_node_feat_dim, hidden_dim=gat_out_dim, num_heads=4)
            
            # MLP branch (same as DRL-OR MLPBase)
            self.actor_mlp = nn.Sequential(
                init_(nn.Linear(obs_dim + condition_space, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            )
            self.critic_mlp = nn.Sequential(
                init_(nn.Linear(obs_dim, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            )
            
            # Fusion: MLP(64) + GAT_pool -> hidden
            self.actor_fusion = nn.Sequential(
                init_(nn.Linear(hidden_size + gat_out_dim, hidden_size)), nn.Tanh(),
            )
            self.critic_fusion = nn.Sequential(
                init_(nn.Linear(hidden_size + gat_out_dim * 2, hidden_size)), nn.Tanh(),
            )
            self.critic_head = init_(nn.Linear(hidden_size, 1))
        else:
            # Original DRL-OR MLP
            self.actor_mlp = nn.Sequential(
                init_(nn.Linear(obs_dim + condition_space, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            )
            self.critic_mlp = nn.Sequential(
                init_(nn.Linear(obs_dim, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            )
            self.critic_head = init_(nn.Linear(hidden_size, 1))
        
        self._hidden_size = hidden_size
        
        if self._recurrent:
            self.gru = nn.GRU(obs_dim, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name: nn.init.constant_(param, 0)
                elif 'weight' in name: nn.init.orthogonal_(param)
        
        # Distribution (DRL-OR's MultiTypeCategorical - PRESERVED)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            if node_num is None and type_num is None:
                self.dist = Categorical(hidden_size, num_outputs)
            elif type_num is None and node_num is not None:
                self.dist = MultiCategorical(hidden_size, num_outputs, node_num)
            elif type_num is not None and node_num is not None:
                self.dist = MultiTypeCategorical(hidden_size, num_outputs, node_num, type_num)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self._recurrent
    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size if self._recurrent else 1

    def _forward_base(self, obs, rnn_hxs, cond):
        S, B, D = obs.shape
        
        if self.is_recurrent:
            obs_seq, rnn_hxs = self.gru(obs.reshape(S, B, D), rnn_hxs.unsqueeze(0))
            rnn_hxs = rnn_hxs.squeeze(0)
            obs_for_mlp = obs_seq
        else:
            obs_for_mlp = obs
        
        if self._use_gat:
            obs_flat = obs.reshape(S * B, D)
            node_feats, adj = self.obs2node(obs_flat)
            node_emb = self.shared_gat(node_feats, adj)
            gat_mean = node_emb.mean(dim=1).view(S, B, -1)
            gat_max = node_emb.max(dim=1).values.view(S, B, -1)
            gat_critic_pool = torch.cat([gat_mean, gat_max], dim=-1)
            
            actor_mlp_feat = self.actor_mlp(torch.cat([obs_for_mlp, cond], dim=-1))
            critic_mlp_feat = self.critic_mlp(obs_for_mlp)
            
            actor_feat = self.actor_fusion(torch.cat([actor_mlp_feat, gat_mean], dim=-1))
            critic_feat = self.critic_fusion(torch.cat([critic_mlp_feat, gat_critic_pool], dim=-1))
            value = self.critic_head(critic_feat)
        else:
            actor_feat = self.actor_mlp(torch.cat([obs_for_mlp, cond], dim=-1))
            critic_feat = self.critic_mlp(obs_for_mlp)
            value = self.critic_head(critic_feat)
        
        return actor_feat, value, rnn_hxs

    def _get_dist(self, actor_feat, obs):
        if self._node_num is not None:
            dst = obs[..., -self._node_num:]
            if self._type_num is not None:
                tp = obs[..., -self._node_num * 2 - self._type_num:-self._node_num * 2]
        if self._node_num is None:
            return self.dist(actor_feat)
        elif self._type_num is None:
            return self.dist(actor_feat, dst)
        else:
            return self.dist(actor_feat, dst, tp)

    def act(self, inputs, rnn_hxs, condition_state, deterministic=False):
        af, val, rnn_hxs = self._forward_base(
            inputs.unsqueeze(0), rnn_hxs, condition_state.unsqueeze(0))
        af = af.squeeze(0); val = val.squeeze(0)
        dist = self._get_dist(af, inputs)
        action = dist.probs.argmax(dim=-1, keepdim=True) if deterministic else dist.sample().unsqueeze(-1)
        log_prob = dist.log_prob(action.squeeze(-1))
        return val, action, log_prob.unsqueeze(-1), rnn_hxs

    def get_value(self, inputs, rnn_hxs, condition_state):
        _, val, _ = self._forward_base(
            inputs.unsqueeze(0), rnn_hxs, condition_state.unsqueeze(0))
        return val.squeeze(0)

    def evaluate_actions(self, inputs, rnn_hxs, condition_state, action):
        af, val, rnn_hxs = self._forward_base(inputs, rnn_hxs, condition_state)
        dist = self._get_dist(af, inputs)
        log_prob = dist.log_prob(action.squeeze(-1))
        entropy = dist.entropy().mean()
        return val, log_prob.unsqueeze(-1), entropy, rnn_hxs