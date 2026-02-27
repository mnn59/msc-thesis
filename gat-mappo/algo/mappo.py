"""
MAPPO-CTDE: Centralized Training with Decentralized Execution.

From drl-or-fake repo (proven to work) with minor cleanup.
Architecture:
  - Actors (Decentralized): Policy networks using LOCAL obs + condition_state
  - Critic (Centralized): Separate network using GLOBAL state (all link info)
  
Key features: PopArt, Huber loss, clipped value loss, separate actor/critic LR.
"""

import math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ==================== PopArt ====================
class PopArt(nn.Module):
    """PopArt value normalization — adjusts weights when stats change."""
    def __init__(self, input_shape, output_shape=1, norm_axes=1,
                 beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        super().__init__()
        self.beta = beta; self.epsilon = epsilon; self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape))
        self.bias = nn.Parameter(torch.Tensor(output_shape))
        self.register_buffer('stddev', torch.ones(output_shape))
        self.register_buffer('mean', torch.zeros(output_shape))
        self.register_buffer('mean_sq', torch.zeros(output_shape))
        self.register_buffer('debiasing_term', torch.tensor(0.0))
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_(); self.mean_sq.zero_(); self.debiasing_term.zero_()

    def forward(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        return F.linear(x.to(**self.tpdv), self.weight, self.bias)

    @torch.no_grad()
    def update(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        x = x.to(**self.tpdv).view(-1, 1)  # Ensure [batch, 1]
        old_mean, old_var = self.debiased_mean_var()
        old_std = torch.sqrt(old_var)
        batch_mean = x.mean(dim=0)
        batch_sq = (x ** 2).mean(dim=0)
        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))
        self.stddev.copy_((self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4))
        new_mean, new_var = self.debiased_mean_var()
        new_std = torch.sqrt(new_var)
        self.weight.data.copy_(self.weight * old_std.unsqueeze(1) / new_std.unsqueeze(1))
        self.bias.data.copy_((old_std * self.bias + old_mean - new_mean) / new_std)

    def debiased_mean_var(self):
        dm = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        dms = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        return dm, (dms - dm ** 2).clamp(min=1e-2)

    def normalize(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        x = x.to(**self.tpdv)
        m, v = self.debiased_mean_var()
        # m and v are [1], broadcast works with any x shape
        return (x - m.item()) / max(torch.sqrt(v).item(), 1e-6)

    def denormalize(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        x = x.to(**self.tpdv)
        m, v = self.debiased_mean_var()
        return x * max(torch.sqrt(v).item(), 1e-6) + m.item()


# ==================== ValueNorm ====================
class ValueNorm(nn.Module):
    """Simple running mean/std normalization (alternative to PopArt)."""
    def __init__(self, input_shape=1, norm_axes=1, beta=0.99999,
                 epsilon=1e-5, device=torch.device("cpu")):
        super().__init__()
        self.norm_axes = norm_axes; self.epsilon = epsilon; self.beta = beta
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.register_buffer('running_mean', torch.zeros(input_shape))
        self.register_buffer('running_mean_sq', torch.zeros(input_shape))
        self.register_buffer('debiasing_term', torch.tensor(0.0))
        self.to(device)

    def running_mean_var(self):
        dm = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        dms = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        return dm, (dms - dm ** 2).clamp(min=1e-2)

    @torch.no_grad()
    def update(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        x = x.to(**self.tpdv).view(-1, 1)  # Ensure [batch, 1]
        bm = x.mean(dim=0)
        bsm = (x ** 2).mean(dim=0)
        self.running_mean.mul_(self.beta).add_(bm * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(bsm * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

    def normalize(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        x = x.to(**self.tpdv)
        m, v = self.running_mean_var()
        return (x - m.item()) / max(torch.sqrt(v).item(), 1e-6)

    def denormalize(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        x = x.to(**self.tpdv)
        m, v = self.running_mean_var()
        return x * max(torch.sqrt(v).item(), 1e-6) + m.item()


# ==================== Centralized Critic ====================
class CentralizedCritic(nn.Module):
    """Separate centralized critic using GLOBAL state. Used only during training."""
    def __init__(self, global_state_dim, hidden_size=64, num_layers=2,
                 use_feature_normalization=True, use_orthogonal=True,
                 use_popart=True, device='cpu'):
        super().__init__()
        self._use_feature_normalization = use_feature_normalization
        self._use_popart = use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        if use_feature_normalization:
            self.feature_norm = nn.LayerNorm(global_state_dim)
        layers = []
        in_dim = global_state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)
        if use_popart:
            self.v_out = PopArt(hidden_size, 1, device=torch.device(device))
        else:
            self.v_out = nn.Linear(hidden_size, 1)
        if use_orthogonal:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            if not use_popart:
                nn.init.orthogonal_(self.v_out.weight, gain=1)
                nn.init.constant_(self.v_out.bias, 0)
        self.to(device)

    def forward(self, gs):
        if isinstance(gs, np.ndarray): gs = torch.from_numpy(gs)
        gs = gs.to(**self.tpdv)
        if self._use_feature_normalization: gs = self.feature_norm(gs)
        return self.v_out(self.mlp(gs))

    def get_value(self, gs):
        return self.forward(gs)


# ==================== Utilities ====================
def huber_loss(e, d=10.0):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

def mse_loss(e):
    return e ** 2 / 2

def construct_global_state(link_capa, link_usage, link_losses,
                           flow_src, flow_dst, flow_type, flow_demand,
                           num_node, num_type, device='cpu'):
    """Construct global state vector for centralized critic."""
    gs = []
    for i in range(num_node):
        for j in range(num_node):
            if link_capa[i][j] > 0:
                gs.append(max(0, min(1, (link_capa[i][j] - link_usage[i][j]) / link_capa[i][j])))
            else:
                gs.append(0)
    for i in range(num_node):
        for j in range(num_node):
            gs.append(link_losses[i][j] / 100.0)
    t_oh = [0]*num_type; t_oh[flow_type] = 1; gs.extend(t_oh)
    s_oh = [0]*num_node; s_oh[flow_src] = 1; gs.extend(s_oh)
    d_oh = [0]*num_node; d_oh[flow_dst] = 1; gs.extend(d_oh)
    gs.append(flow_demand / 2000.0)
    return torch.tensor(gs, dtype=torch.float32, device=device)

def get_global_state_dim(num_node, num_type):
    return num_node * num_node * 2 + num_type + num_node * 2 + 1


# ==================== MAPPO_CTDE ====================
class MAPPO_CTDE:
    """MAPPO with full CTDE: separate actors + centralized critic."""
    def __init__(self, actor_critics, centralized_critic,
                 clip_param=0.2, ppo_epoch=15, num_mini_batch=1,
                 value_loss_coef=1.0, entropy_coef=0.01,
                 actor_lr=5e-4, critic_lr=5e-4, eps=1e-5,
                 max_grad_norm=10.0, use_huber_loss=True, huber_delta=10.0,
                 use_clipped_value_loss=True, use_popart=True,
                 use_valuenorm=False, use_linear_lr_decay=True, device='cpu'):
        self.actors = actor_critics
        self.critic = centralized_critic
        self.num_agents = len(actor_critics)
        self.device = device
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr; self.critic_lr = critic_lr; self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.use_huber_loss = use_huber_loss; self.huber_delta = huber_delta
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_linear_lr_decay = use_linear_lr_decay
        self._use_popart = use_popart; self._use_valuenorm = use_valuenorm

        self.actor_optimizers = [optim.Adam(a.parameters(), lr=actor_lr, eps=eps) for a in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=eps)
        self.value_normalizer = ValueNorm(1, device=torch.device(device)) if (use_valuenorm and not use_popart) else None

    def reset_optimizers(self):
        self.actor_optimizers = [optim.Adam(a.parameters(), lr=self.actor_lr, eps=self.eps) for a in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.eps)

    def lr_decay(self, step, total_steps):
        if self.use_linear_lr_decay:
            frac = 1 - step / float(total_steps)
            for opt in self.actor_optimizers:
                for pg in opt.param_groups: pg['lr'] = self.actor_lr * frac
            for pg in self.critic_optimizer.param_groups: pg['lr'] = self.critic_lr * frac

    def get_values(self, gs):
        return self.critic.get_value(gs)

    def cal_value_loss(self, values, vpred_b, ret_b, masks_b):
        # Flatten to 1D for PopArt/ValueNorm (storage yields [batch,1,1])
        values = values.view(-1)
        vpred_b = vpred_b.view(-1)
        ret_b = ret_b.view(-1)
        masks_b = masks_b.view(-1)
        if self._use_popart and hasattr(self.critic.v_out, 'update'):
            self.critic.v_out.update(ret_b)
            ret_norm = self.critic.v_out.normalize(ret_b)
        elif self._use_valuenorm and self.value_normalizer is not None:
            self.value_normalizer.update(ret_b)
            ret_norm = self.value_normalizer.normalize(ret_b)
        else:
            ret_norm = ret_b
        if self.use_clipped_value_loss:
            vpc = vpred_b + (values - vpred_b).clamp(-self.clip_param, self.clip_param)
            ec = ret_norm - vpc; eo = ret_norm - values
            vlc = huber_loss(ec, self.huber_delta) if self.use_huber_loss else mse_loss(ec)
            vlo = huber_loss(eo, self.huber_delta) if self.use_huber_loss else mse_loss(eo)
            vl = torch.max(vlo, vlc)
        else:
            e = ret_norm - values
            vl = huber_loss(e, self.huber_delta) if self.use_huber_loss else mse_loss(e)
        return (vl * masks_b).sum() / masks_b.sum().clamp(min=1.0)

    def update(self, rollouts, global_states_batch, agent_masks=None):
        # Compute advantages
        all_adv = []
        for k in range(self.num_agents):
            r = rollouts[k]
            if self._use_popart and hasattr(self.critic.v_out, 'denormalize'):
                adv = r.returns[:-1] - self.critic.v_out.denormalize(r.value_preds[:-1])
            elif self._use_valuenorm and self.value_normalizer is not None:
                adv = r.returns[:-1] - self.value_normalizer.denormalize(r.value_preds[:-1])
            else:
                adv = r.returns[:-1] - r.value_preds[:-1]
            all_adv.append((adv - adv.mean()) / (adv.std() + 1e-5))

        tv, ta, te, tr, n = 0, 0, 0, 0, 0
        for _ in range(self.ppo_epoch):
            # Update actors
            for ai in range(self.num_agents):
                gen = (rollouts[ai].recurrent_generator(all_adv[ai], self.num_mini_batch)
                       if self.actors[ai].is_recurrent else
                       rollouts[ai].feed_forward_generator(all_adv[ai], self.num_mini_batch))
                for sample in gen:
                    obs_b, rnn_b, cond_b, act_b, vp_b, ret_b, m_b, olp_b, adv_b = sample
                    _, logp, ent, _ = self.actors[ai].evaluate_actions(obs_b, rnn_b, cond_b, act_b)
                    ratio = torch.exp(logp - olp_b)
                    s1 = ratio * adv_b
                    s2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * adv_b
                    al = -(torch.min(s1, s2) * m_b).sum() / m_b.sum().clamp(min=1.0)
                    self.actor_optimizers[ai].zero_grad()
                    (al - ent * self.entropy_coef).backward()
                    nn.utils.clip_grad_norm_(self.actors[ai].parameters(), self.max_grad_norm)
                    self.actor_optimizers[ai].step()
                    ta += al.item(); te += ent.item(); tr += ratio.mean().item()

            # Update critic
            gen = (rollouts[0].recurrent_generator(all_adv[0], self.num_mini_batch)
                   if self.actors[0].is_recurrent else
                   rollouts[0].feed_forward_generator(all_adv[0], self.num_mini_batch))
            for bi, sample in enumerate(gen):
                _, _, _, _, vp_b, ret_b, m_b, _, _ = sample
                bs = vp_b.shape[0]
                si = bi * bs; ei = min(si + bs, global_states_batch.shape[0])
                gs_b = global_states_batch[si:ei]
                vals = self.critic.get_value(gs_b)
                vals = vals.squeeze(-1) if vals.dim() > 1 else vals
                if vals.shape[0] != vp_b.shape[0]: vals = vals[:vp_b.shape[0]]
                vl = self.cal_value_loss(vals, vp_b, ret_b, m_b)
                self.critic_optimizer.zero_grad()
                (vl * self.value_loss_coef).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                tv += vl.item()
            n += 1
        n = max(n, 1)
        na = max(n * self.num_agents, 1)
        return tv/n, ta/na, te/na, tr/na

    def prep_training(self):
        for a in self.actors: a.train()
        self.critic.train()

    def prep_rollout(self):
        for a in self.actors: a.eval()
        self.critic.eval()













# """
# MAPPO-CTDE: Centralized Training with Decentralized Execution.

# From drl-or-fake repo (proven to work) with minor cleanup.
# Architecture:
#   - Actors (Decentralized): Policy networks using LOCAL obs + condition_state
#   - Critic (Centralized): Separate network using GLOBAL state (all link info)
  
# Key features: PopArt, Huber loss, clipped value loss, separate actor/critic LR.
# """

# import math, torch, numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


# # ==================== PopArt ====================
# class PopArt(nn.Module):
#     """PopArt value normalization — adjusts weights when stats change."""
#     def __init__(self, input_shape, output_shape=1, norm_axes=1,
#                  beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
#         super().__init__()
#         self.beta = beta; self.epsilon = epsilon; self.norm_axes = norm_axes
#         self.tpdv = dict(dtype=torch.float32, device=device)
#         self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape))
#         self.bias = nn.Parameter(torch.Tensor(output_shape))
#         self.register_buffer('stddev', torch.ones(output_shape))
#         self.register_buffer('mean', torch.zeros(output_shape))
#         self.register_buffer('mean_sq', torch.zeros(output_shape))
#         self.register_buffer('debiasing_term', torch.tensor(0.0))
#         self.reset_parameters()
#         self.to(device)

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.bias, -bound, bound)
#         self.mean.zero_(); self.mean_sq.zero_(); self.debiasing_term.zero_()

#     def forward(self, x):
#         if isinstance(x, np.ndarray): x = torch.from_numpy(x)
#         return F.linear(x.to(**self.tpdv), self.weight, self.bias)

#     @torch.no_grad()
#     def update(self, x):
#         if isinstance(x, np.ndarray): x = torch.from_numpy(x)
#         x = x.to(**self.tpdv).view(-1, 1)  # Ensure [batch, 1]
#         old_mean, old_var = self.debiased_mean_var()
#         old_std = torch.sqrt(old_var)
#         batch_mean = x.mean(dim=0)
#         batch_sq = (x ** 2).mean(dim=0)
#         self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
#         self.mean_sq.mul_(self.beta).add_(batch_sq * (1.0 - self.beta))
#         self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))
#         self.stddev.copy_((self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4))
#         new_mean, new_var = self.debiased_mean_var()
#         new_std = torch.sqrt(new_var)
#         self.weight.data.copy_(self.weight * old_std.unsqueeze(1) / new_std.unsqueeze(1))
#         self.bias.data.copy_((old_std * self.bias + old_mean - new_mean) / new_std)

#     def debiased_mean_var(self):
#         dm = self.mean / self.debiasing_term.clamp(min=self.epsilon)
#         dms = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
#         return dm, (dms - dm ** 2).clamp(min=1e-2)

#     def normalize(self, x):
#         if isinstance(x, np.ndarray): x = torch.from_numpy(x)
#         x = x.to(**self.tpdv)
#         m, v = self.debiased_mean_var()
#         # m and v are [1], broadcast works with any x shape
#         return (x - m.item()) / max(torch.sqrt(v).item(), 1e-6)

#     def denormalize(self, x):
#         if isinstance(x, np.ndarray): x = torch.from_numpy(x)
#         x = x.to(**self.tpdv)
#         m, v = self.debiased_mean_var()
#         return x * max(torch.sqrt(v).item(), 1e-6) + m.item()


# # ==================== ValueNorm ====================
# class ValueNorm(nn.Module):
#     """Simple running mean/std normalization (alternative to PopArt)."""
#     def __init__(self, input_shape=1, norm_axes=1, beta=0.99999,
#                  epsilon=1e-5, device=torch.device("cpu")):
#         super().__init__()
#         self.norm_axes = norm_axes; self.epsilon = epsilon; self.beta = beta
#         self.tpdv = dict(dtype=torch.float32, device=device)
#         self.register_buffer('running_mean', torch.zeros(input_shape))
#         self.register_buffer('running_mean_sq', torch.zeros(input_shape))
#         self.register_buffer('debiasing_term', torch.tensor(0.0))
#         self.to(device)

#     def running_mean_var(self):
#         dm = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
#         dms = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
#         return dm, (dms - dm ** 2).clamp(min=1e-2)

#     @torch.no_grad()
#     def update(self, x):
#         if isinstance(x, np.ndarray): x = torch.from_numpy(x)
#         x = x.to(**self.tpdv).view(-1, 1)  # Ensure [batch, 1]
#         bm = x.mean(dim=0)
#         bsm = (x ** 2).mean(dim=0)
#         self.running_mean.mul_(self.beta).add_(bm * (1.0 - self.beta))
#         self.running_mean_sq.mul_(self.beta).add_(bsm * (1.0 - self.beta))
#         self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

#     def normalize(self, x):
#         if isinstance(x, np.ndarray): x = torch.from_numpy(x)
#         x = x.to(**self.tpdv)
#         m, v = self.running_mean_var()
#         return (x - m.item()) / max(torch.sqrt(v).item(), 1e-6)

#     def denormalize(self, x):
#         if isinstance(x, np.ndarray): x = torch.from_numpy(x)
#         x = x.to(**self.tpdv)
#         m, v = self.running_mean_var()
#         return x * max(torch.sqrt(v).item(), 1e-6) + m.item()


# # ==================== Centralized Critic ====================
# class CentralizedCritic(nn.Module):
#     """Separate centralized critic using GLOBAL state. Used only during training."""
#     def __init__(self, global_state_dim, hidden_size=64, num_layers=2,
#                  use_feature_normalization=True, use_orthogonal=True,
#                  use_popart=True, device='cpu'):
#         super().__init__()
#         self._use_feature_normalization = use_feature_normalization
#         self._use_popart = use_popart
#         self.tpdv = dict(dtype=torch.float32, device=device)
#         if use_feature_normalization:
#             self.feature_norm = nn.LayerNorm(global_state_dim)
#         layers = []
#         in_dim = global_state_dim
#         for _ in range(num_layers):
#             layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
#             in_dim = hidden_size
#         self.mlp = nn.Sequential(*layers)
#         if use_popart:
#             self.v_out = PopArt(hidden_size, 1, device=torch.device(device))
#         else:
#             self.v_out = nn.Linear(hidden_size, 1)
#         if use_orthogonal:
#             for m in self.mlp.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
#                     nn.init.constant_(m.bias, 0)
#             if not use_popart:
#                 nn.init.orthogonal_(self.v_out.weight, gain=1)
#                 nn.init.constant_(self.v_out.bias, 0)
#         self.to(device)

#     def forward(self, gs):
#         if isinstance(gs, np.ndarray): gs = torch.from_numpy(gs)
#         gs = gs.to(**self.tpdv)
#         if self._use_feature_normalization: gs = self.feature_norm(gs)
#         return self.v_out(self.mlp(gs))

#     def get_value(self, gs):
#         return self.forward(gs)


# # ==================== Utilities ====================
# def huber_loss(e, d=10.0):
#     a = (abs(e) <= d).float()
#     b = (abs(e) > d).float()
#     return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

# def mse_loss(e):
#     return e ** 2 / 2

# def construct_global_state(link_capa, link_usage, link_losses,
#                            flow_src, flow_dst, flow_type, flow_demand,
#                            num_node, num_type, device='cpu'):
#     """Construct global state vector for centralized critic."""
#     gs = []
#     for i in range(num_node):
#         for j in range(num_node):
#             if link_capa[i][j] > 0:
#                 gs.append(max(0, min(1, (link_capa[i][j] - link_usage[i][j]) / link_capa[i][j])))
#             else:
#                 gs.append(0)
#     for i in range(num_node):
#         for j in range(num_node):
#             gs.append(link_losses[i][j] / 100.0)
#     t_oh = [0]*num_type; t_oh[flow_type] = 1; gs.extend(t_oh)
#     s_oh = [0]*num_node; s_oh[flow_src] = 1; gs.extend(s_oh)
#     d_oh = [0]*num_node; d_oh[flow_dst] = 1; gs.extend(d_oh)
#     gs.append(flow_demand / 2000.0)
#     return torch.tensor(gs, dtype=torch.float32, device=device)

# def get_global_state_dim(num_node, num_type):
#     return num_node * num_node * 2 + num_type + num_node * 2 + 1


# # ==================== MAPPO_CTDE ====================
# class MAPPO_CTDE:
#     """MAPPO with full CTDE: separate actors + centralized critic."""
#     def __init__(self, actor_critics, centralized_critic,
#                  clip_param=0.2, ppo_epoch=15, num_mini_batch=1,
#                  value_loss_coef=1.0, entropy_coef=0.01,
#                  actor_lr=5e-4, critic_lr=5e-4, eps=1e-5,
#                  max_grad_norm=10.0, use_huber_loss=True, huber_delta=10.0,
#                  use_clipped_value_loss=True, use_popart=True,
#                  use_valuenorm=False, use_linear_lr_decay=True, device='cpu'):
#         self.actors = actor_critics
#         self.critic = centralized_critic
#         self.num_agents = len(actor_critics)
#         self.device = device
#         self.clip_param = clip_param
#         self.ppo_epoch = ppo_epoch
#         self.num_mini_batch = num_mini_batch
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#         self.actor_lr = actor_lr; self.critic_lr = critic_lr; self.eps = eps
#         self.max_grad_norm = max_grad_norm
#         self.use_huber_loss = use_huber_loss; self.huber_delta = huber_delta
#         self.use_clipped_value_loss = use_clipped_value_loss
#         self.use_linear_lr_decay = use_linear_lr_decay
#         self._use_popart = use_popart; self._use_valuenorm = use_valuenorm

#         self.actor_optimizers = [optim.Adam(a.parameters(), lr=actor_lr, eps=eps) for a in self.actors]
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=eps)
#         self.value_normalizer = ValueNorm(1, device=torch.device(device)) if (use_valuenorm and not use_popart) else None

#     def reset_optimizers(self):
#         self.actor_optimizers = [optim.Adam(a.parameters(), lr=self.actor_lr, eps=self.eps) for a in self.actors]
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.eps)

#     def lr_decay(self, step, total_steps):
#         if self.use_linear_lr_decay:
#             frac = 1 - step / float(total_steps)
#             for opt in self.actor_optimizers:
#                 for pg in opt.param_groups: pg['lr'] = self.actor_lr * frac
#             for pg in self.critic_optimizer.param_groups: pg['lr'] = self.critic_lr * frac

#     def get_values(self, gs):
#         return self.critic.get_value(gs)

#     def cal_value_loss(self, values, vpred_b, ret_b, masks_b):
#         # Flatten to 1D for PopArt/ValueNorm (storage yields [batch,1,1])
#         values = values.view(-1)
#         vpred_b = vpred_b.view(-1)
#         ret_b = ret_b.view(-1)
#         masks_b = masks_b.view(-1)
#         if self._use_popart and hasattr(self.critic.v_out, 'update'):
#             self.critic.v_out.update(ret_b)
#             ret_norm = self.critic.v_out.normalize(ret_b)
#         elif self._use_valuenorm and self.value_normalizer is not None:
#             self.value_normalizer.update(ret_b)
#             ret_norm = self.value_normalizer.normalize(ret_b)
#         else:
#             ret_norm = ret_b
#         if self.use_clipped_value_loss:
#             vpc = vpred_b + (values - vpred_b).clamp(-self.clip_param, self.clip_param)
#             ec = ret_norm - vpc; eo = ret_norm - values
#             vlc = huber_loss(ec, self.huber_delta) if self.use_huber_loss else mse_loss(ec)
#             vlo = huber_loss(eo, self.huber_delta) if self.use_huber_loss else mse_loss(eo)
#             vl = torch.max(vlo, vlc)
#         else:
#             e = ret_norm - values
#             vl = huber_loss(e, self.huber_delta) if self.use_huber_loss else mse_loss(e)
#         return (vl * masks_b).sum() / masks_b.sum().clamp(min=1.0)

#     def update(self, rollouts, global_states_batch, agent_masks=None):
#         # Compute advantages
#         all_adv = []
#         for k in range(self.num_agents):
#             r = rollouts[k]
#             if self._use_popart and hasattr(self.critic.v_out, 'denormalize'):
#                 adv = r.returns[:-1] - self.critic.v_out.denormalize(r.value_preds[:-1])
#             elif self._use_valuenorm and self.value_normalizer is not None:
#                 adv = r.returns[:-1] - self.value_normalizer.denormalize(r.value_preds[:-1])
#             else:
#                 adv = r.returns[:-1] - r.value_preds[:-1]
#             all_adv.append((adv - adv.mean()) / (adv.std() + 1e-5))

#         tv, ta, te, tr, n = 0, 0, 0, 0, 0
#         for _ in range(self.ppo_epoch):
#             # Update actors
#             for ai in range(self.num_agents):
#                 gen = (rollouts[ai].recurrent_generator(all_adv[ai], self.num_mini_batch)
#                        if self.actors[ai].is_recurrent else
#                        rollouts[ai].feed_forward_generator(all_adv[ai], self.num_mini_batch))
#                 for sample in gen:
#                     obs_b, rnn_b, cond_b, act_b, vp_b, ret_b, m_b, olp_b, adv_b = sample
#                     _, logp, ent, _ = self.actors[ai].evaluate_actions(obs_b, rnn_b, cond_b, act_b)
#                     ratio = torch.exp(logp - olp_b)
#                     s1 = ratio * adv_b
#                     s2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * adv_b
#                     al = -(torch.min(s1, s2) * m_b).sum() / m_b.sum().clamp(min=1.0)
#                     self.actor_optimizers[ai].zero_grad()
#                     (al - ent * self.entropy_coef).backward()
#                     nn.utils.clip_grad_norm_(self.actors[ai].parameters(), self.max_grad_norm)
#                     self.actor_optimizers[ai].step()
#                     ta += al.item(); te += ent.item(); tr += ratio.mean().item()

#             # Update critic
#             gen = (rollouts[0].recurrent_generator(all_adv[0], self.num_mini_batch)
#                    if self.actors[0].is_recurrent else
#                    rollouts[0].feed_forward_generator(all_adv[0], self.num_mini_batch))
#             for bi, sample in enumerate(gen):
#                 _, _, _, _, vp_b, ret_b, m_b, _, _ = sample
#                 bs = vp_b.shape[0]
#                 si = bi * bs; ei = min(si + bs, global_states_batch.shape[0])
#                 gs_b = global_states_batch[si:ei]
#                 vals = self.critic.get_value(gs_b)
#                 vals = vals.squeeze(-1) if vals.dim() > 1 else vals
#                 if vals.shape[0] != vp_b.shape[0]: vals = vals[:vp_b.shape[0]]
#                 vl = self.cal_value_loss(vals, vp_b, ret_b, m_b)
#                 self.critic_optimizer.zero_grad()
#                 (vl * self.value_loss_coef).backward()
#                 nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
#                 self.critic_optimizer.step()
#                 tv += vl.item()
#             n += 1
#         n = max(n, 1)
#         na = max(n * self.num_agents, 1)
#         return tv/n, ta/na, te/na, tr/na

#     def prep_training(self):
#         for a in self.actors: a.train()
#         self.critic.train()

#     def prep_rollout(self):
#         for a in self.actors: a.eval()
#         self.critic.eval()