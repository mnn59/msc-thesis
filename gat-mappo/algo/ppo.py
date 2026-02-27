"""Original DRL-OR PPO (IPPO/DTDE baseline)."""
import torch
import torch.nn as nn
import torch.optim as optim

class PPO:
    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch,
                 value_loss_coef, entropy_coef, lr=None, eps=None,
                 max_grad_norm=None):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        tv, ta, te, n = 0, 0, 0, 0
        for _ in range(self.ppo_epoch):
            gen = (rollouts.recurrent_generator(advantages, self.num_mini_batch)
                   if self.actor_critic.is_recurrent else
                   rollouts.feed_forward_generator(advantages, self.num_mini_batch))
            for sample in gen:
                obs_b, rnn_b, cond_b, act_b, vp_b, ret_b, m_b, olp_b, adv_b = sample
                values, logp, entropy, _ = self.actor_critic.evaluate_actions(
                    obs_b, rnn_b, cond_b, act_b)
                ratio = torch.exp(logp - olp_b)
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * adv_b
                al = -torch.min(s1, s2).mean()
                vl = 0.5 * (ret_b - values).pow(2).mean()
                self.optimizer.zero_grad()
                (vl * self.value_loss_coef + al - entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                tv += vl.item(); ta += al.item(); te += entropy.item(); n += 1
        n = max(n, 1)
        return tv/n, ta/n, te/n