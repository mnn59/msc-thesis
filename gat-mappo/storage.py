"""RolloutStorage - same structure as DRL-OR original."""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutStorage:
    def __init__(self, num_steps, obs_shape, action_space, recurrent_hidden_state_size, condition_state_size):
        self.obs = torch.zeros(num_steps + 1, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, recurrent_hidden_state_size)
        self.condition_states = torch.zeros(num_steps + 1, condition_state_size)
        self.rewards = torch.zeros(num_steps, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        action_shape = 1 if action_space.__class__.__name__ == 'Discrete' else action_space.shape[0]
        self.actions = torch.zeros(num_steps, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.condition_states = self.condition_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, rnn_hxs, cond, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(rnn_hxs)
        self.condition_states[self.step + 1].copy_(cond)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.condition_states[0].copy_(self.condition_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step+1] * self.masks[step+1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step+1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step+1] * gamma * self.masks[step+1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps = self.rewards.size(0)
        mini_batch_size = num_steps // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(num_steps)), mini_batch_size, drop_last=False)
        for indices in sampler:
            yield (self.obs[:-1].unsqueeze(1)[indices],
                   self.recurrent_hidden_states[:-1][indices],
                   self.condition_states[:-1].unsqueeze(1)[indices],
                   self.actions[indices].unsqueeze(1),
                   self.value_preds[:-1][indices].unsqueeze(1),
                   self.returns[:-1][indices].unsqueeze(1),
                   self.masks[:-1][indices],
                   self.action_log_probs[indices].unsqueeze(1),
                   advantages[indices].unsqueeze(1))

    def recurrent_generator(self, advantages, num_mini_batch):
        yield (self.obs[:-1].unsqueeze(1),
               self.recurrent_hidden_states[0:1],
               self.condition_states[:-1].unsqueeze(1),
               self.actions.unsqueeze(1),
               self.value_preds[:-1].unsqueeze(1),
               self.returns[:-1].unsqueeze(1),
               self.masks[:-1],
               self.action_log_probs.unsqueeze(1),
               advantages.unsqueeze(1))