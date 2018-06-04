import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs, disc_avg_v=False):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        self.disc_avg_v = disc_avg_v
        if disc_avg_v:
          self.disc_avg_value_head = nn.Linear(64, 1)
          self.disc_avg_value_head.weight.data.mul_(0.1)
          self.disc_avg_value_head.bias.data.mul_(0.0)

    def forward(self, x, discounted_time_left):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        state_values = self.value_head(x)
        if self.disc_avg_v:
          disc_avg_state_values = self.disc_avg_value_head(x)
          return state_values + discounted_time_left * disc_avg_state_values
        else:
          return state_values


class TimeValue(Value):
    def __init__(self, num_inputs, disc_avg_v=False):
        # Add 1 for time index
        super(TimeValue, self).__init__(num_inputs + 1, disc_avg_v)

    def forward(self, x, discounted_time_left):
        time_states = torch.cat((x, discounted_time_left), 1)
        return super(TimeValue, self).forward(time_states, discounted_time_left)


class EnvModel(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(EnvModel, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.env_model_head = nn.Linear(64, num_outputs)
        self.env_model_head.weight.data.mul_(0.1)
        self.env_model_head.bias.data.mul_(0.0)

        self.reward_head = nn.Linear(64, 1)
        self.reward_head.weight.data.mul_(0.1)
        self.reward_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        reward = self.reward_head(x)
        next_obs = self.env_model_head(x)
        return reward, next_obs
