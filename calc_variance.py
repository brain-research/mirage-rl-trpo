import argparse
from itertools import count
import functools

import random
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

import ipdb
import sys
import json
import os
import pickle

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--baseline', type=str, default="none",
                    help='baseline to use (default: none)')
parser.add_argument('--log-file', type=str, default=None,
                    help='log file to write to (default: None)')
parser.add_argument('--checkpoint-dir', type=str, default=None,
                    help='directory to write checkpoints to (default: None)')
parser.add_argument('--checkpoint', type=int, default=5, metavar='N',
                    help='interval between saving model checkpoints (default: 5)')
eval_args = parser.parse_args()

# Load in checkpoint
with open(os.path.join(eval_args.checkpoint_dir, 'training_args.p'), 'rb') as f:
  args = pickle.load(f)

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(eval_args.seed)
torch.manual_seed(eval_args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
control_variate_net = Value(num_inputs)
advantage_net = Value(num_inputs + num_actions)
env_net = EnvModel(num_inputs + num_actions, num_inputs)

# Read in the networks
nets = [('policy', policy_net),
        ('value', value_net),
        ('v', control_variate_net),
        ('q', advantage_net),
        ('env', env_net), ]
for (net_name, net) in nets:
  net.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, '%s_%d.chkpt' % (net_name, eval_args.checkpoint))))

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    epsilon = Variable(torch.normal(torch.zeros(action_mean.size()),
                                    torch.ones(action_std.size())))
    action = action_mean + epsilon * action_std
    return action, epsilon

def compute_values(value_net, states, discounted_time_left, use_disc_avg_v=False):
  if use_disc_avg_v:
    return discounted_time_left * value_net(states)
  else:
    return value_net(states)

def get_advantages(batch):
  rewards = torch.Tensor(batch.reward)
  masks = torch.Tensor(batch.mask)
  actions = torch.Tensor(np.concatenate(batch.action, 0))
  states = torch.Tensor(batch.state)
  time_left = args.max_time - torch.Tensor(batch.time)
  discounted_time_left = Variable((1 - torch.pow(args.gamma, time_left))/(1 - args.gamma)).view(-1, 1)

  # Compute values and control variates
  values = compute_values(value_net, Variable(states), discounted_time_left, args.use_disc_avg_v)

  returns = torch.Tensor(actions.size(0),1)
  deltas = torch.Tensor(actions.size(0),1)
  advantages = torch.Tensor(actions.size(0),1)

  prev_return = 0
  prev_value = 0
  prev_advantage = 0
  for i in reversed(range(rewards.size(0))):
      returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
      deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
      advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

      prev_return = returns[i, 0]
      prev_value = values.data[i, 0]
      prev_advantage = advantages[i, 0]

  return advantages

def get_policy_grad(batch):
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    advantages = get_advantages(batch)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    return grads

def vectorize(t):
  return torch.cat([x.view(-1) for x in t])

def estimate_variance(batch, g_mu_1, g_mu_2):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    epsilons = torch.Tensor(np.concatenate(batch.epsilon, 0))
    states = torch.Tensor(batch.state)
    time_left = args.max_time - torch.Tensor(batch.time)
    discounted_time_left = Variable((1 - torch.pow(args.gamma, time_left))/(1 - args.gamma)).view(-1, 1)

    # Compute values and control variates
    values = compute_values(value_net, Variable(states), discounted_time_left, args.use_disc_avg_v)

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]


    def grad_log_pi(state, action):
      # Compute \nabla \log \pi (action | state) for a single time step
      state = Variable(torch.Tensor(state).view(1, -1))
      action = Variable(torch.Tensor(action).view(1, -1))
      action_means, action_log_stds, action_stds = policy_net(state)
      log_prob = normal_log_density(action, action_means,
                                    action_log_stds, action_stds)
      grad = torch.autograd.grad(log_prob, policy_net.parameters())
      return vectorize(grad).data.numpy()

    # Select random state, compute gradient estimate for that state
    psi_vars = []
    for _ in range(1000):
      i = random.randint(0, rewards.size(0)-1)

      s = states[i, :].numpy()
      a = actions[i, :].numpy()
      mujoco_state = batch.mujoco_state[i]
      time_left = args.max_time - batch.time[i]

      vecs = []

      common_epsilons = [Variable(torch.normal(torch.zeros(epsilons.size()),
                                               torch.ones(epsilons.size()))) for _ in range(10)]
      mu = 0
      print('new state')
      for _ in range(10):
        a, _ = select_action(s)
        a = a.data[0].numpy()


        w = 0
        for j in range(10):
          w += calc_advantage_estimator(mujoco_state, a, time_left, common_epsilons[j])
        w /= 10
        w *= grad_log_pi(s, a)
        mu += w
        vecs.append(w)

      mu /= len(vecs)
      psi_vars.append(np.mean([np.mean(np.square(vec - mu)) for vec in vecs]))

      print(psi_vars[-1])

    ipdb.set_trace()
    print(np.mean(psi_vars))

    """
    Need 4 rollouts, can reuse 1 rollout throughout

    Function to compute grad log pi would be useful

    get a from action[i]
      x = Advantage(s, a) * grad log pi(a, s)
      y = hat_A(s, a) * grad_log_pi(a, s)

    sample a | s
      w = hat_A(s, a) * grad_log_pi(a, s)

    sample a | s
      z = hat_A(s, a) * grad_log_pi(a, s)

    x and y share the same action (which is the action we took originally)
    reuse precomputed advantage for x

    var.append((x - w)*(y - z))
    """

    # Select random state, compute gradient for that state
    var_hat = []
    g_mu = 0
    for i in range(0, rewards.size(0)):
      #i = random.randint(0, rewards.size(0)-1)

      action_means, action_log_stds, action_stds = policy_net(Variable(states[i, :].view(1, -1)))
      log_prob = normal_log_density(Variable(actions[i, :].view(1, -1)), action_means, action_log_stds, action_stds)
      action_loss = -Variable(advantages[i]) * log_prob

      g = torch.autograd.grad(action_loss, policy_net.parameters())
      g = vectorize(g)
      var_hat.append(((g - g_mu_1) * (g - g_mu_2)).mean().data.numpy())
      g_mu += g

    g_mu /= rewards.size(0)

    return np.array(var_hat).mean()

def select_action_epsilon(state, epsilon):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = action_mean + epsilon * action_std
    return action

def calc_advantage_estimator(mujoco_state, action, time_left, epsilons):
  # Compute the GAE advantage estimate starting from a state and
  # taking a particular action until time_left or done.
  rewards = []
  states = []
  masks = []
  times = []

  env.reset()
  env.env.set_state(*mujoco_state)

  # Rollout until done or time_left
  for t in range(time_left):
    state, reward, done, _ = env.step(action)
    state = running_state(state, update=False)

    # Store path
    states.append(state)
    rewards.append(reward)
    times.append(t)

    if done:
      masks.append(0.)
      break
    else:
      masks.append(1.)

    action = select_action_epsilon(state, epsilons[t, :])
    action = action.data[0].numpy()

  # Compute GAE estimator
  states = torch.Tensor(states)
  time_left = args.max_time - torch.Tensor(times)
  discounted_time_left = Variable((1 - torch.pow(args.gamma, time_left))/(1 - args.gamma)).view(-1, 1)

  # Compute values and control variates
  values = compute_values(value_net, Variable(states), discounted_time_left, args.use_disc_avg_v)

  returns = np.zeros(len(rewards))
  deltas = np.zeros_like(returns)
  advantages = np.zeros_like(returns)

  prev_return = 0
  prev_value = 0
  prev_advantage = 0
  for i in reversed(range(len(rewards))):
    returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
    deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i].numpy()[0]
    advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

    prev_return = returns[i]
    prev_value = values.data[i].numpy()[0]
    prev_advantage = advantages[i]

  return advantages[0]

with open(os.path.join(eval_args.checkpoint_dir, 'zfilter_%d.p' % eval_args.checkpoint), 'rb') as f:
  running_state = pickle.load(f)

def get_batch(batch_size):
  memory = Memory()

  num_steps = 0
  reward_batch = 0
  num_episodes = 0
  while num_steps < batch_size:
      state = env.reset()
      state = running_state(state, update=False)

      reward_sum = 0
      for t in range(args.max_time):
          mujoco_state = env.env.get_state()
          action, epsilon = select_action(state)
          action = action.data[0].numpy()
          epsilon = epsilon.data[0].numpy()
          next_state, reward, done, _ = env.step(action)
          reward_sum += reward

          next_state = running_state(next_state, update=False)

          mask = 1
          if done:
              mask = 0

          memory.push(state, np.array([action]), np.array([epsilon]), mask, next_state, reward, t, mujoco_state)

          if done:
              break

          state = next_state
      num_steps += (t-1)
      num_episodes += 1
      reward_batch += reward_sum

  reward_batch /= num_episodes
  batch = memory.sample()

  return reward_batch, reward_sum, batch

for i_episode in range(args.n_epochs):
  batch_size = args.batch_size
  _, _, batch = get_batch(batch_size)
  g_mu_1 = get_policy_grad(batch)

  _, _, batch = get_batch(batch_size)
  g_mu_2 = get_policy_grad(batch)

  # Compute variance
  _, _, batch = get_batch(batch_size)
  var_hat = estimate_variance(batch, vectorize(g_mu_1),
                    vectorize(g_mu_2))

  #print(var_hat)

