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
value_net = Value(num_inputs, args.use_disc_avg_v)
state_cv_net = Value(num_inputs)
state_action_cv_net = Value(num_inputs + num_actions)
gae_state_cv_net = Value(num_inputs)
gae_state_action_cv_net = Value(num_inputs + num_actions)

# Read in the networks
nets = [('policy', policy_net),
        ('value', value_net),
        ('v', state_cv_net),
        ('q', state_action_cv_net),
        ('gae_v', gae_state_cv_net),
        ('gae_q', gae_state_action_cv_net), ]

for (net_name, net) in nets:
  net.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, '%s_%d.chkpt' % (net_name, eval_args.checkpoint))))

def select_action(state, epsilon=None):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))

    if epsilon is None:
      epsilon = Variable(torch.normal(torch.zeros(action_mean.size()),
                                      torch.ones(action_std.size())))

    action = action_mean + epsilon * action_std
    return action, epsilon

def compute_values(value_net, states, discounted_time_left, use_disc_avg_v=False):
  if use_disc_avg_v:
    state_value, disc_avg_state_value = value_net(states)
    return state_value + discounted_time_left * disc_avg_state_value
  else:
    return value_net(states)

def vectorize(t):
  return torch.cat([x.view(-1) for x in t])

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
  # Policy gradient for an entire batch averaged over time steps
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
  grad = torch.autograd.grad(loss, policy_net.parameters())
  return vectorize(grad).data.numpy()


def grad_log_pi(state, action):
  # Compute \nabla \log \pi (action | state) for a single time step
  state = Variable(torch.Tensor(state).view(1, -1))
  action = Variable(torch.Tensor(action).view(1, -1))
  action_means, action_log_stds, action_stds = policy_net(state)
  log_prob = normal_log_density(action, action_means,
                                action_log_stds, action_stds)
  grad = torch.autograd.grad(log_prob, policy_net.parameters())
  return vectorize(grad).data.numpy()

def compute_estimators(q_x, q_y, q, v_x, v_y, value_func_est, q_func_est, q_func_est_prime,
                       shared_grad_log_pi, unshared_grad_log_pi):
  sq_shared_grad_log_pi = shared_grad_log_pi * shared_grad_log_pi
  term_1 = np.mean(q_x*q_x*sq_shared_grad_log_pi + q_y*q_y*sq_shared_grad_log_pi)/2
  term_2 = np.mean(q_x*q_y*sq_shared_grad_log_pi)
  term_2_value_baseline = np.mean((q_x - v_x)*(q_y - v_y)*sq_shared_grad_log_pi)
  term_2_value_func_est_baseline = np.mean((q_x - value_func_est)*
                                           (q_y - value_func_est)*
                                           sq_shared_grad_log_pi)
  term_2_q_func_est_baseline = np.mean((q_x - q_func_est)*
                                       (q_y - q_func_est)*
                                       sq_shared_grad_log_pi)

  # Might as well use the value estimates to reduce the variance of this term
  term_3 = np.mean((q_x - v_x)*(q - v_y)*shared_grad_log_pi * unshared_grad_log_pi +
                   (q - v_x)*(q_y - v_y)*shared_grad_log_pi * unshared_grad_log_pi)/2
  term_3_q_func_est_baseline = np.mean((q_x - q_func_est)*(q - q_func_est_prime)*shared_grad_log_pi * unshared_grad_log_pi +
                                       (q - q_func_est_prime)*(q_y - q_func_est)*shared_grad_log_pi * unshared_grad_log_pi)/2

  var_term_1 = term_1 - term_2
  var_term_2 = term_2 - term_3
  var_term_2_value_func_est_baseline = term_2_value_func_est_baseline - term_3
  var_term_2_q_func_est_baseline = term_2_q_func_est_baseline - term_3_q_func_est_baseline
  var_term_2_value_baseline = term_2_value_baseline - term_3
  var_term_3 = term_3 # - E[g]^2 which is small, so this is an upper bound

  return var_term_1, var_term_2, var_term_2_value_func_est_baseline, var_term_2_q_func_est_baseline, var_term_2_value_baseline, var_term_3


def estimate_variance(batch, n_samples=50):
  actions = np.concatenate(batch.action, 0)
  epsilons = torch.Tensor(np.concatenate(batch.epsilon, 0))
  states = batch.state
  n_steps = len(batch.state)

  def _select_action(state):
    a, _ = select_action(s)
    a = a.data[0].numpy()
    return a

  variance_hats = []
  for _ in range(n_samples):
    # Select random state
    i = random.randint(0, n_steps-1)

    s = states[i]
    a = actions[i, :]
    mujoco_state = batch.mujoco_state[i]
    time_left = args.max_time - batch.time[i]
    discounted_time_left = (1. - args.gamma ** time_left) / (1. - args.gamma)

    # Compute function approximators
    value_func_base_est = compute_values(value_net, Variable(torch.Tensor(s)),
                                         Variable(torch.Tensor([discounted_time_left])),
                                         args.use_disc_avg_v).data.numpy()[0]
    value_func_est = value_func_base_est + state_cv_net(Variable(torch.Tensor(s))).data.numpy()[0]
    gae_value_func_est = gae_state_cv_net(Variable(torch.Tensor(s))).data.numpy()[0]

    q_func_est = value_func_base_est + state_action_cv_net(
        Variable(torch.cat([torch.Tensor(s), torch.Tensor(a)]))).data.numpy()[0]
    gae_q_func_est = gae_state_action_cv_net(Variable(torch.cat([torch.Tensor(s), torch.Tensor(a)]))).data.numpy()[0]

    shared_grad_log_pi = grad_log_pi(s, a)

    q_x, gae_q_x = calc_advantage_estimator(mujoco_state, a, time_left)
    q_y, gae_q_y = calc_advantage_estimator(mujoco_state, a, time_left)

    # Calculation value function estimates
    a = _select_action(s)
    v_x, gae_v_x = calc_advantage_estimator(mujoco_state, a, time_left)

    a = _select_action(s)
    v_y, gae_v_y = calc_advantage_estimator(mujoco_state, a, time_left)

    a = _select_action(s)
    q, gae_q = calc_advantage_estimator(mujoco_state, a, time_left)
    unshared_grad_log_pi = grad_log_pi(s, a)

    q_func_est_prime = value_func_est + state_action_cv_net(
        Variable(torch.cat([torch.Tensor(s), torch.Tensor(a)]))).data.numpy()[0]
    gae_q_func_est_prime = gae_state_action_cv_net(Variable(torch.cat([torch.Tensor(s), torch.Tensor(a)]))).data.numpy()[0]

    variance_hats.append(
        compute_estimators(q_x, q_y, q, v_x, v_y,
                           value_func_est, q_func_est, q_func_est_prime,
                           shared_grad_log_pi, unshared_grad_log_pi) +
        compute_estimators(gae_q_x, gae_q_y, gae_q, gae_v_x, gae_v_y,
                           gae_value_func_est, gae_q_func_est, gae_q_func_est_prime,
                           shared_grad_log_pi, unshared_grad_log_pi))

  return np.mean(variance_hats, axis=0)


def calc_advantage_estimator(mujoco_state, action, time_left, epsilons=None):
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

    action, _ = select_action(state, epsilon=epsilons[t, :] if epsilons is not None else None)
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

  return returns[0], advantages[0]

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

for i_episode in range(2): #eval_args.n_epochs):
  batch_size = args.batch_size
  #_, _, batch = get_batch(batch_size)
  #g_mu_1 = get_policy_grad(batch)

  #_, _, batch = get_batch(batch_size)
  #g_mu_2 = get_policy_grad(batch)

  # Compute variance
  reward_batch, _, batch = get_batch(batch_size)
  #print(reward_batch)
  var_hat = estimate_variance(batch,
                              n_samples=10)
  #print(np.mean((vectorize(g_mu_1) * vectorize(g_mu_2)).data.numpy()))

  print(json.dumps([eval_args.checkpoint, var_hat.tolist()]))
  sys.stdout.flush()
