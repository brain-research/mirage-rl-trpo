import argparse
from itertools import count
import functools

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

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--n-epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train for (default: 100)')
parser.add_argument('--max-time', type=int, default=1000, metavar='N',
                    help='max number of time steps to run episode for (default: 1000)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--control-variate-lr', type=float, default=3e-4, metavar='G',
                    help='control variate learning rate')
parser.add_argument('--baseline', type=str, default="none",
                    help='baseline to use (default: none)')
parser.add_argument('--log-file', type=str, default=None,
                    help='log file to write to (default: None)')
parser.add_argument('--v-optimizer', type=str, default='lbfgs',
                    help='which optimizer to use for the value function (default: lbfgs)')
parser.add_argument('--use-disc-avg-v', action='store_true',
                    help='use discounted average value parameterization of the value function (default: False)')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
value_optim = torch.optim.Adam(value_net.parameters(),
                               lr=args.control_variate_lr)

disc_avg_value_net = Value(num_inputs)

control_variate_net = Value(num_inputs)
control_variate_optim = torch.optim.Adam(control_variate_net.parameters(),
                                   lr=args.control_variate_lr)
advantage_net = Value(num_inputs + num_actions)
advantage_optim = torch.optim.Adam(advantage_net.parameters(),
                                   lr=args.control_variate_lr)

# 1 step environment model
env_net = EnvModel(num_inputs + num_actions, num_inputs)
env_optim = torch.optim.Adam(env_net.parameters(),
                             lr=args.control_variate_lr/3)

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

def update_params(batch):
    logging_info = {}
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    epsilons = torch.Tensor(np.concatenate(batch.epsilon, 0))
    states = torch.Tensor(batch.state)
    time_left = args.max_time - torch.Tensor(batch.time)
    discounted_time_left = Variable((1 - torch.pow(args.gamma, time_left))/(1 - args.gamma)).view(-1, 1)

    # Compute values and control variates
    values = compute_values(value_net, Variable(states), discounted_time_left, args.use_disc_avg_v)
    control_variates = control_variate_net(Variable(states))
    advantage_estimator = advantage_net(Variable(torch.cat(
        (states, actions), dim=1)))

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

    targets = Variable(returns)
    control_variate_targets = Variable(advantages)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = compute_values(value_net, Variable(states), discounted_time_left, args.use_disc_avg_v)

        value_loss = (values_ -  targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy()[0], get_flat_grad_from(value_net).data.double().numpy())

    #
    # Print debugging values
    #
    print('Mean advantages: %g' % (advantages.mean()))
    print('MSE returns: %g' % (returns.pow(2).mean()))
    print('MSE returns - values: %g' % ((returns - values.data).pow(2).mean()))
    print('MSE advantages: %g' % (advantages.pow(2).mean()))
    print('MSE state baseline advantages: %g' % ((advantages - control_variates.data).pow(2).mean()))
    print('MSE state-action baseline advantages: %g' % ((advantages - advantage_estimator.data).pow(2).mean()))

    logging_info['mse_v_lbfgs'] = (returns - values.data).pow(2).mean()
    logging_info['mse_none'] = advantages.pow(2).mean()
    logging_info['mse_v'] = (advantages - control_variates.data).pow(2).mean()
    logging_info['mse_q'] = (advantages - advantage_estimator.data).pow(2).mean()

    # Disable normalization of advantages
    # advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    reward_hat, next_obs_delta_hat = env_net(Variable(torch.cat(
        (states, actions), dim=1)))
    next_obs_hat = Variable(states) + next_obs_delta_hat
    # TODO: Fix time for 1 step in the future!
    next_value_hat = compute_values(value_net, next_obs_hat, discounted_time_left, args.use_disc_avg_v)
    disc_sum_reward_hat = reward_hat + args.gamma * next_value_hat

    def get_loss(volatile=False, baseline="none"):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)

        if baseline == "none":
          action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        elif baseline == "v":
          action_loss = -(Variable(advantages) - control_variates.detach()) * torch.exp(log_prob - Variable(fixed_log_prob))
        elif baseline == "q":
          current_policy_action = action_means + Variable(epsilons) * action_stds
          current_policy_advantage_estimator = advantage_net(torch.cat(
              (Variable(states), current_policy_action), dim=1))
          action_loss = -((Variable(advantages) - advantage_estimator.detach()) * torch.exp(log_prob - Variable(fixed_log_prob))
                          + current_policy_advantage_estimator)
        elif baseline == "model":
          current_policy_action = action_means + Variable(epsilons) * action_stds
          current_policy_reward_hat, current_policy_next_obs_delta_hat = env_net(torch.cat(
              (Variable(states), current_policy_action), dim=1))
          current_policy_next_obs_hat = Variable(states) + current_policy_next_obs_delta_hat
          current_policy_next_value_hat = value_net(current_policy_next_obs_hat)
          current_policy_disc_sum_reward_hat = current_policy_reward_hat + args.gamma * current_policy_next_value_hat

          action_loss = -((Variable(advantages) + values.detach() - disc_sum_reward_hat.detach()) * torch.exp(log_prob - Variable(fixed_log_prob))
                          + current_policy_disc_sum_reward_hat)
        else:
          exit()

        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, functools.partial(get_loss, baseline=args.baseline), get_kl, args.max_kl, args.damping)

    for baseline in ["none", "v", "q", "model"]:
      loss = get_loss(baseline=baseline)
      grads = torch.autograd.grad(loss, policy_net.parameters())
      loss_grad_mse = torch.cat([grad.view(-1) for grad in grads]).data.pow(2).mean()

      print("%s grad MSE: %g" % (baseline, loss_grad_mse))
      logging_info['grad_mse_%s' % baseline] = loss_grad_mse

    # Update control variates

    # state only baseline
    detached_values = values.detach()
    control_variate_loss_mse = None
    control_variate_epochs = 25  # fixed for now, change LR to change learning dynamics
    for _ in range(control_variate_epochs):
      control_variates = control_variate_net(Variable(states))
      control_variate_optim.zero_grad()
      control_variate_loss = (control_variate_targets - control_variates).pow(2).mean()
      #control_variate_loss = (targets - control_variates).pow(2).mean()
      control_variate_loss.backward()
      control_variate_optim.step()

    # state-action baseline
    advantage_net_loss_mse = None
    for _ in range(control_variate_epochs):
      advantage_estimator = advantage_net(Variable(torch.cat(
        (states, actions), dim=1)))
      advantage_optim.zero_grad()
      #advantage_net_loss = (targets - detached_values - advantage_estimator).pow(2).mean()
      advantage_net_loss = (control_variate_targets - advantage_estimator).pow(2).mean()
      #advantage_net_loss = (targets - advantage_estimator).pow(2).mean()
      advantage_net_loss.backward()
      advantage_optim.step()

    # 1-step model baseline
    env_net_loss_mse = None
    for _ in range(control_variate_epochs):
      reward_hat, next_obs_delta_hat = env_net(Variable(torch.cat(
        (states, actions), dim=1)))
      next_obs_hat = Variable(states) + next_obs_delta_hat
      # TODO: Fix discounted time left to account for 1 time step in the future!
      next_value_hat = compute_values(value_net, next_obs_hat, discounted_time_left, args.use_disc_avg_v)
      disc_sum_reward_hat = reward_hat + args.gamma * next_value_hat

      env_optim.zero_grad()
      #env_net_loss = (Variable(rewards) - reward_hat.view(-1)).pow(2).mean()
      env_net_loss = (control_variate_targets + detached_values - disc_sum_reward_hat).pow(2).mean()
      #env_net_loss = (targets - disc_sum_reward_hat).pow(2).mean()
      env_net_loss.backward()
      env_optim.step()

      if env_net_loss_mse is None:
        env_net_loss_mse = env_net_loss.data.numpy()
        print('Env net MSE: %g' % env_net_loss_mse)
        logging_info['mse_model'] = env_net_loss_mse[0]


    # Update value function
    if args.v_optimizer == 'lbfgs':
      flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
      set_flat_params_to(value_net, torch.Tensor(flat_params))
    elif args.v_optimizer == 'adam':
      for _ in range(control_variate_epochs):
        new_values = compute_values(value_net, Variable(states), discounted_time_left, args.use_disc_avg_v)
        value_optim.zero_grad()
        value_loss = (targets - new_values).pow(2).mean()
        value_loss.backward()
        value_optim.step()
    else:
      exit()

    # Deprecated:
    # Update advantage net
    # Recompute advanatages w/ the new values
    #disc_avg_values = discounted_time_left * disc_avg_value_net(Variable(states))
    #prev_return = 0
    #prev_value = 0
    #prev_advantage = 0
    #for i in reversed(range(rewards.size(0))):
    #    returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
    #    deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - disc_avg_values.data[i]
    #    advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

    #    prev_return = returns[i, 0]
    #    prev_value = disc_avg_values.data[i, 0]
    #    prev_advantage = advantages[i, 0]
    #control_variate_targets = Variable(advantages)


    return logging_info

running_state = ZFilter((num_inputs,), clip=5)
#running_reward = ZFilter((1,), demean=False, clip=10)

if args.log_file is not None:
  log_file = open(args.log_file, 'w')

for i_episode in range(args.n_epochs):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(args.max_time): # Don't infinite loop while learning
            action, epsilon = select_action(state)
            action = action.data[0].numpy()
            epsilon = epsilon.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), np.array([epsilon]), mask, next_state, reward, t)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()

    #if i_episode > 60:
    #  ipdb.set_trace()

    logging_info = update_params(batch)

    if i_episode % args.log_interval == 0:
      print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
          i_episode, reward_sum, reward_batch))
      if args.log_file is not None:
        logging_info['epoch'] = i_episode
        logging_info['reward_sum'] = reward_sum
        logging_info['reward_batch'] = reward_batch
        log_file.write('%s\n' % json.dumps(logging_info))
        #'%d\t%g\t%g\n' % (i_episode, reward_sum, reward_batch))
        log_file.flush()

if args.log_file is not None:
  log_file.close()
