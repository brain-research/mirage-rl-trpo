import os
import glob
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.patches as mpatches

# @Shane: Use matplotlib TeX support
from matplotlib import rc
rc('text', usetex=True)

# @Shane: Use the right color list for the plots!
color_list = sns.color_palette("muted")
sns.palplot(color_list)

log_dir = 'logs'
batch_size = 25000
fnames = glob.glob('{}/**/*.txt'.format(log_dir), recursive=True)

colors = {}
for i, thing in enumerate(['none', 'v', 'vh0', 'vh1']):
  colors[thing] = color_list[i]

map_baselines = {'none': 'mse_none', 'v': 'mse_v', 'vh0': 'mse_h0', 'vh1': 'mse_h1'}

# 'logs/Walker2d-v1/vh0_1002.txt'

results = {}
variances = {}

for fname in fnames:
  _, env, exp = fname.split('/')
  baseline, seed = os.path.splitext(exp)[0].split('_')

  data = []
  with open(fname, 'r') as f:
    for line in f:
      data.append(json.loads(line.strip()))

  if env not in results:
    results[env] = {}
    variances[env] = {}
  if baseline not in results[env]:
    results[env][baseline] = []
    variances[env][baseline] = []

  rewards = [x['reward_batch'] for x in data]
  var = [x[map_baselines[baseline]] for x in data]
  results[env][baseline].append(rewards)
  variances[env][baseline].append(var)


envs = ['HalfCheetah-v1', 'Walker2d-v1', 'Humanoid-v1']
envs_titles = {'halfcheetah': 'HalfCheetah-v1', 'walker2d': 'Walker2d-v1', 'humanoid': 'Humanoid-v1'}
baselines = ['none', 'v', 'vh0', 'vh1']

savgol_window = 21

# Plot rewards
fig, axes = plt.subplots(1, len(envs), figsize=(20, 6))

for envidx, env in enumerate(envs):
  ax = axes[envidx]
  for i, baseline in enumerate(baselines):
    doto = np.array(results[env][baseline])
    min_len = len(doto[0])
    doto_z1 = savgol_filter(doto.mean(0) + doto.std(0), savgol_window, 5)
    doto_z_1 = savgol_filter(doto.mean(0) - doto.std(0), savgol_window, 5)
    doto_max = savgol_filter(doto.max(0), savgol_window, 5)
    doto_min = savgol_filter(doto.min(0), savgol_window, 5)
    doto_mean = savgol_filter(doto.mean(0), savgol_window, 5)

    ax.plot(np.arange(min_len) * 25, doto_mean, color=colors[baseline], label=baseline)
    ax.fill_between(np.arange(min_len) * 25, doto_mean, np.where(doto_z1 > doto_max, doto_max, doto_z1), color=colors[baseline], alpha=0.2)
    ax.fill_between(np.arange(min_len) * 25, np.where(doto_z_1 < doto_min, doto_min, doto_z_1), doto_mean, color=colors[baseline], alpha=0.2)

  ax.tick_params(axis='both', which='major', labelsize=11)
  ax.tick_params(axis='both', which='minor', labelsize=11)

  ax.set_xlabel('Steps (thousands)')
  ax.set_ylabel('Average Reward')

  ax.grid(alpha=0.5)
  ax.set_title("{}".format(env), fontsize=18)

h1 = mpatches.Patch(color=color_list[0], label='TRPO (no baseline)')
h2 = mpatches.Patch(color=color_list[1], label='TRPO (V(s) baseline)')
h3 = mpatches.Patch(color=color_list[2], label='TRPO (simple horizon-aware V(s) baseline)')
h4 = mpatches.Patch(color=color_list[3], label='TRPO (horizon-aware V(s) baseline)')
leg = fig.legend(handles=[h4, h3, h2, h1], loc='lower center', ncol=2, prop={'size': 14})

bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += -0.15
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

fig.savefig('plots/disc_value_reward.pdf', bbox_inches='tight', format='pdf', dpi=200)


# Plot variances
fig, axes = plt.subplots(1, len(envs), figsize=(20, 6))

for envidx, env in enumerate(envs):
  ax = axes[envidx]
  for i, baseline in enumerate(baselines):
    doto = np.array(variances[env][baseline])
    min_len = len(doto[0])
    doto_z1 = savgol_filter(doto.mean(0) + doto.std(0), savgol_window, 5)
    doto_z_1 = savgol_filter(doto.mean(0) - doto.std(0), savgol_window, 5)
    doto_max = savgol_filter(doto.max(0), savgol_window, 5)
    doto_min = savgol_filter(doto.min(0), savgol_window, 5)
    doto_mean = savgol_filter(doto.mean(0), savgol_window, 5)

    ax.plot(np.arange(min_len) * 25, doto_mean, color=colors[baseline], label=baseline)
    ax.fill_between(np.arange(min_len) * 25, doto_mean, np.where(doto_z1 > doto_max, doto_max, doto_z1), color=colors[baseline], alpha=0.2)
    ax.fill_between(np.arange(min_len) * 25, np.where(doto_z_1 < doto_min, doto_min, doto_z_1), doto_mean, color=colors[baseline], alpha=0.2)

  ax.tick_params(axis='both', which='major', labelsize=11)
  ax.tick_params(axis='both', which='minor', labelsize=11)

  ax.set_xlabel('Steps (thousands)')
  ax.set_ylabel('MSE')

  ax.grid(alpha=0.5)
  ax.set_title("{}".format(env), fontsize=18)

# @Shane: Create mpatches to fill in the colors (make sure they correspond to the ones you used)
h1 = mpatches.Patch(color=color_list[0], label='TRPO (no baseline)')
h2 = mpatches.Patch(color=color_list[1], label='TRPO (V(s) baseline)')
h3 = mpatches.Patch(color=color_list[2], label='TRPO (simple horizon-aware V(s) baseline)')
h4 = mpatches.Patch(color=color_list[3], label='TRPO (horizon-aware V(s) baseline)')
leg = fig.legend(handles=[h4, h3, h2, h1], loc='lower center', ncol=2, prop={'size': 14})

# @Shane: Use to adjust the legend down.
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += -0.15
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

fig.savefig('plots/disc_value_variance.pdf', bbox_inches='tight', format='pdf', dpi=200)

