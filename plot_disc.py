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

log_dir = 'gs_results'
batch_size = 25000
fnames = glob.glob('{}/*.txt'.format(log_dir), recursive=True)

envs = ['HalfCheetah-v1', 'Walker2d-v1', 'Humanoid-v1'] #'Walker2d-v1'
baselines = ['v', 'horizon_v', 'time_v']
colors = dict(zip(baselines, color_list[:len(baselines)]))

savgol_window = 21
smooth = lambda x: savgol_filter(x, savgol_window, 5)

results = {}
variances = {}

for fname in fnames:
  exp = os.path.splitext(os.path.basename(fname))[0].split('_')
  env = exp[0]
  seed = exp[-1]
  if len(exp) == 5:
    baseline = 'v'
  else:
    baseline = '%s_v' % exp[2]

  data = []
  with open(fname, 'r') as f:
    for line in f:
      data.append(json.loads(line.strip()))

  if env not in results:
    results[env] = {}
    variances[env] = {}
  if baseline not in results[env]:
    results[env][baseline] = []
    variances[env][baseline] = {}

  rewards = [x['reward_batch'] for x in data]
  results[env][baseline].append(rewards)
  for var_baseline in baselines:
    var = [x['var_%s' % var_baseline] for x in data]
    variances[env][baseline][var_baseline] = var

# Plot rewards
def plot(data, ylabel, out):
  fig, axes = plt.subplots(1, len(envs), figsize=(20, 6))

  for envidx, env in enumerate(envs):
    ax = axes[envidx]
    for i, baseline in enumerate(baselines):
      doto = data[env][baseline]
      min_len = min(map(len, doto))
      doto = [x[:min_len] for x in doto]
      doto = np.array(doto)
      doto_z1 = savgol_filter(doto.mean(0) + doto.std(0), savgol_window, 5)
      doto_z_1 = savgol_filter(doto.mean(0) - doto.std(0), savgol_window, 5)
      doto_max = savgol_filter(doto.max(0), savgol_window, 5)
      doto_min = savgol_filter(doto.min(0), savgol_window, 5)
      doto_mean = savgol_filter(doto.mean(0), savgol_window, 5)

      ax.plot(np.arange(min_len) * 25, doto_mean, color=colors[baseline])
      ax.fill_between(np.arange(min_len) * 25, doto_mean, np.where(doto_z1 > doto_max, doto_max, doto_z1), color=colors[baseline], alpha=0.2)
      ax.fill_between(np.arange(min_len) * 25, np.where(doto_z_1 < doto_min, doto_min, doto_z_1), doto_mean, color=colors[baseline], alpha=0.2)

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=11)

    ax.set_xlabel('Steps (thousands)')
    ax.set_ylabel(ylabel)

    ax.grid(alpha=0.5)
    ax.set_title("{}".format(env), fontsize=18)

  legend_handles = [
    mpatches.Patch(color=colors['v'], label='Value baseline'),
    mpatches.Patch(color=colors['horizon_v'], label='Horizon-aware value baseline'),
    mpatches.Patch(color=colors['time_v'], label='Value with timeleft baseline'),
  ]
  leg = plt.legend(handles=legend_handles, loc='lower center', ncol=2, prop={'size': 14})

  bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
  bb.y0 += -0.25
  bb.x0 -= 2.4
  leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

  fig.savefig('plots/%s.pdf' % out, bbox_inches='tight', format='pdf', dpi=200)

plot(results, 'Average Reward', 'reward')

# Plot variances
fig, axes = plt.subplots(1, len(envs), figsize=(20, 6))

for envidx, env in enumerate(envs):
  ax = axes[envidx]
  for i, baseline in enumerate(baselines):
    doto = np.array(variances[env]['v'][baseline])
    ax.plot(np.arange(len(doto)) * 25, smooth(doto), color=colors[baseline], label=baseline)

  ax.tick_params(axis='both', which='major', labelsize=11)
  ax.tick_params(axis='both', which='minor', labelsize=11)

  ax.set_xlabel('Steps (thousands)')
  ax.set_ylabel('Variance')

  ax.grid(alpha=0.5)
  ax.set_title("{}".format(env), fontsize=18)

legend_handles = [
  mpatches.Patch(color=colors['v'], label='Value baseline'),
  mpatches.Patch(color=colors['horizon_v'], label='Horizon-aware value baseline'),
  mpatches.Patch(color=colors['time_v'], label='Value with timeleft baseline'),
]
leg = plt.legend(handles=legend_handles, loc='lower center', ncol=2, prop={'size': 14})

# @Shane: Use to adjust the legend down.
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += -0.25
bb.x0 -= 2.4
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

fig.savefig('plots/variance.pdf', bbox_inches='tight', format='pdf', dpi=200)

