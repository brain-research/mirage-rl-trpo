import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.patches as mpatches

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


color_list = sns.color_palette("muted")
sns.palplot(color_list)

log_dir = 'gcp3-logs'

def ema(x, decay=0.99, init=0.):
  res = []
  prev = init
  for val in x:
    prev = decay * prev + (1 - decay) * val
    res.append(prev)

  return res

# Read in results
results = {}
for log_file in os.listdir(log_dir):
  if log_file.endswith('.json'):
    with open(os.path.join(log_dir, log_file), 'r') as f:
      env, optimizer, seed = log_file[:-5].split('-')
      key = (env, optimizer)
      # print(key, seed)
      if key not in results:
        results[key] = []
      result = []
      for line in f:
        result.append(json.loads(line.strip()))
      results[key].append(result)

envs = ['halfcheetah', 'walker2d', 'humanoid']
envs_titles = {'halfcheetah': 'HalfCheetah', 'walker2d': 'Walker2d', 'humanoid': 'Humanoid'}
savgol_window = 7
colors = {}
for i, thing in enumerate(['v', 'q', 'disc', 'none']):
  colors[thing] = color_list[i]

window_len = 1000
fig, axes = plt.subplots(1, len(envs), figsize=(20, 6))
for envidx, env in enumerate(envs):
  ax = axes[envidx]
  print('envidx', envidx, 'env', env)
  ys = []
  for k, v in results.items():
    if k[0] == env:
      y = [[row['reward_batch'] for row in res] for res in v]
      print(len(y))
      print(len(y[0]))
      print(len(y[1]))
      print(len(y[2]))
      print(len(y[3]))
      print(len(y[4]))
      min_len = min([len(x) for x in y])
      y = [[row['reward_batch'] for row in res][:min_len] for res in v]

      y = np.stack(y)
      y_z1 = savgol_filter(y.mean(0) + y.std(0), savgol_window, 5)
      y_z_1 = savgol_filter(y.mean(0) - y.std(0), savgol_window, 5)
      y_max = savgol_filter(y.max(0), savgol_window, 5)
      y_min = savgol_filter(y.min(0), savgol_window, 5)
      y_mean = savgol_filter(y.mean(0), savgol_window, 5)
      # ax.plot(np.arange(len(res)) * 5, ema(res, 0.95), color=colors[k[1]], label='-'.join(k) if j == 0 else None)

      ax.plot(np.arange(min_len) * 5, y_mean, color=colors[k[1]], label='-'.join(k))
      ax.fill_between(np.arange(min_len) * 5, y_mean, np.where(y_z1 > y_max, y_max, y_z1), color=colors[k[1]], alpha=0.2)
      ax.fill_between(np.arange(min_len) * 5, np.where(y_z_1 < y_min, y_min, y_z_1), y_mean, color=colors[k[1]], alpha=0.2)

  # Sorting and plotting legend entries
  handles, labels = axes[envidx].get_legend_handles_labels()
  import operator
  hl = sorted(zip(handles, labels),
              key=operator.itemgetter(1))
  handles2, labels2 = zip(*hl)
  # ax.legend(handles2, labels2, loc='upper left')
  ax.set_title(envs_titles[env])

h1 = mpatches.Patch(color=color_list[0], label='State baseline')
h2 = mpatches.Patch(color=color_list[1], label='State-action baseline')
h3 = mpatches.Patch(color=color_list[2], label='Discounted value function with no baseline')
h4 = mpatches.Patch(color=color_list[3], label='No baseline')
leg = fig.legend(handles=[h3, h4, h1, h2], loc='lower center', ncol=4, prop={'size': 10})

fig.savefig('disc_value_mean_stds.png')
quit(1)

# Plot reward for lbfg/y
plt.figure()
i = 0
for k, v in results.items():
  if k[1] == 'lbfgs' and k[2] == 'y':
    y = [[row['reward_batch'] for row in res] for res in v]

    for j, res in enumerate(y):
      plt.plot(np.arange(len(res)) * 5, ema(res, 0.95), color=colors[i],
               label='-'.join(k) if j == 0 else None)
    i += 1
  plt.legend()
  plt.xlabel('Steps (thousands)')
  plt.ylabel('Reward')
  plt.title(control_alg)
  plt.savefig('plots/reward_lbfgs_y.png')

for k, v in results.items():
  plt.figure()
  fields = ['mse_none', 'mse_v', 'mse_q', 'mse_model']

  for i, res in enumerate(v):
    plt.subplot(len(v), 1, i+1)
    for j, field in enumerate(fields):
      y = [row[field]/row['mse_none'] for row in res]
      plt.plot(np.arange(len(y)) * 5, ema(y, 0.95, 1.), color=colors[j], label=field)
      plt.ylim(0.95, 1.05)

  plt.legend()
  plt.xlabel('Steps (thousands)')
  plt.ylabel('MSE relative to no baseline')
  plt.title('-'.join(k))
  plt.savefig('plots/mse_%s.png' % ('-'.join(k)))

