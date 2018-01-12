import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import savgol_filter

log_dir = 'gcp-logs'

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
      control_alg, optimizer, use_disc, _, seed = log_file[:-5].split('-')
      key = (control_alg, optimizer, use_disc)
      print(key, seed)
      if key not in results:
        results[key] = []
      result = []
      for line in f:
        result.append(json.loads(line.strip()))
      results[key].append(result)

colors = ['r', 'g', 'b', 'c']

    #min_len = min(map(len, y))
    #y = np.array([res[:min_len] for res in y])
    #plt.plot(np.arange(y.shape[1]) * 5, ema(np.mean(y, axis=0), 0.95), color=colors[i], label='-'.join(k))
    #plt.plot(np.arange(y.shape[1]) * 5, np.mean(y, axis=0), alpha=0.1, color=colors[i])

for control_alg in ['none', 'v', 'q', 'model']:
  plt.figure()
  i = 0
  for k, v in results.items():
    if k[0] == control_alg:
      y = [[row['reward_batch'] for row in res] for res in v]

      for j, res in enumerate(y):
        plt.plot(np.arange(len(res)) * 5, ema(res, 0.95), color=colors[i],
                 label='-'.join(k) if j == 0 else None)
      i += 1
    plt.legend()
    plt.xlabel('Steps (thousands)')
    plt.ylabel('Reward')
    plt.title(control_alg)
    plt.savefig('plots/reward_control_alg_%s.png' % control_alg)

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

