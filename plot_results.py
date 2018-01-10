import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import savgol_filter

#log_dir = os.path.join('logs', 'halfcheetah_q_lr')
log_dir = os.path.join('logs', 'halfcheetah')
log_dir = 'logs'

results = {}
for log_file in os.listdir(log_dir):
  if log_file.endswith('.json'):
    with open(os.path.join(log_dir, log_file), 'r') as f:
      baseline, trial = log_file[:-4].rsplit('_', 1)
      if baseline not in results:
        results[baseline] = []
      result = []
      for line in f:
        result.append(json.loads(line.strip()))
      results[baseline].append(result)

colors = ['r', 'g', 'b', 'c']

for i, (k, v) in enumerate(results.items()):
#  plt.plot(np.mean(np.array(v), axis=0), color=colors[i], label=k)
  for j, res in enumerate(v):
    #plt.plot([row['reward_batch'] for row in res])

    fields = ['mse_none', 'mse_v', 'mse_q', 'mse_model']

    for k, field in enumerate(fields):
      y = [row[field]/row['mse_none'] for row in res]
      if j == 0:
        plt.plot(np.arange(len(y)) * 16, savgol_filter(y, 21, 2), color=colors[k], label=field)
        plt.plot(np.arange(len(y)) * 16, y, color=colors[k], alpha=0.1)
      else:
        pass
        #plt.plot(y, color=colors[k])

plt.legend()
plt.ylim(0.9, 1.1)
plt.xlabel('Steps (thousands)')
plt.savefig('tmp.png')

