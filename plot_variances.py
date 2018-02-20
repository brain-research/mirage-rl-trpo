import sys
import numpy as np
import json
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc('text', usetex=True)
import matplotlib.patches as mpatches

color_list = sns.color_palette("muted")
sns.palplot(color_list)

def plot(log_file, title, smoother=None, gae=True):
  with open(log_file, 'r') as f:
    variances = {}

    for line in f:
      episode, reward, variance = json.loads(line.strip())
      if episode not in variances:
        variances[episode] = []
      variances[episode].append(variance)

  data = [(x, np.mean(variance, axis=0), np.std(variance, axis=0)/np.sqrt(len(variance))) for x, variance in sorted(variances.items())]
  x, y_mu, y_std = map(np.array, list(zip(*data)))

  legend_handles = []
  for i, (index, label) in enumerate([
      (0, '$\Sigma_\\tau$'),
      (1, '$\Sigma_a^0$'),
      (2, '$\Sigma_a^{\hat{\phi}(s)}$ (learned)'),
      (3, '$\Sigma_a^{\hat{\phi}(s, a)}$ (learned)'),
      (4, '$\Sigma_a^{\phi(s)}$'),
      (5, '$\geq \Sigma_s$'),
  ]):

    if gae:
      index += 6

    # Smooth, clip, log y
    processor = lambda y: np.clip(smoother(y), 1e-3, None)

    plot_y = processor(y_mu[:, index])
    plot_y_upper = processor(y_mu[:, index] + 2*y_std[:, index])
    plot_y_lower = processor(y_mu[:, index] - 2*y_std[:, index])
    plot_x = x * 5
    plt.plot(plot_x, plot_y, color=color_list[i], label=label)
    plt.fill_between(plot_x, plot_y_lower, plot_y_upper, color=color_list[i], alpha=0.2)

    legend_handles.append(mpatches.Patch(color=color_list[i], label=label))

  plt.tick_params(labelsize=11)
  plt.yscale('log', nonposy='clip')
  plt.ylabel('Variance', fontsize=16)
  plt.xlabel('Steps (thousands)', fontsize=16)

  if gae:
    plt.title(title + ' (GAE)', fontsize=18)
  else:
    plt.title(title, fontsize=18)

  plt.grid(alpha=0.5)
  return legend_handles

def ema(x, alpha=0.95):
  res = []
  mu = 0.
  for val in x:
    mu = alpha*mu + (1 - alpha)*val
    res.append(mu)

  return np.array(res)

if __name__ == '__main__':
  log_file = sys.argv[1]
  title=sys.argv[2]

  smoother=lambda x: x
  #smoother=lambda x: ema(x)
  #smoother=lambda x: savgol_filter(x, 21, 1)

  fig = plt.figure(figsize=(20,6))
  plt.subplot(1, 2, 1)
  legend_handles = plot(log_file,
                        title=title,
                        smoother=smoother,
                        gae=False,
                        )
  plt.subplot(1, 2, 2)
  legend_handles = plot(log_file,
                        title=title,
                        smoother=smoother,
                        gae=True,
                        )

  plt.legend(handles=legend_handles,
             loc='upper center', bbox_to_anchor=(-0.10, -0.13),
             ncol=6,
             prop={'size': 20})

  plt.savefig('plots/variance/%s_variance.pdf' % title, bbox_inches='tight')

