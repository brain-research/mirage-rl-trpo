import sys
import numpy as np
import json
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
matplotlib.rc('text', usetex=True)

def plot(log_file, title, smoother=None):
  with open(log_file, 'r') as f:
    variances = {}

    for line in f:
      episode, variance = json.loads(line.strip())
      if episode not in variances:
        variances[episode] = []
      variances[episode].append(variance)

  data = [(x, np.mean(variance, axis=0), np.std(variance, axis=0)/np.sqrt(len(variance))) for x, variance in sorted(variances.items())]
  x, y_mu, y_std = map(np.array, list(zip(*data)))

  plt.figure()
  for index, label in [
      (0, '$\Sigma_1$'),
      (1, '$\Sigma_2$ (no control variate)'),
      (2, '$\Sigma_2$ ($\phi(s)$ estimator)'),
      (3, '$\Sigma_2$ ($\phi(s, a)$ estimator)'),
      (4, '$\Sigma_2$ ($\phi(s)$)'),
      (5, '$\Sigma_3$'),
        ]:

    # Smooth, clip, log y
    processor = lambda y: np.log(np.clip(smoother(y), 1e-3, 1e6))

    plot_y = processor(y_mu[:, index])
    plot_y_upper = processor(y_mu[:, index] + 2*y_std[:, index])
    plot_y_lower = processor(y_mu[:, index] - 2*y_std[:, index])
    plot_x = x * 5/1000
    plt.plot(plot_x, plot_y, label=label)
    plt.fill_between(plot_x, plot_y_lower, plot_y_upper, alpha=0.3)

  plt.title(title)
  plt.ylabel('ln(Variance)')
  plt.xlabel('Steps (millions)')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
  plt.savefig('%s_variance_gae.pdf' % title, bbox_inches='tight')
  plt.savefig('%s_variance_gae.png' % title, bbox_inches='tight')

if __name__ == '__main__':
  log_file = sys.argv[1]

  plot(log_file,
       title='HalfCheetah-v1',
       #title='Humanoid-v1',
       #smoother=lambda x: savgol_filter(x, 21, 1))
       smoother=lambda x: x)

