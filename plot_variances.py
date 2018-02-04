import numpy as np
import json
from scipy.signal import savgol_filter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

with open('log.txt', 'r') as f:
  episodes = []
  variances = []

  for line in f:
    episode, variance = json.loads(line.strip())

    episodes.append(episode)
    variances.append(variance)


episodes = np.array(episodes)
variances = np.array(variances)

plt.figure()
for index, label in [
    (0, 'Term 1'),
    (1, 'Term 2 (no control variate)'),
    (2, 'Term 2 (phi(s) estimator)'),
    (3, 'Term 2 (phi(s, a) estimator)'),
    (4, 'Term 2 (phi(s))'),
    (5, 'Term 3'),
      ]:
  plt.plot(episodes * 5/1000, np.log(np.clip(
      savgol_filter(variances[:, index], 21, 1), 1e-3, 1e6)), label=label)

plt.title('HalfCheetah-v1')
plt.ylabel('ln(Variance)')
plt.xlabel('Steps (millions)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('halfcheetah_variance_gae.pdf', bbox_inches='tight')




