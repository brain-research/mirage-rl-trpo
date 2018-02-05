import math
import os

env_batches = [5000]
for seed_idx, seed in enumerate([1234, 2345, 3456, 4567, 5678]):
  fname = 'run_i{}.sh'.format(seed_idx + 14)
  print('fname', fname)
  with open(fname, 'w') as f:
    for env_idx, env in enumerate(['HalfCheetah-v1', 'Walker2d-v1', 'Humanoid-v1']):
      for control_idx, control in enumerate(['none', 'v', 'q', 'disc']):
        log_fname = 'gcp3-logs/{}-{}-{}.json'.format(env[:-3].lower(), control, seed)
        if control == 'disc':
          line = "python main.py --env-name '{}' --gamma 0.99 --tau 0.95 --baseline none --n-epochs 3000 --max-time 1000 --seed {} --control-variate-lr 1e-3 --log-file {} --v-optimizer lbfgs --use-disc-avg-v --batch-size {} &\n".format(env, seed, log_fname, env_batches[0])
        else:
          line = "python main.py --env-name '{}' --gamma 0.99 --tau 0.95 --baseline {} --n-epochs 3000 --max-time 1000 --seed {} --control-variate-lr 1e-3 --log-file {} --v-optimizer lbfgs --batch-size {} &\n".format(env, control, seed, log_fname, env_batches[0])
        f.write(line)

