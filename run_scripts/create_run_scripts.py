import math
import os

run_id = 1
env_batches = [25000, 50000]
for idx, env in enumerate(['Walker2d-v1', 'Humanoid-v1']):
  for jdx, control in enumerate(['none', 'v', 'q', 'model']):

    fname = 'run_i{}.sh'.format(run_id)
    run_id += 1
    print('fname', fname)
    with open(fname, 'w') as f:
      for kdx, seed in enumerate([1000, 1111, 2222, 3333, 4444]):
        log_fname = 'gcp2-logs/{}-{}-{}_{}.json'.format(env[:-3].lower(), control, seed, kdx)
        line = "python main.py --env-name '{}' --gamma 0.99 --tau 0.95 --baseline {} --n-epochs 3000 --max-time 1000 --seed {} --control-variate-lr 1e-3 --log-file {} --v-optimizer lbfgs --use-disc-avg-v --batch-size {} &\n".format(env, control, seed, log_fname, env_batches[idx])
        # print(line)
        f.write(line)

    fname = 'run_i{}.sh'.format(run_id)
    run_id += 1
    print('fname', fname)
    with open(fname, 'w') as f:
      for kdx, seed in enumerate([5555, 6666, 7777, 8888, 9999]):
        log_fname = 'gcp2-logs/{}-{}-{}_{}.json'.format(env[:-3].lower(), control, seed, kdx)
        line = "python main.py --env-name '{}' --gamma 0.99 --tau 0.95 --baseline {} --n-epochs 3000 --max-time 1000 --seed {} --control-variate-lr 1e-3 --log-file {} --v-optimizer lbfgs --use-disc-avg-v --batch-size {} &\n".format(env, control, seed, log_fname, env_batches[idx])
        # print(line)
        f.write(line)


