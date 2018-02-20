#!/bin/bash

SEED=2
DATE=2_16

python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline none --n-epochs 1000 --max-time 1000 --seed ${SEED} --control-variate-lr 1e-3 --v-optimizer lbfgs --batch-size 4900 --checkpoint-dir chkpts/halfcheetah_${DATE}_normal_value

python main.py --env-name "Humanoid-v1" --gamma 0.99 --tau 0.95 --baseline none --n-epochs 1000 --max-time 1000 --seed ${SEED} --control-variate-lr 1e-3 --v-optimizer lbfgs --batch-size 4900 --checkpoint-dir chkpts/humanoid_${DATE}_normal_value

# Use new value func
python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline none --n-epochs 1000 --max-time 1000 --seed ${SEED} --control-variate-lr 1e-3 --v-optimizer lbfgs --batch-size 4900 --checkpoint-dir chkpts/halfcheetah_${DATE}_disc_value --use-disc-avg-v

python main.py --env-name "Humanoid-v1" --gamma 0.99 --tau 0.95 --baseline none --n-epochs 1000 --max-time 1000 --seed ${SEED} --control-variate-lr 1e-3 --v-optimizer lbfgs --batch-size 4900 --checkpoint-dir chkpts/humanoid_${DATE}_disc_value --use-disc-avg-v
