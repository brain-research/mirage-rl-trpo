#!/bin/bash

ENV="Walker2d-v2"
ENV="Humanoid-v2"
ENV="HalfCheetah-v2"

#for SEED in `seq 1 5`
#do
#  for BASELINE in q #none q model
#  do
#    for LR in 1e-2 1e-3 1e-1
#    do
#      python main.py --env-name $ENV --gamma 0.99 --tau 0.95 --baseline $BASELINE --n-epochs 70 --max-time 1000 --seed $SEED --control-variate-lr $LR --log-file logs/halfcheetah_q_lr/${BASELINE}.${LR}_${SEED}.txt
#    done
#  done
#done

# Baselines to try: 'none', 'v', 'vh0', 'vh1'
# Envs to try: 'HalfCheetah', 'Walker', Humanoid'
# Batch size: 25000
# Tau: 1.00

# for SEED in `seq 1 5`
for SEED in `seq 1 1`
do
  BASELINE=vh0
  # python main.py --env-name $ENV --gamma 0.99 --tau 0.95 --baseline $BASELINE --n-epochs 250 --max-time 1000 --seed $SEED --log-file logs/halfcheetah/${BASELINE}_disc_${SEED}.txt
  python main.py --env-name $ENV --gamma 0.99 --tau 1.00 --baseline $BASELINE --batch-size 25000 --n-epochs 50 --max-time 1000 --seed $SEED --log-file logs/${ENV}/${BASELINE}/sample.txt
done
