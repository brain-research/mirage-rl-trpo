#!/bin/bash

ENV="HalfCheetah-v1"

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


for SEED in `seq 1 5`
do
  BASELINE=none
  python main.py --env-name $ENV --gamma 0.99 --tau 0.95 --baseline $BASELINE --n-epochs 250 --max-time 1000 --seed $SEED --log-file logs/halfcheetah/${BASELINE}_disc_${SEED}.txt
done
