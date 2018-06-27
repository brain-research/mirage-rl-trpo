#!/bin/bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

for SEED in `seq 1 5`
do
  for ENV in Walker2d-v1 HalfCheetah-v1 Humanoid-v1
  do
    for BASELINE in v horizon_v time_v
    do
      berg run -i 'echo Starting' "mkdir -p /home/gjt/berg_results && python main.py --env-name $ENV --baseline ${BASELINE} --gamma 0.99 --tau 1.00 --batch-size 25000 --n-epochs 200 --max-time 1000 --seed $SEED --log-file /home/gjt/berg_results/${ENV}_baseline_${BASELINE}_seed_${SEED}.txt"
    done
  done
done
