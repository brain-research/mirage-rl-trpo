#!/bin/bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

DATE=2_23
for CHKPT in halfcheetah_${DATE}_normal_value humanoid_${DATE}_normal_value halfcheetah_${DATE}_disc_value humanoid_${DATE}_disc_value
do
  for I in `seq 0 100 999` 995
  do
    python calc_variance.py \
      --checkpoint-dir chkpts/$CHKPT \
      --checkpoint $I \
      --n-epochs 40 \
      --n-samples 50 \
      >> logs/variance/$CHKPT.txt
  done
done


