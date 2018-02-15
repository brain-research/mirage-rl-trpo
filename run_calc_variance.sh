#!/bin/bash

#for I in `seq 0 100 999` 995
for I in `seq 0 100 999` 995
do
  python calc_variance.py --checkpoint-dir halfcheetah_2_6/ --checkpoint $I --n-epochs 40 --n-samples 50 >> log_halfcheetah_2_6.txt
done

