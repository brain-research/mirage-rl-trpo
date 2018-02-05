#!/bin/bash

#for I in `seq 0 100 999` 995
for I in `seq 0 5 999`
do
  python calc_variance.py --checkpoint-dir tmp/ --checkpoint $I >> log_humanoid.txt
done

