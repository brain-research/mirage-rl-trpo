#!/bin/bash

for I in `seq 0 100 999` 995
do
  python calc_variance.py --checkpoint-dir chkpts/ --checkpoint $I >> log_cheetah2.txt
done

