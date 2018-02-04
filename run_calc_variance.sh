#!/bin/bash

for I in `seq 0 5 1000`
do
  python calc_variance.py --checkpoint-dir chkpts/ --checkpoint $I >> log.txt
done

