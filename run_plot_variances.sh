#!/bin/bash

# Plots with the normal value function
python plot_variances.py logs/variance/halfcheetah_2_15_normal_value.txt HalfCheetah-v1
python plot_variances.py logs/variance/humanoid_2_16_normal_value.txt Humanoid-v1

# Plots with horizon aware value function
python plot_variances.py logs/variance/halfcheetah_2_16_disc_value.txt HalfCheetah-v1-disc
python plot_variances.py logs/variance/humanoid_2_16_disc_value.txt Humanoid-v1-disc
