#!/bin/bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Plots with the normal value function
python plot_variances.py logs/variance/halfcheetah_2_23_normal_value.txt HalfCheetah-v1
python plot_variances.py logs/variance/humanoid_2_23_normal_value.txt Humanoid-v1

# Plots with horizon aware value function
python plot_variances.py logs/variance/halfcheetah_2_23_disc_value.txt HalfCheetah-v1-disc
python plot_variances.py logs/variance/humanoid_2_23_disc_value.txt Humanoid-v1-disc
