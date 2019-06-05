#!/usr/bin/env bash

set -x

NUM_ITER=100
echo "Start CNN Neural Network training ..."

python cnn.py --max_iter $NUM_ITER --experiment_name "iter=${NUM_ITER}"
echo "Done"

