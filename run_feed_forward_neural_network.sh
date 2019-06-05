#!/usr/bin/env bash

set -x

NUM_ITER=100
echo "Start Neural Network training ..."

python feed_forward_neural_network.py --max_iter $NUM_ITER --experiment_name "iter=${NUM_ITER}"
echo "Done"

