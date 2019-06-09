#!/usr/bin/env bash

set -x

NUM_ITER=100
echo "Start CNN Neural Network hyperparameter for l2 regularization ..."


for REG in 0.001 0.01 0.05 0.1
		do
			# # Resnet
			echo ">>> Training model with l2_reg=$REG ..."
			python cnn.py --max_iter $NUM_ITER --l2_reg=$REG --experiment_name "iter=${NUM_ITER}_l2reg=${REG}"
			echo "Done"
		done

echo "All Done"

