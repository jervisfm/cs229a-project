#!/usr/bin/env bash

set -x

NUM_ITER=100
echo "Start SVM trainings ..."
echo ">>> Training SVM RBF ..."
python svm.py --max_iter $NUM_ITER --kernel "rbf"  --experiment_name "kernel=rbf_iter=${NUM_ITER}"
echo "Done"


echo ">>> Training SVM linear ..."
python svm.py --max_iter $NUM_ITER --kernel "linear"  --experiment_name "kernel=linear_iter=${NUM_ITER}"
echo "Done"


echo ">>> Training SVM poly ..."
python svm.py --max_iter $NUM_ITER --kernel "poly"  --experiment_name "kernel=poly_iter=${NUM_ITER}"
echo "Done"

echo ">>> Training SVM Sigmoid ..."
python svm.py --max_iter $NUM_ITER --kernel "sigmoid"  --experiment_name "kernel=sigmoid_iter=${NUM_ITER}"
echo "Done"


