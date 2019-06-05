#!/usr/bin/env bash

set -x

MAX_ITER=100
echo "Start Transfer Learning Neural Network training ..."

echo "Transfer learning with MobileNet"
python3 transfer_learning.py --max_iter=$MAX_ITER  --transfer_model=MobileNet --experiment_name="transfer_learning_with_MobileNet_iter=${MAX_ITER}" "$@"

echo "Transfer learning with ResNet50"
python3 transfer_learning.py --max_iter=$MAX_ITER --transfer_model=ResNet50 --experiment_name="transfer_learning_with_ResNet50_iter=${MAX_ITER}"  "$@"

echo "Transfer learning with VGG19"
python3 transfer_learning.py --max_iter=$MAX_ITER --transfer_model=VGG19 --experiment_name="transfer_learning_with_VGG19_iter=${MAX_ITER}" "$@"

echo "Transfer learning with InceptionV3"
python3 transfer_learning.py --max_iter=$MAX_ITER  --transfer_model=InceptionV3 --experiment_name="transfer_learning_with_InceptionV3_iter=${MAX_ITER}_tunesourcemodel=true" --tune_source_model=True "$@"


# All done
echo "All done!"
