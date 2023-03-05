#!/bin/sh

# Main dataset path
main_path="/mnt/disk2/ranked_MNIST_family"
num_epoch=50
backbone="resnet18"
dataset="ranked_mnist"

methods=("gaussian_mlr" "clr" "lsep")
supervisions=("weak" "strong")

# Create empty list of config paths and experiment names
config_paths=()
experiment_names=()

# Add config paths and experiment names

# Gray Small Scale
config_paths+=("dataset_creation/configs/ranked_mnist_gray_small_scale_small_variance.json")
experiment_names+=("gray_small_scale_small_variance")

# For each config path and experiment name run the experiment

for method in "${methods[@]}"
do 
    for supervision in "${supervisions[@]}"
    do

        for i in `seq 0 $((${#config_paths[@]} - 1))`; do
            config_path=${config_paths[$i]}
            experiment_name=${experiment_names[$i]}"_"$backbone"_"$method"_"$supervision
            echo "Running experiment $experiment_name"

            python new_metrics.py  --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --dataset $dataset --method $method

        done
    done
done