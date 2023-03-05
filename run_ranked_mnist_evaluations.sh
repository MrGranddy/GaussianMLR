#!/bin/sh

# Main dataset path
main_path="/mnt/disk2/ranked_MNIST_family"
backbone="resnet18"
dataset="ranked_mnist"

# Create empty list of config paths and experiment names
config_paths=()
experiment_names=()

# Add config paths and experiment names

# Gray Small Scale
config_paths+=("dataset_creation/configs/ranked_mnist_gray_small_scale.json")
experiment_names+=("gray_small_scale")

# Color Small Scale
config_paths+=("dataset_creation/configs/ranked_mnist_color_small_scale.json")
experiment_names+=("color_small_scale")

# Gray Small Brightness
config_paths+=("dataset_creation/configs/ranked_mnist_gray_small_brightness.json")
experiment_names+=("gray_small_brightness")

# Color Small Brightness
config_paths+=("dataset_creation/configs/ranked_mnist_color_small_brightness.json")
experiment_names+=("color_small_brightness")

# Gray Small Brightness Scale -> Ratio
config_paths+=("dataset_creation/configs/ranked_mnist_gray_small_brightness_scale_ratio.json")
experiment_names+=("gray_small_brightness_scale_ratio")

# Gray Small Brightness Scale -> Brightness
config_paths+=("dataset_creation/configs/ranked_mnist_gray_small_brightness_scale_brightness.json")
experiment_names+=("gray_small_brightness_scale_brightness")

# Color Small Brightness Scale -> Ratio
config_paths+=("dataset_creation/configs/ranked_mnist_color_small_brightness_scale_ratio.json")
experiment_names+=("color_small_brightness_scale_ratio")

# Color Small Brightness Scale -> Brightness
config_paths+=("dataset_creation/configs/ranked_mnist_color_small_brightness_scale_brightness.json")
experiment_names+=("color_small_brightness_scale_brightness")

methods=("gaussian_mlr" "lsep" "clr")
supervisions=("strong" "weak")

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
