#!/bin/sh

# Main dataset path
main_path="/mnt/disk2/ranked_MNIST_family"
num_epoch=50
backbone="resnet18"
dataset="ranked_mnist"
method="gaussian_mlr"
supervision="strong"

# If supervision is strong then mode is ranked
if [ $supervision = "strong" ]; then
    mode="ranked"
# Else if supervision is weak then mode is unranked
else
    mode="unranked"
fi

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

# For each config path and experiment name run the experiment
for i in `seq 0 $((${#config_paths[@]} - 1))`; do
    config_path=${config_paths[$i]}
    experiment_name=${experiment_names[$i]}"_"$backbone"_"$method"_"$supervision
    echo "Running experiment $experiment_name"

    # If method is gaussian_mlr
    if [ $method == "gaussian_mlr" ]; then
        python gaussian_mlr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision
        python gaussian_mlr_calculate_metrics.py  --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --dataset $dataset --mode $mode
    fi

    if [ $method == "clr" ]; then
        python clr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision
        python clr_calculate_metrics.py  --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --dataset $dataset --mode $mode
    fi

    if [ $method == "lsep" ]; then
        python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --stage "ranking"
        python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --stage "threshold"
        python lsep_calculate_metrics.py  --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --dataset $dataset --mode $mode
    fi

done