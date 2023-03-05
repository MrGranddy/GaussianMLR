#!/bin/sh

# Main dataset path
main_path="landscape_dataset"
backbone="resnet18"
dataset="landscape"
num_runs=5

# List of methods
methods=("gaussian_mlr" "clr" "lsep")
supervisions=("strong" "weak")
config_path="A"

# Add config paths and experiment names

# for num_runs
for method in "${methods[@]}"
do
    for supervision in "${supervisions[@]}"
    do
        experiment_name="error_bar_landscape_"$backbone"_"$method"_"$supervision
        echo "Running experiment $experiment_name"
        python make_error_bar.py  --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --dataset $dataset --method $method --num_runs $num_runs
    done
done

