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
for i in $(seq 0 $((num_runs-1)))
do
    for method in "${methods[@]}"
    do
        for supervision in "${supervisions[@]}"
        do
            experiment_name="error_bar_landscape_"$backbone"_"$method"_"$supervision"_"$i
            echo "Running experiment $experiment_name"
            python new_metrics.py  --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --dataset $dataset --method $method
        done
    done
done

