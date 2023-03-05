#!/bin/sh

# Main dataset path
main_path="landscape_dataset"
backbone="resnet18"
dataset="landscape"

# List of methods
methods=("gaussian_mlr" "clr" "lsep")
supervisions=("strong" "weak")
config_path="A"

# Save path
save_path="visual_interpolation_test_results"
# If does not exist, create
if [ ! -d "$save_path" ]; then
    mkdir $save_path
fi


# Add config paths and experiment names

for method in "${methods[@]}"
do
    for supervision in "${supervisions[@]}"
    do
        experiment_name="landscape_"$backbone"_"$method"_"$supervision
        echo "Running experiment $experiment_name"
        python visual_interpolation.py  --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --method $method
    done
done

