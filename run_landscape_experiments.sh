#!/bin/sh

# Main dataset path
main_path="landscape_dataset"
num_epoch=200
backbone="resnet18"
dataset="landscape"

# List of methods
methods=("gaussian_mlr" "clr" "lsep")
supervisions=("strong" "weak")
config_path="A"

for method in "${methods[@]}"
do
    for supervision in "${supervisions[@]}"
    do
        experiment_name="landscape_"$backbone"_"$method"_"$supervision
        echo "Running experiment $experiment_name"

        # If method is gaussian_mlr
        if [ $method == "gaussian_mlr" ]; then
            python gaussian_mlr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision
        fi

        if [ $method == "clr" ]; then
            python clr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision
        fi

        if [ $method == "lsep" ]; then
            python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --stage "ranking"
            python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --stage "threshold"
        fi
    done
done
