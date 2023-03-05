#!/bin/sh

# Main dataset path
main_path="/mnt/disk2/ranked_MNIST_family"
num_epoch=50
backbone="resnet18"
dataset="ranked_mnist"

#methods=("lsep" "clr" "gaussian_mlr")
#methods=("gaussian_mlr")
#methods=("clr")
methods=("lsep")
supervisions=("weak" "strong")
num_runs=5

# Create empty list of config paths and experiment names
config_paths=()
experiment_names=()

# Add config paths and experiment names

# Gray Small Scale
config_paths+=("dataset_creation/configs/ranked_mnist_gray_small_scale.json")
experiment_names+=("gray_small_scale")

# Gray Small Brightness
config_paths+=("dataset_creation/configs/ranked_mnist_gray_small_brightness.json")
experiment_names+=("gray_small_brightness")


# for all methods and supervisions and num_runs and configs
for method in "${methods[@]}"
do
    for supervision in "${supervisions[@]}"
    do
        for j in $(seq 0 $((num_runs-1)))
        do
            for i in `seq 0 $((${#config_paths[@]} - 1))`
            do
                config_path=${config_paths[$i]}
                experiment_name="error_bar_"${experiment_names[$i]}"_"$backbone"_"$method"_"$supervision"_"$j
                echo "Running experiment $experiment_name"

                # If method is gaussian_mlr
                if [ $method == "gaussian_mlr" ]; then
                    python gaussian_mlr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --subset 1
                fi

                if [ $method == "clr" ]; then
                    python clr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --subset 1
                fi

                if [ $method == "lsep" ]; then
                    python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --stage "ranking" --subset 1
                    python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --stage "threshold" --subset 1
                fi

            done
        done
    done
done
