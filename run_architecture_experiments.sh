#!/bin/sh

# Main dataset path
main_path="/mnt/disk1/mimarlik-multilabel/dataset"
num_epoch=100
backbone="resnet18"
dataset="architecture"

methods=("gaussian_mlr" "clr" "lsep")
supervisions=("strong" "weak")
domains=("ARC")
config_path="A"

for domain in "${domains[@]}"
do
    for method in "${methods[@]}"
    do
        for supervision in "${supervisions[@]}"
        do
            experiment_name="architecture_"$domain"_"$backbone"_"$method"_"$supervision
            echo "Running experiment $experiment_name"

            if [ $method == "gaussian_mlr" ]; then
                python gaussian_mlr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --domain $domain
            fi

            if [ $method == "clr" ]; then
                python clr_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --domain $domain
            fi

            if [ $method == "lsep" ]; then
                python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --domain $domain --stage "ranking"
                python lsep_trainer.py --config_path $config_path --main_path $main_path --num_epoch $num_epoch --experiment_name $experiment_name --backbone $backbone --dataset $dataset --supervision $supervision --domain $domain --stage "threshold"
            fi

        done
    done
done
