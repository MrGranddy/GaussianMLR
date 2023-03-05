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
            python new_metrics.py --config_path $config_path --main_path $main_path --experiment_name $experiment_name --backbone $backbone --dataset $dataset --method $method --domain $domain
        done
    done
done

