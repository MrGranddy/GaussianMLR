#!/bin/sh

methods=("gaussian_mlr" "clr" "lsep")
datasets=("ranked_mnist_color" "ranked_mnist_gray" "architecture" "landscape")

for method in "${methods[@]}"
do
    for dataset in "${datasets[@]}"
    do
        python bar_plotter.py --dataset $dataset --method $method
    done
done