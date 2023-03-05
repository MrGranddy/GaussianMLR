#!/bin/sh

# Main dataset path
backbone="resnet18"

# List of methods
methods=("gaussian_mlr" "clr" "lsep")
supervisions=("strong" "weak")
modes=("scale" "brightness")

num_digits=(4)
config_path="A"

# Add config paths and experiment names


for mode in "${modes[@]}"
do

    # Save path
    save_path="calibration_test_"$mode"_results"
    # If does not exist, create
    if [ ! -d "$save_path" ]; then
        mkdir $save_path
    fi

    for num_digit in ${num_digits[@]}
    do
        for method in "${methods[@]}"
        do
            for supervision in "${supervisions[@]}"
            do
                echo "Running experiment $num_digit $method $supervision"
                python calibration_test_mean.py --backbone $backbone --method $method --supervision $supervision --num_digits $num_digit --mode $mode
            done
        done
    done
    
done