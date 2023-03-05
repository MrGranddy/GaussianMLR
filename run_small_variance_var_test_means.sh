#!/bin/sh

# Main dataset path
main_path="/mnt/disk2/ranked_MNIST_family"
num_epoch=50
backbone="resnet18"

# Save path
save_path="interpolation_pairwise_var_test_results"
# If does not exist, create
if [ ! -d "$save_path" ]; then
    mkdir $save_path
fi

methods=("gaussian_mlr")
supervisions=("strong" "weak")

# Run for loop for idxs in range(len(modes))
for method in "${methods[@]}"
do
    for supervision in "${supervisions[@]}"
    do
        echo "Small change pairwise experiment: $method, $supervision"
        python interpolation_var_test_mean_small_change.py --backbone $backbone --method $method --supervision $supervision 
    done
done