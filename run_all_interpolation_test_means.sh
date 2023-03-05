#!/bin/sh

# Main dataset path
main_path="/mnt/disk2/ranked_MNIST_family"
num_epoch=50
backbone="resnet18"

# Save path
save_path="interpolation_test_results"
# If does not exist, create
if [ ! -d "$save_path" ]; then
    mkdir $save_path
fi

# Create run setups
modes=(         "gray"       "color"      "gray"       "color"      "gray"       "gray"       "color"       "color")
interpolations=("scale"      "scale"      "brightness" "brightness" "scale"      "brightness" "scale"       "brightness")
randomize=(     "None"       "None"       "None"       "None"       "brightness" "scale"      "brightness"  "scale")
static=(        "brightness" "brightness" "scale"      "scale"      "None"       "None"       "None"        "None")

#modes=(          "gray"         "gray"        )
#interpolations=( "scale"        "brightness"  )
#randomize=(      "None"         "None"        )
#static=(         "brightness"   "scale"       )


methods=("gaussian_mlr" "clr" "lsep")
supervisions=("strong" "weak")

# Run for loop for idxs in range(len(modes))
for idx in $(seq 0 $((${#modes[@]}-1)))
do
    for method in "${methods[@]}"
    do
        for supervision in "${supervisions[@]}"
        do
                mode=${modes[$idx]}
                interpolate=${interpolations[$idx]}
                randomize=${randomize[$idx]}
                static=${static[$idx]}

                echo "mode=$mode, interpolate=$interpolate, randomize=$randomize, static=$static"
                python interpolation_test_mean.py --mode $mode --interpolate $interpolate --randomize $randomize --static $static --backbone $backbone --method $method --supervision $supervision
                
        done
    done
done