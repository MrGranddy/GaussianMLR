datasets=("ranked_mnist_gray" "ranked_mnist_color" "landscape" "architecture")
methods=("gaussian_mlr" "clr" "lsep")

for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        python bar_plotter.py --dataset $dataset --method $method
    done
done

