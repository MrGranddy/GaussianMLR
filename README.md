>GaussianMLR is a unified multi-label ranking method solving both multi classification and label ranking problems together. The repo also provides Ranked MNIST dataset creating codes and two novel multi-label ranking methods CRPC (Strong) and LSEP (Strong) which utilizes our novel paradigm.

# GaussianMLR: Learning Implicit Class Significance via Calibrated Multi-Label Ranking

Access to paper provided at [https://arxiv.org/abs/2303.03907](https://arxiv.org/abs/2303.03907)

## Requirements

We use Torch 1.10.0, Numpy 1.21.2, Matplotlib 3.5.0, SciPy 1.7.2, Pillow 8.4.0, Torchvision 0.11.1, SkLearn 1.0.2.

To create the Ranked MNIST dataset you need the original MNIST digits, http://yann.lecun.com/exdb/mnist/, two source directories should be provided MNIST-train and MNIST-test, in which there are 10 folders with each of the digits' names 0, 1, 2 ..., 9. Then the digits should be converted into RGB PNG format and saved with unique names into the according folders. Then after changing the source paths in the dataset_creation folders you can create Ranked MNIST datasets.

We also provide our preprocessed version of the digits https://doi.org/10.5281/zenodo.6585131 

## Training and Evaluation

To train the models one should use the training scripts:

```train
bash run_ranked_mnist_experiments.sh
```
```train
bash run_landscape_experiments.sh
```
```train
bash run_architecture_experiments.sh
```

The script have the necessary parameters inside, after training to create an evaluation file you can use the same script with "evaluations" instead of "experiments".

## Experiments

To run the experiments conducted in the paper you can use the corresponding experiment script.

"Calibration" is for 5.6. Calibration Experiment
"interpolation" is for 5.5 Adjusting Significance Effects Experiment
To run the above experiments you should create the necessary data using the scripts in "dataset_creation/interpolation_test_set.py", "dataset_creation/calibration_test_set.py", 
"dataset_creation/create_brightness_calibration_test_set.py",
"dataset_creation/create_interpolation_test_set_small_change.py"

"visual_interpolation" is for 5.7 Extracted Significance Value Experiment
and "interpolation_var" is for Appendix §10

## TL; DR

"loss.py", "model.py", "*_trainer.py", "loss.py", "new_metrics.py", "reader.py", "utils.py" and the "dataset_creation" directory are all you need to train and evalutate the methods provided in our paper. "*.sh" script are there to easily run multiple experiments.

## Pre-trained Weights

The pre-trained weights can be found in the link:
https://doi.org/10.5281/zenodo.6585212
