import torch
import os
import numpy as np

from torch.utils.data import DataLoader
from reader import RankedMNISTReader
from model import GaussianModel, Model, LSEPModel

import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

import argparse

sqrt_two = np.sqrt(2)

# The probability of a Gaussian variable being positive
def gaussian_variable_positive_probability(z_mean, z_std):
    return 0.5 * (1 - torch.erf(-z_mean / (z_std * sqrt_two)))


device_name = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--supervision", type=str)
args = parser.parse_args()


backbone = args.backbone
method = args.method
supervision = args.supervision


ranked_mnist_path = "/mnt/disk2/interpolation_test_images/small_change"

def read_model(path):
    seq_path = os.path.join(path)
    ckpt = torch.load(seq_path)
    state_dict = ckpt["state_dict"]

    return state_dict


colors = ["#004D40", "#D81B60", "#1E88E5", "#FFC107"]
color_map = [colors[0]] + [colors[idx] for idx in range(1, 4)] + [colors[0]] * 6


if args.method == "lsep":
    path = "results/gray_small_scale_small_variance_%s_%s_%s/saves/threshold_best.pth" % (backbone, method, supervision)
else:
    path = "results/gray_small_scale_small_variance_%s_%s_%s/saves/best.pth" % (backbone, method, supervision)

if method == "gaussian_mlr":
    model = GaussianModel(10, backbone).to(device_name)
elif method == "clr":
    model = Model((11*10)//2, backbone).to(device_name)
elif method == "lsep":
    model = LSEPModel(10, backbone).to(device_name)

model.load_state_dict(torch.load(path, map_location=device_name)["state_dict"])
model = model.eval()

for param in model.parameters():
    model.requires_grad = False

all_scores = []

for dir_name in os.listdir(ranked_mnist_path):

    scores = []

    # Load images from directory
    images = []
    for file in os.listdir(os.path.join(ranked_mnist_path, dir_name)):
        if file.endswith(".png"):
            images.append(os.path.join(ranked_mnist_path, dir_name, file))
    images = sorted(images, key=lambda x: int(x.split(".")[0].split("/")[-1]))

    sel_digits = list(map(int, dir_name.split("/")[-1].split("_")[1:]))

    for t_idx, image_path in enumerate(images):

        image = TF.to_tensor(Image.open(image_path).convert("RGB")).to(device_name).unsqueeze(0) - 0.5
        mean, logvar = model(image)
        var = torch.exp(logvar)

        score = np.array(var.detach().cpu())[0, sel_digits]
        scores.append(score)

    scores = np.array(scores)
    all_scores.append(scores)

all_scores = np.array(all_scores)
scores = np.mean(all_scores, axis=0)

t = np.linspace(0.0, 1.0, len(images))

#plt.figure(figsize=(6, 3))
fig, ax = plt.subplots()
ax.set_box_aspect(1)

ax.plot(t, scores[:, 0], color="#D81B60", label="1st Digit", linewidth=4)
ax.plot(t, scores[:, 1], color="#1E88E5", label="2nd Digit", linewidth=4)
ax.plot(t, scores[:, 2], color="#FFC107", label="3rd Digit", linewidth=4)

ax.set_xlabel("t", fontsize=18, fontweight="heavy")
ax.set_ylabel("$\sigma^2$", fontsize=18, fontweight="heavy")
plt.xticks(fontsize=18, fontweight="heavy")
plt.yticks(fontsize=18, fontweight="heavy")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

plt.savefig("interpolation_var_test_results/small_change_%s_%s_%s.pdf" % (backbone, method, supervision), bbox_inches="tight")
