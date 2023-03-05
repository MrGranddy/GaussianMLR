import torch
import os
import numpy as np

from model import GaussianModel, Model, LSEPModel

import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

import argparse

device_name = "cuda:0"

mode = "gray" # or "color"
interpolate = "scale" # or "brightness"
randomize = "" # "scale" # or "brightness"
static = "brightness" # or "scale"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str)
parser.add_argument("--interpolate", type=str)
parser.add_argument("--randomize", type=str)
parser.add_argument("--static", type=str, default="simple")
parser.add_argument("--backbone", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--supervision", type=str)
args = parser.parse_args()

mode = args.mode
interpolate = args.interpolate

randomize = args.randomize if args.randomize != "None" else ""
static = args.static if args.static != "None" else ""

backbone = args.backbone
method = args.method
supervision = args.supervision


ranked_mnist_path = "/mnt/disk2/interpolation_test_images/%s_%s_%s_%s" % (mode, interpolate, randomize, static)

def read_model(path):
    seq_path = os.path.join(path)
    ckpt = torch.load(seq_path)
    state_dict = ckpt["state_dict"]

    return state_dict


colors = ["#004D40", "#D81B60", "#1E88E5", "#FFC107"]
color_map = [colors[0]] + [colors[idx] for idx in range(1, 4)] + [colors[0]] * 6

if randomize == "":
    if args.method == "lsep":
        path = "results/%s_small_%s_%s_%s_%s/saves/threshold_best.pth" % (mode, interpolate, backbone, method, supervision)
    else:
        path = "results/%s_small_%s_%s_%s_%s/saves/best.pth" % (mode, interpolate, backbone, method, supervision)
else:
    if interpolate == "brightness":
        _interpolate = "brightness"
    elif interpolate == "scale":
        _interpolate = "ratio"
    else:
        print("ERROR")
        exit()
    if args.method == "lsep":
        path = "results/%s_small_brightness_scale_%s_%s_%s_%s/saves/threshold_best.pth" % (mode, _interpolate, backbone, method, supervision)
    else:
        path = "results/%s_small_brightness_scale_%s_%s_%s_%s/saves/best.pth" % (mode, _interpolate, backbone, method, supervision)


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

    if method == "gaussian_mlr":
        for t_idx, image_path in enumerate(images):

            image = TF.to_tensor(Image.open(image_path).convert("RGB")).to(device_name).unsqueeze(0) - 0.5
            mean, logvar = model(image)
            mean[mean < 0] = 0.0
    
            score = np.array(mean.detach().cpu())[0, sel_digits]
            scores.append(score)

    elif method == "clr":
        for t_idx, image_path in enumerate(images):

            image = TF.to_tensor(Image.open(image_path).convert("RGB")).to(device_name).unsqueeze(0) - 0.5
            logits = model(image)
            probs = torch.sigmoid(logits)

            N, _ = probs.shape
            K = 11

            pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)]).to(device_name)
            left_scores = probs >= 0.5
            right_scores = probs < 0.5

            score_matrix = torch.zeros((N, K)).to(device_name)

            for j in range(K):
                score_matrix[:, j] += torch.sum(left_scores[:, pair_map[:, 0] == j] * probs[:, pair_map[:, 0] == j], dim=1)
                score_matrix[:, j] += torch.sum(right_scores[:, pair_map[:, 1] == j] * probs[:, pair_map[:, 1] == j], dim=1)

            negative_map = score_matrix < score_matrix[:, -1].unsqueeze(1).repeat(1, K)
            score_matrix[negative_map] = 0

            score = np.array(score_matrix.detach().cpu())[0, sel_digits]
            scores.append(score)
    elif method == "lsep":
        for t_idx, image_path in enumerate(images):

            image = TF.to_tensor(Image.open(image_path).convert("RGB")).to(device_name).unsqueeze(0) - 0.5
            score, thresholds = model(image)
            score[score < thresholds] = 0.0

            score = np.array(score.detach().cpu())[0, sel_digits]
            scores.append(score)

    scores = np.array(scores)
    all_scores.append(scores)

all_scores = np.array(all_scores)
scores = np.mean(all_scores, axis=0)

scores -= np.min(scores)
scores /= np.max(scores)

t = np.linspace(0.0, 1.0, len(images))

fig, ax = plt.subplots()
ax.set_box_aspect(1)
fig.tight_layout()
fig.figsize = (4, 4)

ax.plot(t, scores[:, 0], color="#D81B60", label="1st Digit", linewidth=4)
ax.plot(t, scores[:, 1], color="#1E88E5", label="2nd Digit", linewidth=4)
ax.plot(t, scores[:, 2], color="#FFC107", label="3rd Digit", linewidth=4)

ax.set_xlabel("t", fontsize=18, fontweight="heavy")
ax.set_ylabel("Scores", fontsize=18, fontweight="heavy")
plt.xticks(fontsize=18, fontweight="heavy")
plt.yticks(fontsize=18, fontweight="heavy")

plt.savefig("interpolation_test_results/%s_%s_%s_%s_%s_%s_%s.pdf" % (mode, interpolate, randomize, static, backbone, method, supervision), bbox_inches="tight")
