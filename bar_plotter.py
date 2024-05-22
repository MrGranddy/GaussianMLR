import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from model import GaussianModel, LSEPModel, Model

bs = 64
device_name = "cuda:1"
backbone = "resnet18"

ROTATION = 20

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--method", type=str)
args = parser.parse_args()

if args.dataset == "ranked_mnist_color":
    n_classes = 10
    save_name = "color_small_scale_resnet18_%s_strong" % args.method
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
elif args.dataset == "ranked_mnist_gray":
    n_classes = 10
    save_name = "gray_small_scale_resnet18_%s_strong" % args.method
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
elif args.dataset == "landscape":
    n_classes = 9
    save_name = "landscape_resnet18_%s_strong" % args.method
    classes = [
        "plant",
        "sky",
        "cloud",
        "snow",
        "building",
        "desert",
        "mountain",
        "water",
        "sun",
    ]
elif args.dataset == "architecture":
    n_classes = 9
    save_name = "architecture_ARC_resnet18_%s_strong" % args.method
    classes = ["asym", "clr", "crys", "flow", "iso", "prog", "reg", "shp", "sym"]

dataset_path = "bar_plots/%s" % args.dataset

image_paths = [os.path.join(dataset_path, path) for path in os.listdir(dataset_path)]

if args.method == "gaussian_mlr":
    model = GaussianModel(n_classes, backbone).to(device_name)
    best_path = "results/%s/saves/best.pth" % save_name

elif args.method == "clr":
    n_classes += 1  # Add virtual label
    model = Model((n_classes * (n_classes - 1)) // 2, backbone).to(device_name)
    best_path = "results/%s/saves/best.pth" % save_name

elif args.method == "lsep":
    model = LSEPModel(n_classes, backbone).to(device_name)
    best_path = "results/%s/saves/threshold_best.pth" % save_name

model.load_state_dict(torch.load(best_path, map_location=device_name)["state_dict"])
model = model.eval()
for param in model.parameters():
    model.requires_grad = False

if args.dataset == "ranked_mnist_color":
    MEAN = [0.5, 0.5, 0.5]
    STD = [1.0, 1.0, 1.0]
elif args.dataset == "ranked_mnist_gray":
    MEAN = [0.5, 0.5, 0.5]
    STD = [1.0, 1.0, 1.0]
elif args.dataset == "landscape":
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
elif args.dataset == "architecture":
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

MEAN = torch.tensor(MEAN).reshape(3, 1, 1)
STD = torch.tensor(STD).reshape(3, 1, 1)

all_scores = []
all_thresholds = []

with torch.no_grad():
    for image_path in image_paths:

        image = (
            TF.pil_to_tensor(
                Image.open(image_path).convert("RGB").resize((224, 224))
            ).float()
            / 255.0
        )
        image = (image - MEAN) / STD
        image = image.unsqueeze(0).to(device_name)

        if args.method == "gaussian_mlr":
            mean, logvar = model(image)
            all_thresholds.append(0.0)
            scores = mean

        elif args.method == "clr":
            logits = model(image)
            probs = torch.sigmoid(logits)

            pair_map = torch.tensor(
                [(i, j) for i in range(n_classes - 1) for j in range(i + 1, n_classes)]
            ).to(device_name)
            left_scores = probs >= 0.5
            right_scores = probs < 0.5

            score_matrix = torch.zeros((1, n_classes)).to(device_name)

            for j in range(n_classes):
                score_matrix[:, j] += torch.sum(
                    left_scores[:, pair_map[:, 0] == j] * probs[:, pair_map[:, 0] == j],
                    dim=1,
                )
                score_matrix[:, j] += torch.sum(
                    right_scores[:, pair_map[:, 1] == j]
                    * probs[:, pair_map[:, 1] == j],
                    dim=1,
                )

            all_thresholds.append(score_matrix[0, -1].item())
            scores = score_matrix[:, :-1]

        elif args.method == "lsep":
            scores, thresholds = model(image)
            all_thresholds.append(thresholds[0].cpu().detach().numpy())

        scores = scores.cpu().detach().numpy()
        all_scores.append(scores[0])

if args.method == "clr":
    n_classes -= 1


all_scores = np.array(all_scores)
x = np.arange(n_classes)

for i in range(len(image_paths)):

    fig, ax = plt.subplots()

    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.figsize = (4, 4)

    ax.bar(
        x[all_scores[i] >= all_thresholds[i]],
        all_scores[i][all_scores[i] >= all_thresholds[i]],
        width=0.5,
        color="green",
    )
    ax.bar(
        x[all_scores[i] < all_thresholds[i]],
        all_scores[i][all_scores[i] < all_thresholds[i]],
        width=0.5,
        color="red",
    )

    if args.method == "clr":
        ax.bar(n_classes, all_thresholds[i], width=0.5, color="purple")

    x_lim_max = np.max(all_scores[i]) * 1.1
    ax.set_ylim(-x_lim_max, x_lim_max)

    if args.method == "clr":
        ax.set_xticks(
            x.tolist() + [n_classes], classes + ["vl"], rotation=ROTATION, ha="right"
        )
        ax.plot([0.0, n_classes + 0.5], [0.0, 0.0], color="black", linestyle="-")
    else:
        ax.set_xticks(x.tolist(), classes, rotation=ROTATION, ha="right")

    if args.method == "lsep":
        for j in range(len(all_thresholds[i])):
            ax.plot(
                [float(j) - 0.25, float(j) + 0.25],
                [all_thresholds[i][j], all_thresholds[i][j]],
                color="purple",
                linestyle="-",
                linewidth=4,
            )
            pass
        ax.plot([0.0, n_classes], [0.0, 0.0], color="black", linestyle="-")
    else:
        ax.plot(
            [0.0, n_classes],
            [all_thresholds[i], all_thresholds[i]],
            color="purple",
            linestyle="--",
        )

    ax.set_ylabel("Scores", fontsize=18, fontweight="heavy")
    plt.xticks(fontsize=12, fontweight="heavy")
    plt.yticks(fontsize=12, fontweight="heavy")

    plt.savefig(
        "bar_plots/%s_%s.pdf"
        % (args.method, image_paths[i].split("/")[-1].split(".")[0]),
        bbox_inches="tight",
    )
    plt.close()
