import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torchvision.transforms.functional as TF
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

from model import GaussianModel, LSEPModel, Model

device_name = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--supervision", type=str)
parser.add_argument("--num_digits", type=int)
parser.add_argument("--mode", type=str)
args = parser.parse_args()

backbone = args.backbone
method = args.method
supervision = args.supervision
num_digits = args.num_digits

ranked_mnist_path = "/mnt/disk2/calibration_%s_test_%d_images/" % (
    args.mode,
    num_digits,
)


def read_model(path):
    seq_path = os.path.join(path)
    ckpt = torch.load(seq_path)
    state_dict = ckpt["state_dict"]

    return state_dict


colors = ["#004D40", "#D81B60", "#1E88E5", "#FFC107"]
color_map = [colors[0]] + [colors[idx] for idx in range(1, 4)] + [colors[0]] * 6


if args.method == "lsep":
    path = "results/gray_small_%s_%s_%s_%s/saves/threshold_best.pth" % (
        args.mode,
        backbone,
        method,
        supervision,
    )
else:
    path = "results/gray_small_%s_%s_%s_%s/saves/best.pth" % (
        args.mode,
        backbone,
        method,
        supervision,
    )


if method == "gaussian_mlr":
    model = GaussianModel(10, backbone).to(device_name)
elif method == "clr":
    model = Model((11 * 10) // 2, backbone).to(device_name)
elif method == "lsep":
    model = LSEPModel(10, backbone).to(device_name)

model.load_state_dict(torch.load(path, map_location=device_name)["state_dict"])
model = model.eval()

for param in model.parameters():
    model.requires_grad = False

all_scores = []

for img_name in os.listdir(ranked_mnist_path):

    img_path = os.path.join(ranked_mnist_path, img_name)

    image = (
        TF.to_tensor(Image.open(img_path).convert("RGB")).to(device_name).unsqueeze(0)
        - 0.5
    )

    sel_digits = list(map(int, img_name.split(".")[0].split("_")[1:]))

    if method == "gaussian_mlr":
        mean, logvar = model(image)
        mean[mean < 0] = 0.0

        score = np.array(mean.detach().cpu())[0, sel_digits]

    elif method == "gaussian_mlr":
        mean, logvar = model(image)
        mean[mean < mean[:, -1]] = 0.0

        score = np.array(mean.detach().cpu())[0, sel_digits]

    elif method == "clr":
        logits = model(image)
        probs = torch.sigmoid(logits)

        N, _ = probs.shape
        K = 11

        pair_map = torch.tensor(
            [(i, j) for i in range(K - 1) for j in range(i + 1, K)]
        ).to(device_name)
        left_scores = probs >= 0.5
        right_scores = probs < 0.5

        score_matrix = torch.zeros((N, K)).to(device_name)

        for j in range(K):
            score_matrix[:, j] += torch.sum(
                left_scores[:, pair_map[:, 0] == j] * probs[:, pair_map[:, 0] == j],
                dim=1,
            )
            score_matrix[:, j] += torch.sum(
                right_scores[:, pair_map[:, 1] == j] * probs[:, pair_map[:, 1] == j],
                dim=1,
            )

        negative_map = score_matrix < score_matrix[:, -1].unsqueeze(1).repeat(1, K)
        score_matrix[negative_map] = 0

        score = np.array(score_matrix.detach().cpu())[0, sel_digits]

    elif method == "lsep":
        score, thresholds = model(image)
        score[score < thresholds] = 0.0

        score = np.array(score.detach().cpu())[0, sel_digits]

    all_scores.append(score)

all_scores = np.array(all_scores)
score_means = np.mean(all_scores, axis=0)
score_stds = np.std(all_scores, axis=0)

plt.figure(figsize=(8, 4))
plt.tight_layout()

max_prob = 0

for i in range(num_digits):

    x_lim = score_means[i] + score_stds[i] * 3
    x_range = np.linspace(0, x_lim, 100)
    y_range = stats.norm.pdf(x_range, score_means[i], score_stds[i])
    if np.max(y_range) > max_prob:
        max_prob = np.max(y_range)

    plt.plot(
        x_range,
        y_range,
        color=color_map[i],
        label="Scale=%.1f" % (1 + i * 0.5),
        linewidth=4,
    )

plt.xlabel("x", fontsize=18, fontweight="heavy")
plt.ylabel("p(x)", fontsize=18, fontweight="heavy")
plt.xticks(fontsize=18, fontweight="heavy")
plt.yticks(
    fontsize=18,
    fontweight="heavy",
)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


plt.xlim(0, score_means[-1] + score_stds[-1] * 3)
plt.ylim(0, max_prob)

plt.savefig(
    os.path.join(
        "calibration_test_%s_results" % args.mode,
        "%s_%s_%s_%d.pdf" % (backbone, method, supervision, num_digits),
    ),
    bbox_inches="tight",
)
