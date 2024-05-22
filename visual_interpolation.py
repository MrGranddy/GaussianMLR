import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import GaussianModel, LSEPModel, Model
from reader import LandscapeReader

class_names = [
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

bs = 64
device_name = "cuda:1"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--main_path", type=str)
parser.add_argument("--backbone", type=str, default="simple")
parser.add_argument("--method", type=str)
parser.add_argument("--domain", type=str)
args = parser.parse_args()


val_loader = DataLoader(
    LandscapeReader(args.main_path, "test"),
    batch_size=bs,
    shuffle=False,
    num_workers=8,
)

n_classes = 9

if args.method == "gaussian_mlr":
    model = GaussianModel(n_classes, args.backbone).to(device_name)
    best_path = "results/%s/saves/best.pth" % args.experiment_name

elif args.method == "clr":
    n_classes += 1  # Add virtual label
    model = Model((n_classes * (n_classes - 1)) // 2, args.backbone).to(device_name)
    best_path = "results/%s/saves/best.pth" % args.experiment_name

elif args.method == "lsep":
    model = LSEPModel(n_classes, args.backbone).to(device_name)
    best_path = "results/%s/saves/threshold_best.pth" % args.experiment_name

model.load_state_dict(torch.load(best_path, map_location=device_name)["state_dict"])
model = model.eval()
for param in model.parameters():
    model.requires_grad = False

all_scores = []
all_paths = []

with torch.no_grad():
    for batch in val_loader:

        images = batch[0].to(device_name)
        labels = batch[1].to(device_name)
        paths = batch[2]

        N, K = labels.shape

        if args.method == "gaussian_mlr":
            mean, logvar = model(images)
            mean[mean < 0] = 0.0
            scores = mean

        elif args.method == "clr":
            K += 1
            logits = model(images)
            probs = torch.sigmoid(logits)

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
                    right_scores[:, pair_map[:, 1] == j]
                    * probs[:, pair_map[:, 1] == j],
                    dim=1,
                )

            negative_map = score_matrix < score_matrix[:, -1].unsqueeze(1).repeat(1, K)
            score_matrix[negative_map] = 0

            scores = score_matrix[:, :-1]

        elif args.method == "lsep":
            scores, thresholds = model(images)
            scores[scores < thresholds] = 0.0

        labels = labels.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        all_scores.append(scores)
        all_paths += [*paths]

if args.method == "clr":
    n_classes -= 1  # Remove virtual label

all_scores = np.concatenate(all_scores, axis=0)
# class_mins = np.min(all_scores, axis=0)
# class_maxs = np.max(all_scores, axis=0)
# class_mins = np.array([np.min(all_scores[:, i][all_scores[:, i] != class_mins[i]]) for i in range(n_classes)]) # Start from second min
# class_maxs = np.array([np.max(all_scores[:, i][all_scores[:, i] != class_maxs[i]]) for i in range(n_classes)]) # Start from second max
# class_interpolations = np.linspace(class_mins, class_maxs, 10)

method_path = "visual_interpolation_test_results/%s" % (args.experiment_name)
os.makedirs(method_path, exist_ok=True)
for cidx, class_name in enumerate(class_names):
    class_path = os.path.join(method_path, class_name + ".png")

    sorted_idxs = np.argsort(all_scores[:, cidx])
    sorted_paths = [all_paths[i] for i in sorted_idxs]
    sorted_scores = all_scores[sorted_idxs, cidx]
    first_nonzero = np.where(sorted_scores != 0)[0][0]

    selecteds = np.linspace(first_nonzero, all_scores.shape[0] - 1, 10).astype("int32")
    selected_paths = [[sorted_paths[i + j] for i in selecteds] for j in range(1)]

    imgs = [
        [np.array(Image.open(col).resize((224, 224)).convert("RGB")) for col in row]
        for row in selected_paths
    ]
    all_img = np.concatenate([np.concatenate(row, axis=1) for row in imgs], axis=0)
    print(all_img.shape)

    Image.fromarray(all_img).save(class_path)
