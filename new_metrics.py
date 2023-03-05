import torch
from torch.utils.data import DataLoader

import os

import numpy as np

from model import GaussianModel, Model, LSEPModel
from reader import RankedMNISTReader, LandscapeReader, ArchitectureReader

from scipy.stats import spearmanr
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_loss, label_ranking_average_precision_score

import argparse

def ranking_metrics(scores, labels):

    N, K = labels.shape

    pair_map = np.array([(i, j) for i in range(K - 1) for j in range(i + 1, K)])
    n0 = K * (K - 1) / 2 # Number of pairs
    
    full_tie_index = np.where( scores.sum(1) != 0 )[0]
    scores = scores[full_tie_index, :]
    labels = labels[full_tie_index, :]

    score_greater = ( scores[:, pair_map[:, 0]] > scores[:, pair_map[:, 1]]  )
    score_smaller = ( scores[:, pair_map[:, 0]] < scores[:, pair_map[:, 1]]  )
    score_equals =  ( scores[:, pair_map[:, 0]] == scores[:, pair_map[:, 1]] )

    label_greater = ( labels[:, pair_map[:, 0]] > labels[:, pair_map[:, 1]]  )
    label_smaller = ( labels[:, pair_map[:, 0]] < labels[:, pair_map[:, 1]]  )
    label_equals =  ( labels[:, pair_map[:, 0]] == labels[:, pair_map[:, 1]] )

    n1 = score_equals.sum(1).astype("float32") # Number of tied pairs in scores
    n2 = label_equals.sum(1).astype("float32") # Number of tied pairs in labels

    nc = (
        ((score_greater == 1) * (label_greater == 1)).astype("int32") +
        ((score_smaller == 1) * (label_smaller == 1)).astype("int32")
    ).sum(1).astype("float32") # Number of concordant pairs
    nd = (
        ((score_greater == 1) * (label_smaller == 1)).astype("int32") +
        ((score_smaller == 1) * (label_greater == 1)).astype("int32")
    ).sum(1).astype("float32") # Number of discordant pairs


    # Kendall's Tau-a
    # https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    tau_a = np.sum( (nc - nd) / n0 ) / N

    # Kendall's Tau-b
    # https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    tau_b = np.sum( (nc - nd) / np.sqrt( (n0 - n1) * (n0 - n2) ) ) / N

    # Spearman's Rho
    # https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

    spearman_rho = np.sum( [spearmanr(scores[i], labels[i], axis=1)[0] for i in range(scores.shape[0])] ) / N

    # Gamma Correlation
    # https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_gamma

    gamma = np.sum( (nc - nd) / (nc + nd) ) / N

    return {
        "tau_a": tau_a,
        "tau_b": tau_b,
        "spearman_rho": spearman_rho,
        "gamma": gamma,
    }

def classification_metrics(scores, labels):

    N, K = labels.shape
    
    discrete_scores = (scores > 0).astype("int32")
    discrete_labels = (labels > 0).astype("int32")

    # Hamming loss
    # https://en.wikipedia.org/wiki/Hamming_loss
    hamming = hamming_loss(discrete_labels, discrete_scores)

    # Max One Error
    max_idxs = np.argmax(scores, 1)
    max_one_error = np.zeros_like(scores)
    max_one_error[np.arange(len(max_one_error)), max_idxs] = 1
    max_one_error = (1 - (max_one_error * discrete_labels).sum(1)).astype("float32").mean()

    # Coverage Error
    coverage = coverage_error(discrete_labels, scores)

    # Ranking
    ranking_loss = label_ranking_loss(discrete_labels, scores)
    
    # Average Precision
    ranking_average_precision = label_ranking_average_precision_score(discrete_labels, scores)

    # F1-Score
    TP = (discrete_scores * discrete_labels).sum()
    FP = (discrete_scores * (1 - discrete_labels)).sum()
    FN = ((1 - discrete_scores) * discrete_labels).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    return {
        "hamming": hamming,
        "max_one_error": max_one_error,
        "coverage": coverage,
        "ranking_loss": ranking_loss,
        "ranking_average_precision": ranking_average_precision,
        "f1_score": f1_score,
    }

bs = 64
device_name = "cuda:1"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--main_path", type=str)
parser.add_argument("--backbone", type=str, default="simple")
parser.add_argument("--dataset", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--domain", type=str)
parser.add_argument("--alt_save_path", type=str, default=None)
args = parser.parse_args()

if args.alt_save_path is not None:
    os.makedirs(args.alt_save_path, exist_ok=True)
    metric_save_path = os.path.join(args.alt_save_path, "%s_metrics.txt" % args.experiment_name)
else:
    metric_save_path = "results/%s/metrics.txt" % args.experiment_name


if args.dataset == "ranked_mnist":

    val_loader = DataLoader(
        RankedMNISTReader(args.main_path, args.config_path, mode="test"),
        batch_size=bs,
        shuffle=False,
        num_workers=8,
    )

    n_classes = 10

elif args.dataset == "landscape":

    val_loader = DataLoader(
        LandscapeReader(args.main_path, "test"),
        batch_size=bs,
        shuffle=False,
        num_workers=8,
    )

    n_classes = 9

elif args.dataset == "architecture":

    val_loader = DataLoader(
        ArchitectureReader(args.main_path, mode="test", domain=args.domain),
        batch_size=bs,
        shuffle=False,
        num_workers=8,
    )

    n_classes = 9


if args.method == "gaussian_mlr":
    model = GaussianModel(n_classes, args.backbone).to(device_name)
    best_path = "results/%s/saves/best.pth" % args.experiment_name


elif args.method == "clr":
    n_classes += 1 # Add virtual label
    model = Model((n_classes * (n_classes - 1)) // 2, args.backbone).to(device_name)
    best_path = "results/%s/saves/best.pth" % args.experiment_name

elif args.method == "lsep":
    model = LSEPModel(n_classes, args.backbone).to(device_name)
    best_path = "results/%s/saves/threshold_best.pth" % args.experiment_name

model.load_state_dict(torch.load(best_path, map_location=device_name)["state_dict"])
model = model.eval()
for param in model.parameters():
    model.requires_grad = False

ranking = {}
classification = {}

with torch.no_grad():
    for batch in val_loader:

        images = batch[0].to(device_name)
        labels = batch[1].to(device_name)

        N, K = labels.shape

        if args.method == "gaussian_mlr":
            mean, logvar = model(images)
            mean[mean < 0] = 0.0
            scores = mean

        elif args.method == "clr":
            K += 1
            logits = model(images)
            probs = torch.sigmoid(logits)
    
            pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)]).to(device_name)
            left_scores = probs >= 0.5
            right_scores = probs < 0.5

            score_matrix = torch.zeros((N, K)).to(device_name)

            for j in range(K):
                score_matrix[:, j] += torch.sum(left_scores[:, pair_map[:, 0] == j] * probs[:, pair_map[:, 0] == j], dim=1)
                score_matrix[:, j] += torch.sum(right_scores[:, pair_map[:, 1] == j] * probs[:, pair_map[:, 1] == j], dim=1)

            negative_map = score_matrix < score_matrix[:, -1].unsqueeze(1).repeat(1, K)
            score_matrix[negative_map] = 0

            scores = score_matrix[:, :-1]

        elif args.method == "lsep":
            scores, thresholds = model(images)
            scores[scores < thresholds] = 0.0

        labels = labels.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        new_metrics = ranking_metrics(scores, labels)
        for key in new_metrics:
            if key not in ranking:
                ranking[key] = []
            ranking[key].append(new_metrics[key])
        
        new_metrics = classification_metrics(scores, labels)
        for key in new_metrics:
            if key not in classification:
                classification[key] = []
            classification[key].append(new_metrics[key])

for key in ranking:
    ranking[key] = np.mean(ranking[key])

for key in classification:
    classification[key] = np.mean(classification[key])

with open(metric_save_path, "w") as f:
    f.write("Ranking Metrics\n")
    for key in ranking:
        f.write("%s: %.4f\n" % (key, ranking[key].item()))
    f.write("\n")
    f.write("Classification Metrics\n")
    for key in classification:
        f.write("%s: %.4f\n" % (key, classification[key].item()))