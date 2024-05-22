import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def save_plot(stats, epoch_idx, path, prefix=""):

    _, ax = plt.subplots()

    for mode in ["train", "val"]:

        for label, arr in stats[mode].items():
            ax.plot(
                np.arange(1, len(arr) + 1),
                arr,
                label="%s_%s" % (mode, label),
            )

    ax.legend()

    plt.savefig(os.path.join(path, "%s_loss_%d.png" % (prefix, epoch_idx)), dpi=300)
    plt.cla()
    plt.close()


def calc_unranked_metrics(scores, labels):

    pred_labels = (scores > 0).long()
    labels = (labels != 0).long()

    # Number of TP by ANDing ground truth and prediction
    TP_one_hot = pred_labels * labels
    TP = torch.sum(TP_one_hot, dim=1)

    # TODO: Buradaki lojiği açıklamak
    info_one_hot = pred_labels - labels
    FP = torch.sum(info_one_hot > 0, dim=1)
    FN = torch.sum(info_one_hot < 0, dim=1)
    TN = 9 - (TP + FP + FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    metrics = {"precision": precision, "recall": recall, "acc": acc, "f1": f1}
    for key in metrics:
        metrics[key][metrics[key].isnan()] = 0
        metrics[key] = torch.mean(metrics[key]).detach().cpu().item()

    metrics["TP"] = TP.int().detach().cpu()
    metrics["TN"] = TN.int().detach().cpu()
    metrics["FP"] = FP.int().detach().cpu()
    metrics["FN"] = FN.int().detach().cpu()

    return metrics


def mAP(scores, labels):

    N, K = labels.shape

    pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)])

    precisions = torch.zeros(N, K).to(labels.device)

    partial_labels = labels.clone()

    N_idxs = torch.arange(N)

    for k_idx in range(K):

        left_labels = partial_labels[:, pair_map[:, 0]]
        right_labels = partial_labels[:, pair_map[:, 1]]
        bigger_labels = (left_labels > right_labels).int()
        smaller_labels = (left_labels < right_labels).int()

        left_scores = scores[:, pair_map[:, 0]]
        right_scores = scores[:, pair_map[:, 1]]
        bigger_scores = (left_scores > right_scores).int()

        TP = (bigger_labels * bigger_scores).sum(1).float()
        # TN = (smaller_labels * (1 - bigger_scores)).sum(1).float()
        FP = (smaller_labels * bigger_scores).sum(1).float()
        # FN = (bigger_labels * (1 - bigger_scores)).sum(1).float()

        precision = TP / (TP + FP)
        precision[precision.isnan()] = 0
        precisions[:, k_idx] = precision

        partial_labels[partial_labels == 0] = K + 1
        min_idxs = torch.argmin(partial_labels, dim=1)
        partial_labels[partial_labels == (K + 1)] = 0
        partial_labels[N_idxs, min_idxs] = 0

    mAP = precisions.sum(1) / (labels != 0).sum(1)

    return mAP


def calc_ranked_metrics(scores, labels):

    N, K = labels.shape

    pair_map = torch.tensor([(i, j) for i in range(K - 1) for j in range(i + 1, K)])

    left_labels = labels[:, pair_map[:, 0]]
    right_labels = labels[:, pair_map[:, 1]]
    bigger_labels = (left_labels > right_labels).int()
    smaller_labels = (left_labels < right_labels).int()

    left_scores = scores[:, pair_map[:, 0]]
    right_scores = scores[:, pair_map[:, 1]]
    bigger_scores = (left_scores > right_scores).int()
    # smaller_scores = (left_scores < right_scores).int()

    TP = (bigger_labels * bigger_scores).sum(1).float()
    TN = (smaller_labels * (1 - bigger_scores)).sum(1).float()
    FP = (smaller_labels * bigger_scores).sum(1).float()
    FN = (bigger_labels * (1 - bigger_scores)).sum(1).float()

    # Prediction bigger demedi ise smaller demiştir

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    metrics = {"precision": precision, "recall": recall, "acc": acc, "f1": f1}
    for key in metrics:
        metrics[key][metrics[key].isnan()] = 0
        metrics[key] = torch.mean(metrics[key]).detach().cpu().item()

    metrics["TP"] = TP.int().detach().cpu()
    metrics["TN"] = TN.int().detach().cpu()
    metrics["FP"] = FP.int().detach().cpu()
    metrics["FN"] = FN.int().detach().cpu()

    return metrics
