import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import strong_CLR, weak_CLR
from model import Model
from reader import ArchitectureReader, LandscapeReader, RankedMNISTReader
from utils import save_plot

warnings.filterwarnings("ignore")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--main_path", type=str)
parser.add_argument("--num_epoch", type=int, default=20)
parser.add_argument("--backbone", type=str, default="simple")
parser.add_argument("--dataset", type=str)
parser.add_argument("--supervision", type=str)
parser.add_argument("--domain", type=str, default="ARC")
parser.add_argument("--subset", type=bool, default=False)
args = parser.parse_args()

main_result_path = os.path.join("results", args.experiment_name)

loss_path = os.path.join(main_result_path, "losses")
save_path = os.path.join(main_result_path, "saves")

plot_freq = 1
save_freq = 10000
preprint_freq = 100

if not os.path.isdir(main_result_path):
    os.makedirs(loss_path)
    os.makedirs(save_path)
else:
    print("DIRECTORY ALREADY EXISTS, continuing")
    time.sleep(1)

device_name = "cuda:0"

n_epoch = args.num_epoch
bs = 64

# Load data

if args.dataset == "ranked_mnist":
    train_loader = DataLoader(
        RankedMNISTReader(
            args.main_path, args.config_path, mode="train", subset=args.subset
        ),
        batch_size=bs,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        RankedMNISTReader(
            args.main_path, args.config_path, mode="val", subset=args.subset
        ),
        batch_size=bs,
        shuffle=False,
        num_workers=8,
    )

    n_classes = 10 + 1

elif args.dataset == "landscape":

    train_loader = DataLoader(
        LandscapeReader(args.main_path, "train"),
        batch_size=bs,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        LandscapeReader(args.main_path, "test"),
        batch_size=bs,
        shuffle=False,
        num_workers=8,
    )

    n_classes = 9 + 1

elif args.dataset == "architecture":

    train_loader = DataLoader(
        ArchitectureReader(args.main_path, mode="train", domain=args.domain),
        batch_size=bs,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        ArchitectureReader(args.main_path, mode="val", domain=args.domain),
        batch_size=bs,
        shuffle=False,
        num_workers=8,
    )

    n_classes = 9 + 1

# Model

if args.supervision == "weak":
    criterion = weak_CLR
elif args.supervision == "strong":
    criterion = strong_CLR

if args.dataset == "ranked_mnist":
    model = Model((n_classes * (n_classes - 1)) // 2, args.backbone).to(device_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-5)
    schedual = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
elif args.dataset == "landscape" or args.dataset:
    model = Model(
        (n_classes * (n_classes - 1)) // 2, args.backbone, pretrained=True
    ).to(device_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-5)
    schedual = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

stats = {"train": {}, "val": {}}

best_val_loss = 9999999

for epoch_idx in range(n_epoch):

    start_time = time.time()

    # Training
    model = model.train()
    running_stats = {}
    for iter_idx, (images, labels) in enumerate(train_loader):

        model.zero_grad()
        optimizer.zero_grad()

        images = images.to(device_name)
        labels = labels.to(device_name)

        logits = model(images)

        loss = criterion(logits, labels)
        losses = {"loss": loss}

        loss.backward()
        optimizer.step()

        losses = {key: val.detach().cpu().item() for key, val in losses.items()}

        for key, val in losses.items():
            if key not in running_stats:
                running_stats[key] = [val]
            else:
                running_stats[key].append(val)

        if (iter_idx + 1) % preprint_freq == 0:
            print("(%d/%d) %.6f" % (iter_idx + 1, len(train_loader), loss))

    average_stats = {key: np.mean(val) for key, val in running_stats.items()}

    for key, val in average_stats.items():
        if key not in stats["train"]:
            stats["train"][key] = [val]
        else:
            stats["train"][key].append(val)

    # Validation
    model = model.eval()
    running_stats = {}

    with torch.no_grad():
        for batch in val_loader:

            images = images.to(device_name)
            labels = labels.to(device_name)

            logits = model(images)

            loss = criterion(logits, labels)
            losses = {"loss": loss}

            losses = {key: val.detach().cpu().item() for key, val in losses.items()}

            for key, val in losses.items():
                if key not in running_stats:
                    running_stats[key] = [val]
                else:
                    running_stats[key].append(val)

    average_stats = {key: np.mean(val) for key, val in running_stats.items()}

    for key, val in average_stats.items():
        if key not in stats["val"]:
            stats["val"][key] = [val]
        else:
            stats["val"][key].append(val)

    end_time = time.time()

    if (epoch_idx + 1) % plot_freq == 0:
        save_plot(stats, epoch_idx + 1, loss_path)

    if (epoch_idx + 1) % save_freq == 0:
        torch.save(
            {"state_dict": model.state_dict(), "stats": stats},
            os.path.join(save_path, "ckpt_%d.pth" % (epoch_idx + 1)),
        )

    last_train_loss = sum(val[-1] for _, val in stats["train"].items())
    last_val_loss = sum(val[-1] for _, val in stats["val"].items())
    duration = end_time - start_time

    print(
        "Epoch %d: Train: %.6f, Val: %.6f, Time: %.2f"
        % (epoch_idx + 1, last_train_loss, last_val_loss, duration)
    )

    if last_val_loss < best_val_loss:
        best_val_loss = last_val_loss
        torch.save(
            {"state_dict": model.state_dict(), "stats": stats},
            os.path.join(save_path, "best.pth"),
        )

    schedual.step()
