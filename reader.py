import json
import os
from configparser import Interpolation

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

"""
1	plant
2	sky
3	cloud
4	snow
5	building
6	desert
7	mountain
8	water
9	sun
10	virtual
"""

arc_categories = [
    "asymmetric",
    "color",
    "crystallographic",
    "flowing",
    "isolation",
    "progressive",
    "regular",
    "shape",
    "symmetric",
]


class ArchitectureReader(torch.utils.data.Dataset):
    def __init__(self, main_path, mode="train", domain="ALL"):
        self.paths = []
        self.labels = []

        self.dataset_path = main_path

        with open(os.path.join(self.dataset_path, "splitted_labels.json"), "r") as f:
            data_json = json.load(f)

        if domain == "ALL":
            path_label_pairs = []
            for _, labels in data_json.items():
                for label in labels[mode]:
                    path_label_pairs.append(label)

        else:
            path_label_pairs = []
            for label in data_json[domain][mode]:
                path_label_pairs.append(label)

        self.transform = transforms.Compose(
            [
                transforms.RandAugment(
                    num_ops=3,
                    magnitude=10,
                    num_magnitude_bins=40,
                    interpolation=TF.InterpolationMode.BILINEAR,
                ),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.pairs = path_label_pairs

        self.mode = mode

    def load_image(self, path):
        return Image.open(path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        path, label = self.pairs[idx]

        img_path = os.path.join(self.dataset_path, path)
        ranked_one_hot_label = torch.zeros(len(arc_categories), dtype=torch.int32)

        ranked_one_hot_label[label[0] - 1] = 3
        if label[1] != -1:
            ranked_one_hot_label[label[1] - 1] = 2
            if label[2] != -1:
                ranked_one_hot_label[label[2] - 1] = 1

        if self.mode == "train":

            img = self.load_image(img_path)
            img = self.transform(img)

        else:
            img = self.load_image(img_path)
            img = self.transform_test(img)

        if self.mode != "test":
            return img, ranked_one_hot_label
        else:
            return img, ranked_one_hot_label, img_path


class LandscapeReader(torch.utils.data.Dataset):
    def __init__(self, main_path, mode="train"):

        self.main_path = main_path
        self.config_path = os.path.join(main_path, "config.json")
        self.mode = mode

        self.paths = sorted(
            os.listdir(os.path.join(main_path, "images")),
            key=lambda x: int(x.split(".")[0]),
        )
        self.paths = [os.path.join(main_path, "images", path) for path in self.paths]
        # Load data config
        with open(self.config_path, "r") as f:
            self.config = json.load(f)[mode]

        self.raw_labels = sio.loadmat(os.path.join(main_path, "pic_scene.mat"))

        self.raw_labels = [
            [
                [self.raw_labels["pic"][0][i][0][j][k] for k in range(10)]
                for j in range(10)
            ]
            for i in range(2000)
        ]
        self.raw_labels = np.array(self.raw_labels)

        self.labels = np.ones_like(self.raw_labels) * 0
        for i in range(2000):
            for j in range(10):
                row = self.raw_labels[i, j, :]
                vl_idx = np.where(row == 10)[0][0]
                self.labels[i, j, row - 1] = np.arange(10)
                self.labels[i, j, self.labels[i, j, :] > vl_idx] = vl_idx + 1

        self.labels = np.mean(self.labels, axis=1)

        virtual_label_threshold = np.expand_dims(self.labels[:, -1], axis=1).repeat(
            10, axis=1
        )
        self.labels[self.labels > virtual_label_threshold] = float("inf")
        self.labels = self.labels[:, :-1]

        for i in range(self.labels.shape[0]):
            uniques = np.unique(self.labels[i, :])
            ranks = np.arange(0, len(uniques))[::-1]
            for j in range(len(uniques)):
                self.labels[i, self.labels[i, :] == uniques[j]] = ranks[j]

        # for i in range(self.labels.shape[0]):
        #    if np.sum(self.labels[i, :]) == 0:
        #        print(self.labels[i, :], i)
        # INDEX 253 IS ILL LABELED
        self.config = [x for x in self.config if x != 253]

        self.transform = transforms.Compose(
            [
                transforms.RandAugment(
                    num_ops=3,
                    magnitude=10,
                    num_magnitude_bins=40,
                    interpolation=TF.InterpolationMode.BILINEAR,
                ),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.mode = mode

    def load_image(self, path):
        return Image.open(path)

    def __len__(self):
        return len(self.config)

    def __getitem__(self, idx):

        iidx = self.config[idx] - 1

        img = self.load_image(self.paths[iidx])
        label = torch.tensor(self.labels[iidx]).long()

        # Label: [0, 0, 2, 0, 0, 1, 0, 0, 3] -> falan filan gibi

        if self.mode == "train":
            img = self.transform(img)
        else:
            img = self.transform_test(img)

        if self.mode != "test":
            return img, label
        else:
            return img, label, self.paths[iidx]


class RankedMNISTReader(torch.utils.data.Dataset):
    def __init__(self, main_path, config_path, mode="train", subset=False):

        self.main_path = main_path
        self.paths = []
        self.labels = []
        self.mode = mode

        # Load data config
        with open(config_path, "r") as f:
            self.data_json = json.load(f)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
            ]
        )

        self.mode = mode

        if subset:
            self.data_json[self.mode] = self.data_json[self.mode][:2000]

    def load_image(self, path):
        return Image.open(path)

    def __len__(self):
        return len(self.data_json[self.mode])

    def __getitem__(self, idx):

        path, label = self.data_json[self.mode][idx]
        # Label: [0, 0, 2, 0, 0, 1, 0, 0, 0, 3] -> falan filan gibi

        # Join path with main path
        path = os.path.join(self.main_path, path)

        if self.mode == "train":

            img = self.load_image(path)
            img = self.transform(img)

        else:
            img = self.load_image(path)
            img = self.transform_test(img)

        if self.mode != "test":
            return img, torch.tensor(label)
        else:
            return img, torch.tensor(label), path


# reader = LandscapeReader("landscape_dataset", "test")
