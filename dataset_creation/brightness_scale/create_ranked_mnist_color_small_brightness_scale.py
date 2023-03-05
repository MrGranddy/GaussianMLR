import os
import shutil
import json

import numpy as np
import random
from PIL import Image

# from scipy.ndimage import rotate
import colorsys
import hashlib


dataset_name = "ranked_mnist_color_small_brightness_scale"
save_directory = "/mnt/disk2/ranked_MNIST_family"
config_path = "../configs"

hash_object = hashlib.md5(dataset_name.encode())
seed = int(hash_object.hexdigest(), 16) % (10 ** 9 + 7)

random.seed(seed)
np.random.seed(seed)

if not os.path.isdir(save_directory):
    os.makedirs(save_directory)

if not os.path.isdir(config_path):
    os.makedirs(config_path)

dataset_path = os.path.join(save_directory, dataset_name)

if os.path.isdir(dataset_path):
    sel = input(
        "A folder for this dataset already exists, do you want to override? (Y/n)"
    )
    if sel.lower() == "y":
        shutil.rmtree(dataset_path)
        os.makedirs(dataset_path)
    else:
        print("Quitting...")
        exit(1)

MIN_NUM_LABELS = 1
MAX_NUM_LABELS = 10

c_height, c_width = 224, 224

RATIO_LIM = (1, 3)
MAX_ROT = 180 * np.pi / 16
MIN_MARGIN = 1

TRAIN_SIZE = 60000
VAL_SIZE = 10000
TEST_SIZE = 10000

# The paths for MNIST in you local system
# All MNIST images are converted into RGB PNG format
mnist_paths = {
    "train": "/mnt/disk1/documents/Digit-Five/MNIST_train",
    "test": "/mnt/disk1/documents/Digit-Five/MNIST_test",
}

# Create dictionary for easy reading
mnist = {
    "train": {
        int(digit): [
            os.path.join(mnist_paths["train"], digit, img_name)
            for img_name in os.listdir(os.path.join(mnist_paths["train"], digit))
        ]
        for digit in os.listdir(mnist_paths["train"])
    },
    "test": {
        int(digit): [
            os.path.join(mnist_paths["test"], digit, img_name)
            for img_name in os.listdir(os.path.join(mnist_paths["test"], digit))
        ]
        for digit in os.listdir(mnist_paths["test"])
    },
}
# Dict format: { "mode": {"which_digit? e.g. 0, 2, 7": ["full_path_of_image"] } }

for mode in ["train", "val", "test"]:
    os.makedirs(os.path.join(dataset_path, mode))

num_label_choices = range(MIN_NUM_LABELS, MAX_NUM_LABELS + 1)

splitted_labels_ratio = {}
splitted_labels_brightness = {}
counts = [0] * 12

for mode in ["train", "val", "test"]:

    if mode == "train":
        set_size = TRAIN_SIZE
        mnist_mode = "train"
    elif mode == "val":
        set_size = VAL_SIZE
        mnist_mode = "train"
    else:
        set_size = TEST_SIZE
        mnist_mode = "test"

    splitted_labels_ratio[mode] = []
    splitted_labels_brightness[mode] = []

    for idx in range(set_size):

        canvas = np.zeros((c_height, c_width, 3), dtype="float32")
        num_labels = random.choice(num_label_choices)
        labels = np.random.choice(range(10), num_labels, replace=False)
        put_digits = []

        for label in labels:

            try_count = 0  # If can't place the digit (e.g. there is no room for another digit) stop trying
            while try_count < 1000:

                img_path = random.choice(mnist[mnist_mode][label])

                # Sample a uniform number using RATIO_LIM name it ratio then create scaled_len by multiplying with constant 28
                ratio = np.random.uniform(*RATIO_LIM)
                scaled_len = int(28 * ratio)
                brightness = np.random.rand()

                l = np.random.randint(MIN_MARGIN, c_width - scaled_len - MIN_MARGIN)
                t = np.random.randint(MIN_MARGIN, c_width - scaled_len - MIN_MARGIN)

                can_fit = True
                for digit in put_digits:
                    ll, tt, ss, _, _, _, _ = digit

                    intersect_check = 2 * abs(
                        (l + scaled_len // 2) - (ll + ss // 2)
                    ) < (ss + scaled_len) and 2 * abs(
                        (t + scaled_len // 2) - (tt + ss // 2)
                    ) < (
                        ss + scaled_len
                    )

                    if intersect_check:
                        can_fit = False
                        break

                if can_fit:
                    break

                try_count += 1

            put_digits.append((l, t, scaled_len, brightness, ratio, img_path, label))

        ratio_gt = np.array([-1] * 10).astype("float32")
        brightness_gt = np.array([-1] * 10).astype("float32")

        for rank, digit in enumerate(put_digits):
            l, t, scaled_len, brightness, ratio, img_path, label = digit

            img = np.array(
                Image.open(img_path)
                .convert("RGB")
                .resize((scaled_len,) * 2, Image.BILINEAR)
            )[..., 0]

            img = img.astype("float32") / 255
            rand_hsv_color = (np.random.rand(), np.random.uniform(0.5, 1.0), brightness)
            rand_rgb_color = colorsys.hsv_to_rgb(*rand_hsv_color)
            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2) * np.expand_dims(
                np.array([*rand_rgb_color]), axis=(0, 1)
            )

            canvas[t : t + scaled_len, l : l + scaled_len, ...] = img

            ratio_gt[label] = ratio
            brightness_gt[label] = brightness

        ratio_argsort = np.argsort(ratio_gt)
        brightness_argsort = np.argsort(brightness_gt)

        ratio_rank = np.zeros((10,), dtype="int32")
        brightness_rank = np.zeros((10,), dtype="int32")

        ratio_rank[ratio_argsort] = np.arange(10)
        brightness_rank[brightness_argsort] = np.arange(10)

        ratio_rank[ratio_gt == -1] = -1
        brightness_rank[brightness_gt == -1] = -1

        min_ratio = np.min(ratio_rank[ratio_rank != -1])
        min_brightness = np.min(brightness_rank[brightness_rank != -1])

        ratio_rank -= min_ratio
        brightness_rank -= min_brightness

        ratio_rank += 1
        brightness_rank += 1

        ratio_rank[ratio_gt == -1] = 0
        brightness_rank[brightness_gt == -1] = 0

        # convert rank arrays to python lists
        ratio_rank = ratio_rank.tolist()
        brightness_rank = brightness_rank.tolist()

        Image.fromarray((canvas * 255).astype("uint8")).save(
            os.path.join(dataset_path, mode, "%d.png" % idx)
        )
        splitted_labels_ratio[mode].append(
            (os.path.join(dataset_name, mode, "%d.png" % idx), ratio_rank)
        )
        splitted_labels_brightness[mode].append(
            (os.path.join(dataset_name, mode, "%d.png" % idx), brightness_rank)
        )
        counts[sum(1 for g in ratio_rank if g > 0)] += 1

        if idx % 100 == 0:
            print(mode, "%d/%d" % (idx, set_size))

with open(os.path.join(config_path, "%s_ratio.json" % dataset_name), "w") as f:
    json.dump(splitted_labels_ratio, f)
with open(os.path.join(config_path, "%s_brightness.json" % dataset_name), "w") as f:
    json.dump(splitted_labels_brightness, f)
print([c / sum(counts) for c in counts])
