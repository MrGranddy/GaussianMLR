# from scipy.ndimage import rotate
import colorsys
import hashlib
import json
import os
import random
import shutil

import numpy as np
from PIL import Image

dataset_name = "ranked_mnist_gray_small_scale_small_variance"
save_directory = "/mnt/disk2/ranked_MNIST_family"
config_path = "configs"

hash_object = hashlib.md5(dataset_name.encode())
seed = int(hash_object.hexdigest(), 16) % (10**9 + 7)

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

RATIO_LIM = (1, 1.5)
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

splitted_labels = {}
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

    splitted_labels[mode] = []
    for idx in range(set_size):

        canvas = np.zeros((c_height, c_width, 3), dtype="float32")
        num_labels = random.choice(num_label_choices)
        labels = np.random.choice(range(10), num_labels, replace=False)
        put_digits = []

        for label in labels:

            try_count = 0  # If can't place the digit (e.g. there is no room for another digit) stop trying
            while try_count < 1000:

                ratio = np.random.uniform(RATIO_LIM[0], RATIO_LIM[1])
                img_path = random.choice(mnist[mnist_mode][label])

                scaled_len = int(28 * ratio)

                l = np.random.randint(MIN_MARGIN, c_width - scaled_len - MIN_MARGIN)
                t = np.random.randint(MIN_MARGIN, c_width - scaled_len - MIN_MARGIN)

                can_fit = True
                for digit in put_digits:
                    ll, tt, ss, _, _, _ = digit

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

            put_digits.append((l, t, scaled_len, ratio, img_path, label))

        put_digits = sorted(put_digits, key=lambda x: x[3])

        gt = [0] * 10

        for rank, digit in enumerate(put_digits):
            l, t, scaled_len, ratio, img_path, label = digit

            img = np.array(
                Image.open(img_path).resize((scaled_len,) * 2, Image.BILINEAR)
            )[..., 0]

            img = img.astype("float32") / 255
            # rand_hsv_color = (np.random.rand(), np.random.uniform(0.5, 1.0), 1.0)
            # rand_rgb_color = colorsys.hsv_to_rgb(*rand_hsv_color)
            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)

            # img = rotate(img, np.random.uniform(-MAX_ROT, MAX_ROT))

            canvas[t : t + scaled_len, l : l + scaled_len, ...] = img

            gt[label] = rank + 1

        Image.fromarray((canvas * 255).astype("uint8")).save(
            os.path.join(dataset_path, mode, "%d.png" % idx)
        )
        splitted_labels[mode].append(
            (os.path.join(dataset_name, mode, "%d.png" % idx), gt)
        )

        counts[sum(1 for g in gt if g > 0)] += 1

        if idx % 100 == 0:
            print(mode, "%d/%d" % (idx, set_size))

with open(os.path.join(config_path, "%s.json" % dataset_name), "w") as f:
    json.dump(splitted_labels, f)
print([c / sum(counts) for c in counts])
