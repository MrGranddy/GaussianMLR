import argparse
import hashlib
import json
import os
import random
import shutil
from typing import List

import numpy as np
import yaml
from PIL import Image

from digit_placement import DigitPlacement, place_digit
from mnist_utils import load_mnist_paths
from image_creator import ImageCreator

def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset using MNIST digits."
    )
    parser.add_argument("config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        common_config_path = config["COMMON_CONFIG_PATH"]

    with open(common_config_path, "r") as f:
        common_config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = config["DATASET_NAME"]
    datasets_path = common_config["DATASETS_PATH"]
    labels_path = common_config["LABELS_PATH"]

    # Create unique seed for this dataset name for reproducibility
    hash_object = hashlib.md5(dataset_name.encode())
    seed = int(hash_object.hexdigest(), 16) % (10**9 + 7)
    random.seed(seed)
    np.random.seed(seed)

    # Ensure dataset and label directories exist
    os.makedirs(datasets_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    dataset_path = os.path.join(datasets_path, dataset_name)

    if os.path.exists(dataset_path):
        sel = input(
            "A folder for this dataset already exists, do you want to override? (Y/n) "
        )
        if sel.lower() == "y":
            shutil.rmtree(dataset_path)
        else:
            print("Quitting...")
            exit(1)
    os.makedirs(dataset_path)

    mnist_paths = {
        "train": common_config["MNIST_TRAIN_PATH"],
        "test": common_config["MNIST_TEST_PATH"],
    }
    mnist = load_mnist_paths(mnist_paths["train"], mnist_paths["test"])

    # Directory setup for train, val, test
    for mode in ["train", "val", "test"]:
        os.makedirs(os.path.join(dataset_path, mode), exist_ok=True)

    # Generation process
    counts = [0] * 12
    splitted_labels = {mode: [] for mode in ["train", "val", "test"]}
    sizes = {
        "train": config["TRAIN_SIZE"],
        "val": config["VAL_SIZE"],
        "test": config["TEST_SIZE"],
    }

    for mode, size in sizes.items():
        mnist_mode = "train" if mode != "test" else "test"
        for idx in range(size):
            num_labels = random.choice(
                range(config["MIN_NUM_LABELS"], config["MAX_NUM_LABELS"] + 1)
            )
            labels = np.random.choice(range(10), num_labels, replace=False)
            put_digits: List[DigitPlacement] = []

            for label in labels:
                digit = place_digit(
                    mnist[mnist_mode],
                    label,
                    config["C_HEIGHT"],
                    config["C_WIDTH"],
                    config["RATIO_LIM"],
                    config["MIN_MARGIN"],
                    put_digits,
                )
                if digit:
                    put_digits.append(digit)

            put_digits.sort(key=lambda x: x.ratio)

            image_creator = ImageCreator(
                (config["C_HEIGHT"], config["C_WIDTH"])
            )
            image_creator.create_empty_canvas()

            gt = [0] * 10

            for rank, digit in enumerate(put_digits):

                img = Image.open(digit.img_path).convert("RGB").resize(
                    (digit.scaled_length,) * 2, Image.BILINEAR
                )
                img_array = np.array(img)
                
                # Add alpha channel to the image
                alpha = (img_array[:, :, 0] != 0).astype(np.uint8) * 255
                img_array = np.dstack([img_array, alpha])

                image_creator.add_element(
                    img_array,
                    (digit.y, digit.x),
                )

                gt[digit.label] = rank + 1

            canvas = image_creator.get_canvas()

            Image.fromarray(canvas).save(
                os.path.join(dataset_path, mode, f"{idx}.png")
            )
            splitted_labels[mode].append(
                (os.path.join(dataset_name, mode, f"{idx}.png"), gt)
            )

            if idx % 100 == 0:
                print(f"{mode}: {idx}/{size}")

    # Save label data
    with open(os.path.join(labels_path, f"{dataset_name}.json"), "w") as f:
        json.dump(splitted_labels, f)
    print([c / sum(counts) for c in counts])


if __name__ == "__main__":
    main()
