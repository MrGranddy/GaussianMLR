import colorsys
import hashlib
import json
import os
import random
import shutil

import numpy as np
from PIL import Image

num_digits = 4
num_tests = 50

hash_object = hashlib.md5(("calibration_brightness_%s" % num_digits).encode())
seed = int(hash_object.hexdigest(), 16) % (10**9 + 7)

random.seed(seed)
np.random.seed(seed)

c_height, c_width = 224, 224

# Give the path for your MNIST
mnist_paths = {
    "train": "/mnt/disk1/documents/Digit-Five/MNIST_train",
    "test": "/mnt/disk1/documents/Digit-Five/MNIST_test",
}

mnist = {
    int(digit): [
        os.path.join(mnist_paths["test"], digit, img_name)
        for img_name in os.listdir(os.path.join(mnist_paths["test"], digit))
    ]
    for digit in os.listdir(mnist_paths["test"])
}

ranked_mnist_path = "/mnt/disk2/calibration_brightness_test_%d_images" % num_digits

if os.path.isdir(ranked_mnist_path):
    shutil.rmtree(ranked_mnist_path)
os.makedirs(ranked_mnist_path)


def put_digit(canvas, digit, center):

    h, w, _ = digit.shape
    x, y = center

    hlx = w // 2
    hly = h // 2

    l = x - hlx
    r = l + w
    t = y - hly
    b = t + h

    canvas[t:b, l:r] = digit
    return canvas


canvas = np.zeros((c_height, c_width, 3), dtype="float32")

one_center = (56, 56)
two_center = (168, 168)
three_center = (56, 168)
four_center = (168, 56)

scales = [1.0 for x in range(num_digits)]
brightnesses = [0.25 + x * 0.25 for x in range(num_digits)]
# scales = map(lambda x: 3 * x / num_digits, scales)
# scales = [scale for scale in scales]

for n_idx in range(num_tests):

    # Select three unique digits from 0 to 9
    sel_digits = np.random.choice(list(mnist.keys()), num_digits, replace=False)

    one_image = Image.open(random.choice(mnist[sel_digits[0]]))
    two_image = Image.open(random.choice(mnist[sel_digits[1]]))
    if num_digits > 2:
        three_image = Image.open(random.choice(mnist[sel_digits[2]]))
    if num_digits > 3:
        four_image = Image.open(random.choice(mnist[sel_digits[3]]))

    image_name = "_".join(map(str, [n_idx] + [x for x in sel_digits])) + ".png"

    sample_path = os.path.join(ranked_mnist_path, image_name)

    one_scaled = (
        np.array(
            one_image.resize((int(28 * scales[0]), int(28 * scales[0])), Image.BILINEAR)
        ).astype("float32")
        / 255
    )
    one_scaled = one_scaled * brightnesses[0]

    two_scaled = (
        np.array(
            two_image.resize((int(28 * scales[1]), int(28 * scales[1])), Image.BILINEAR)
        ).astype("float32")
        / 255
    )
    two_scaled = two_scaled * brightnesses[1]

    if num_digits > 2:

        three_scaled = (
            np.array(
                three_image.resize(
                    (int(28 * scales[2]), int(28 * scales[2])), Image.BILINEAR
                )
            ).astype("float32")
            / 255
        )
        three_scaled = three_scaled * brightnesses[2]

    if num_digits > 3:

        four_scaled = (
            np.array(
                four_image.resize(
                    (int(28 * scales[3]), int(28 * scales[3])), Image.BILINEAR
                )
            ).astype("float32")
            / 255
        )
        four_scaled = four_scaled * brightnesses[3]

    canvas = np.zeros((c_height, c_width, 3), dtype="float32")

    coords = [one_center, two_center, three_center, four_center]
    random.shuffle(coords)

    canvas = put_digit(canvas, one_scaled, coords[0])
    canvas = put_digit(canvas, two_scaled, coords[1])
    if num_digits > 2:
        canvas = put_digit(canvas, three_scaled, coords[2])
    if num_digits > 3:
        canvas = put_digit(canvas, four_scaled, coords[3])

    Image.fromarray((canvas * 255).astype("uint8")).save(sample_path)
