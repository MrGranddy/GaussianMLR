import colorsys
import hashlib
import json
import os
import random
import shutil

import numpy as np
from PIL import Image

mode = "color"  # or "color"
interpolate = "brightness"  # or "brightness"
randomize = "scale"  # "scale" # or "brightness"
static = ""  # or "scale"

interpolation_steps = 50
num_tests = 50

hash_object = hashlib.md5("_".join([mode, interpolate, randomize]).encode())
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

ranked_mnist_path = "/mnt/disk2/interpolation_test_images/%s_%s_%s_%s" % (
    mode,
    interpolate,
    randomize,
    static,
)

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

one_center = (112, 56)
two_center = (56, 168)
three_center = (168, 168)

smallest_scale = 1.0
biggest_scale = 3.0
center_scale = 2.0

static_scale = 1.0

smallest_brightness = 0.0
biggest_brightness = 1.0
center_brightness = 0.5

static_brightness = 1.0

scale_interpolation = np.linspace(smallest_scale, biggest_scale, interpolation_steps)
brightness_interpolation = np.linspace(
    smallest_brightness, biggest_brightness, interpolation_steps
)

for n_idx in range(num_tests):

    # Select three unique digits from 0 to 9
    sel_digits = np.random.choice(list(mnist.keys()), 3, replace=False)

    one_image = Image.open(random.choice(mnist[sel_digits[0]]))
    two_image = Image.open(random.choice(mnist[sel_digits[1]]))
    three_image = Image.open(random.choice(mnist[sel_digits[2]]))

    sample_path = os.path.join(
        ranked_mnist_path,
        "%d_%d_%d_%d" % (n_idx, sel_digits[0], sel_digits[1], sel_digits[2]),
    )
    os.makedirs(sample_path)

    for t_idx in range(interpolation_steps):

        if interpolate == "scale":
            one_scale = scale_interpolation[t_idx]
            two_scale = center_scale
            three_scale = smallest_scale + biggest_scale - scale_interpolation[t_idx]
        elif randomize == "scale":
            one_scale = random.uniform(smallest_scale, biggest_scale)
            two_scale = random.uniform(smallest_scale, biggest_scale)
            three_scale = random.uniform(smallest_scale, biggest_scale)
        elif static == "scale":
            one_scale = static_scale
            two_scale = static_scale
            three_scale = static_scale

        if interpolate == "brightness":
            one_brightness = brightness_interpolation[t_idx]
            two_brightness = center_brightness
            three_brightness = (
                smallest_brightness
                + biggest_brightness
                - brightness_interpolation[t_idx]
            )
        elif randomize == "brightness":
            one_brightness = random.uniform(smallest_brightness, biggest_brightness)
            two_brightness = random.uniform(smallest_brightness, biggest_brightness)
            three_brightness = random.uniform(smallest_brightness, biggest_brightness)
        elif static == "brightness":
            one_brightness = static_brightness
            two_brightness = static_brightness
            three_brightness = static_brightness

        one_scaled = (
            np.array(
                one_image.resize(
                    (int(28 * one_scale), int(28 * one_scale)), Image.BILINEAR
                )
            ).astype("float32")
            / 255
        )
        two_scaled = (
            np.array(
                two_image.resize(
                    (int(28 * two_scale), int(28 * two_scale)), Image.BILINEAR
                )
            ).astype("float32")
            / 255
        )
        three_scaled = (
            np.array(
                three_image.resize(
                    (int(28 * three_scale), int(28 * three_scale)), Image.BILINEAR
                )
            ).astype("float32")
            / 255
        )

        if mode == "gray":
            one_scaled = one_scaled * one_brightness
            two_scaled = two_scaled * two_brightness
            three_scaled = three_scaled * three_brightness
        elif mode == "color":
            one_rand_hsv_color = (
                np.random.rand(),
                np.random.uniform(0.5, 1.0),
                one_brightness,
            )
            two_rand_hsv_color = (
                np.random.rand(),
                np.random.uniform(0.5, 1.0),
                two_brightness,
            )
            three_rand_hsv_color = (
                np.random.rand(),
                np.random.uniform(0.5, 1.0),
                three_brightness,
            )

            one_rand_rgb_color = colorsys.hsv_to_rgb(*one_rand_hsv_color)
            two_rand_rgb_color = colorsys.hsv_to_rgb(*two_rand_hsv_color)
            three_rand_rgb_color = colorsys.hsv_to_rgb(*three_rand_hsv_color)

            one_scaled = one_scaled * np.expand_dims(
                np.array([*one_rand_rgb_color]), axis=(0, 1)
            )
            two_scaled = two_scaled * np.expand_dims(
                np.array([*two_rand_rgb_color]), axis=(0, 1)
            )
            three_scaled = three_scaled * np.expand_dims(
                np.array([*three_rand_rgb_color]), axis=(0, 1)
            )

        canvas = np.zeros((c_height, c_width, 3), dtype="float32")

        canvas = put_digit(canvas, one_scaled, one_center)
        canvas = put_digit(canvas, two_scaled, two_center)
        canvas = put_digit(canvas, three_scaled, three_center)

        Image.fromarray((canvas * 255).astype("uint8")).save(
            os.path.join(sample_path, "%d.png" % t_idx)
        )
