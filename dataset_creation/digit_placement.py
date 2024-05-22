import random
from collections import namedtuple
from skimage import color
from typing import Dict, List, Optional, Tuple

import numpy as np

# Define a namedtuple for digit placement details
DigitPlacement = namedtuple("DigitPlacement", "x y scaled_length ratio img_path label")


def place_digit(
    mnist: Dict[int, List[str]],
    label: int,
    c_height: int,
    c_width: int,
    ratio_lim: Tuple[float, float],
    min_margin: int,
    put_digits: List[DigitPlacement],
) -> Optional[DigitPlacement]:
    """
    Attempts to place a digit image on a canvas while avoiding overlaps.

    Parameters:
        mnist (dict): Dictionary containing MNIST image paths.
        label (int): The label of the digit to place.
        c_height (int): Height of the canvas.
        c_width (int): Width of the canvas.
        ratio_lim (tuple): A tuple containing the min and max scaling ratios.
        min_margin (int): Minimum margin around the digit.
        put_digits (list): List of DigitPlacement tuples for digits already placed on the canvas.

    Returns:
        DigitPlacement: A namedtuple containing placement x, y, scaled length, ratio, image path, and label of the digit, if successful.
        None: If no placement is found after 1000 attempts.
    """
    try_count = 0
    while try_count < 1000:
        ratio = np.random.uniform(ratio_lim[0], ratio_lim[1])
        img_path = random.choice(mnist[label])
        scaled_len = int(28 * ratio)
        l = np.random.randint(min_margin, c_width - scaled_len - min_margin)
        t = np.random.randint(min_margin, c_height - scaled_len - min_margin)

        can_fit = True
        for digit in put_digits:
            intersect_check = (
                2 * abs((l + scaled_len // 2) - (digit.x + digit.scaled_length // 2))
                < (digit.scaled_length + scaled_len)
            ) and (
                2 * abs((t + scaled_len // 2) - (digit.y + digit.scaled_length // 2))
                < (digit.scaled_length + scaled_len)
            )
            if intersect_check:
                can_fit = False
                break

        if can_fit:
            return DigitPlacement(l, t, scaled_len, ratio, img_path, label)

        try_count += 1

    return None

def rgb_to_lab(rgb_vector):
    """
    Convert a 3D RGB vector to a 3D LAB vector.

    Args:
        rgb_vector (tuple or list or np.ndarray): RGB vector with values in the range [0, 255].

    Returns:
        np.ndarray: LAB vector with values typically in the range of [0, 100] for L 
                    and [-128, 127] for a and b.
    """
    # Ensure the input is a numpy array and normalize the RGB values to [0, 1]
    rgb_normalized = np.array(rgb_vector) / 255.0
    
    # Reshape to a 3D array with one pixel (1, 1, 3)
    rgb_normalized = rgb_normalized.reshape((1, 1, 3))
    
    # Convert RGB to LAB using skimage
    lab_vector = color.rgb2lab(rgb_normalized)
    
    # Reshape back to a 1D array (3,)
    lab_vector = lab_vector.reshape((3,))
    
    return lab_vector

def lab_to_rgb(lab_vector):
    """
    Convert a 3D LAB vector to a 3D RGB vector.

    Args:
        lab_vector (tuple or list or np.ndarray): LAB vector with values typically in the range 
                                                  of [0, 100] for L and [-128, 127] for a and b.

    Returns:
        np.ndarray: RGB vector with values in the range [0, 255].
    """
    # Ensure the input is a numpy array and reshape to a 3D array with one pixel (1, 1, 3)
    lab_vector = np.array(lab_vector).reshape((1, 1, 3))
    
    # Convert LAB to RGB using skimage
    rgb_normalized = color.lab2rgb(lab_vector)
    
    # Reshape back to a 1D array (3,)
    rgb_normalized = rgb_normalized.reshape((3,))
    
    # Denormalize RGB values to [0, 255]
    rgb_vector = (rgb_normalized * 255).astype(np.uint8)
    
    return rgb_vector