import numpy as np
import random
from typing import Dict, Tuple, List, Optional
from collections import namedtuple

# Define a namedtuple for digit placement details
DigitPlacement = namedtuple('DigitPlacement', 'x y scaled_length ratio img_path label')

def place_digit(
    mnist: Dict[int, List[str]],
    label: int,
    c_height: int,
    c_width: int,
    ratio_lim: Tuple[float, float],
    min_margin: int,
    put_digits: List[DigitPlacement]
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
            intersect_check = (2 * abs((l + scaled_len // 2) - (digit.x + digit.scaled_length // 2)) < (digit.scaled_length + scaled_len)) and \
                              (2 * abs((t + scaled_len // 2) - (digit.y + digit.scaled_length // 2)) < (digit.scaled_length + scaled_len))
            if intersect_check:
                can_fit = False
                break

        if can_fit:
            return DigitPlacement(l, t, scaled_len, ratio, img_path, label)

        try_count += 1

    return None