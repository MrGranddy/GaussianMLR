import os
from typing import Dict, List


def load_mnist_paths(
    mnist_train_path: str, mnist_test_path: str
) -> Dict[str, Dict[int, List[str]]]:
    """
    Loads MNIST dataset paths from the given directory paths for train and test datasets.

    Args:
        mnist_train_path (str): Path to the MNIST training images.
        mnist_test_path (str): Path to the MNIST testing images.

    Returns:
        Dict[str, Dict[int, List[str]]]: A dictionary containing paths grouped by training and testing modes and labels.
    """
    mnist = {}
    for mode, path in [("train", mnist_train_path), ("test", mnist_test_path)]:
        mnist[mode] = {
            int(digit): [
                os.path.join(path, digit, img_name)
                for img_name in os.listdir(os.path.join(path, digit))
            ]
            for digit in os.listdir(path)
        }
    return mnist
