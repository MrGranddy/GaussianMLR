from typing import Tuple, List

import numpy as np


class ImageCreator:
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        """
        Args:
            size (Tuple[int, int]): The size of the canvas to be created in the format (height, width)
        """

        if (
            not isinstance(size, tuple)
            or len(size) != 2
            or not all(isinstance(i, int) for i in size)
        ):
            raise ValueError("Size must be a tuple of two integers")

        self.size = size
        self.canvas = None

    def create_empty_canvas(self):
        """
        Create an empty canvas with the given size
        """
        if self.canvas is not None:
            print("Canvas already exists, overwriting it")

        self.canvas = np.zeros((self.size[0], self.size[1], 3), dtype="uint8")

    def set_solid_color(self, color: Tuple[int, int, int]):
        """
        Set the canvas to a solid color

        Args:
            color (Tuple[int, int, int]): The color to set the canvas to in the format (R, G, B)
        """

        if self.canvas is None:
            self.create_empty_canvas()
            print("Canvas was empty, created a new one")

        self.canvas[:, :] = color  # type: ignore

    def add_element(self, element: np.ndarray, position: Tuple[int, int]):
        """
        Add an element to the canvas

        Args:
            element (np.ndarray): The element to add to the canvas, must have a shape of (height, width, 4) where the alpha channel
                is used for transparency, it is either 0 or 255, the image itself is in RGB format with values between 0 and 255 uint8.
            position (Tuple[int, int]): The position to add the element in the format (y, x), where (y, x) is the top left corner of the element
        """

        if self.canvas is None:
            raise ValueError("Canvas is not created, please create a canvas first")

        if (
            not isinstance(element, np.ndarray)
            or element.ndim != 3
            or element.shape[2] != 4
        ):
            raise ValueError(
                "Element must be a 3D numpy array with the last dimension being 4 for RGBA"
            )

        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not all(isinstance(i, int) for i in position)
        ):
            raise ValueError("Position must be a tuple of two integers")

        y, x = position
        height, width = element.shape[:2]

        # Calculate the boundaries ensuring they are within the canvas dimensions
        top = max(0, y)
        bottom = min(self.canvas.shape[0], y + height)
        left = max(0, x)
        right = min(self.canvas.shape[1], x + width)

        # Calculate the corresponding element boundaries
        element_top = max(0, -y)
        element_bottom = height - max(0, (y + height) - self.canvas.shape[0])
        element_left = max(0, -x)
        element_right = width - max(0, (x + width) - self.canvas.shape[1])

        # Get the region of the canvas and the corresponding region of the element
        canvas_region = self.canvas[top:bottom, left:right]
        element_region = element[element_top:element_bottom, element_left:element_right]

        # Create mask for the element region
        mask = element_region[:, :, 3] == 255

        # Apply the mask
        canvas_region[mask] = element_region[:, :, :3][mask]


    def get_canvas(self):
        """
        Get the canvas

        Returns:
            np.ndarray: The canvas
        """
        if self.canvas is None:
            raise ValueError("Canvas is not created, please create a canvas first")

        return self.canvas
