from typing import Tuple

import numpy as np


class ImageCreator:
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        """
        Args:
            size (Tuple[int, int]): The size of the canvas to be created in the format (height, width)
        """

        if not isinstance(size, tuple) or len(size) != 2 or not all(isinstance(i, int) for i in size):
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

        self.canvas[:, :] = color # type: ignore

    def add_element(self, element: np.ndarray, position: Tuple[int, int]):
        """
        Add an element to the canvas

        Args:
            element (np.ndarray): The element to add to the canvas, must have a shape of (height, width, 4) where the alpha channel
                is used for transparency, it is either 0 or 255, the image itself is in RGB format with values between 0 and 255 uint8.
            position (Tuple[int, int]): The position to add the element in the format (y, x), where the position is the center of the element
        """

        if self.canvas is None:
            raise ValueError("Canvas is not created, please create a canvas first")
        
        if not isinstance(element, np.ndarray) or element.ndim != 3 or element.shape[2] != 4:
            raise ValueError("Element must be a 3D numpy array with the last dimension being 4 for RGBA")
    
        if not isinstance(position, tuple) or len(position) != 2 or not all(isinstance(i, int) for i in position):
            raise ValueError("Position must be a tuple of two integers")
        
        y, x = position

        if y < 0 or y >= self.size[0] or x < 0 or x >= self.size[1]:
            raise ValueError("Position is out of bounds")
        
        height, width = element.shape[:2]

        if y - height // 2 < 0 or y + height // 2 >= self.size[0] or x - width // 2 < 0 or x + width // 2 >= self.size[1]:
            raise ValueError("Element is out of bounds")
        
        self.canvas[y - height // 2 : y + height // 2, x - width // 2 : x + width // 2, element[:, :, 3] == 255] = element[element[:, :, 3] == 255]

    def get_canvas(self):
        """
        Get the canvas

        Returns:
            np.ndarray: The canvas
        """

        return self.canvas

