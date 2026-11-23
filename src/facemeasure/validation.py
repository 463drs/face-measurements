"""Input checks shared by pipeline components."""
import numpy as np


def check_image(image: np.ndarray) -> None:
    """Raise unless `image` is a non-empty RGB uint8 array of shape (H, W, 3)."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3) RGB, got {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image is empty: {image.shape}")
