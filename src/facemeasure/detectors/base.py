"""Project-wide face detector interface.

Every detector takes an RGB uint8 image of shape (H, W, 3) and returns a list of
`Detection`s sorted by score, highest first. Coordinates are in input-image pixels.
"""
from abc import ABC, abstractmethod

import numpy as np

from ..structures import Detection


def check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3) RGB, got {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image is empty: {image.shape}")


class Detector(ABC):
    keypoint_schema: str | None = None
    """Schema name put into `Detection.keypoint_schema`. None if the detector has no keypoints."""
    keypoint_names: tuple[str, ...] = ()
    """Names of the keypoints in `Detection.keypoints`, in order. Empty if the detector has none."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in an RGB uint8 (H, W, 3) image."""
        check_image(image)
        return self._detect(image)

    __call__ = detect

    @abstractmethod
    def _detect(self, image: np.ndarray) -> list[Detection]:
        """`image` is already validated. Must return detections sorted by descending score."""
