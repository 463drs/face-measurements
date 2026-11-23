"""Project-wide face detector interface.

Every detector takes an RGB uint8 image of shape (H, W, 3) and returns a list of
`Detection`s sorted by score, highest first. Coordinates are in input-image pixels.
"""
from abc import ABC, abstractmethod

import numpy as np

from ..structures import Detection
from ..validation import check_image


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
