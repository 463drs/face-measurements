"""Target selectors: pick the one detection to run the landmark model on.

In production a selector works from the detections alone (e.g. highest score, largest face).
On test datasets an image can contain several faces but only one is annotated, so evaluation
uses GT-based selectors (`requires_gt = True`) that pick the detection matching the annotation,
or report why they couldn't.
"""
from abc import ABC, abstractmethod

import numpy as np

from ..structures import Detection, Selection


class TargetSelector(ABC):
    requires_gt: bool = False
    """True if the selector needs `gt_points` (evaluation only); calling it without them raises."""

    def select(self, detections: list[Detection], *, gt_points: np.ndarray | None = None) -> Selection:
        """Pick the target among `detections` (may be empty).

        `gt_points`: (N, 2) annotated points of the target face, when available (test datasets).
        """
        if self.requires_gt and gt_points is None:
            raise ValueError(f"{type(self).__name__} requires gt_points")
        return self._select(detections, gt_points)

    __call__ = select

    @abstractmethod
    def _select(self, detections: list[Detection], gt_points: np.ndarray | None) -> Selection:
        """`gt_points` is guaranteed non-None if `requires_gt`. Must handle empty `detections`."""
