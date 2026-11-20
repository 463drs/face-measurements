"""Data structures shared between pipeline components (detectors, selectors, landmark models)."""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A detected face. Coordinates are in pixels of the original input image."""

    box: np.ndarray
    """(4,) float32, `x1, y1, x2, y2` in pixels. Not clipped: may extend past the image border."""
    score: float
    """Detector confidence. Scale is detector-specific (YuNet: 0..1)."""
    keypoints: np.ndarray | None = None
    """(K, 2) float32 `x, y` pixels, ordered as defined by `keypoint_schema`; None if unsupported."""
    keypoint_schema: str | None = None
    """Name of the keypoint layout (e.g. "yunet5", "ibug68"); set iff `keypoints` is set."""

    def __post_init__(self):
        if (self.keypoints is None) != (self.keypoint_schema is None):
            raise ValueError("keypoints and keypoint_schema must be both set or both None")


class SelectionStatus(StrEnum):
    OK = "ok"
    """A detection matched the annotated face."""
    NO_DETECTION = "no_detection"
    """The detector returned nothing."""
    NO_MATCH = "no_match"
    """There were detections, but none passed the selector's matching criterion."""


@dataclass(frozen=True)
class Selection:
    """Result of picking the annotated (target) face among detections, with diagnostics."""

    status: SelectionStatus
    detection: Detection | None
    """The target detection; set iff `status` is OK."""
    n_detections: int
    """How many detections the selector was given."""
    best_index: int | None = None
    """Index (in the input list) of the best candidate, even if it was rejected; None if no detections."""
    best_iou: float | None = None
    """IoU of the best candidate with `target_box`; None if no detections or the selector doesn't use IoU."""
    target_box: np.ndarray | None = None
    """(4,) float32 `x1, y1, x2, y2` box derived from `gt_points`, if the selector builds one."""

    def __post_init__(self):
        if (self.detection is not None) != (self.status == SelectionStatus.OK):
            raise ValueError(f"detection must be set iff status is OK (status={self.status})")
