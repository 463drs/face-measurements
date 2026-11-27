"""Box and crop helpers. Boxes are `x1, y1, x2, y2` in continuous pixel coordinates."""
from typing import Literal, Sequence
from .structures import CropResult

import cv2
import numpy as np


def points_to_box(points: np.ndarray) -> np.ndarray:
    """(N, 2) points -> (4,) float32 tight bounding box."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("cannot build a box from zero points")
    return np.concatenate([points.min(0), points.max(0)])


def union_box(*boxes: np.ndarray | Sequence[float] | None) -> np.ndarray:
    """Smallest (4,) float32 box containing all given boxes; `None` entries are skipped."""
    present = [np.asarray(b, dtype=np.float32).reshape(4) for b in boxes if b is not None]
    if not present:
        raise ValueError("union_box needs at least one box")
    b = np.stack(present)
    return np.concatenate([b[:, :2].min(0), b[:, 2:].max(0)])


def box_iou(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    """IoU of each of (N, 4) `boxes` with a single (4,) `box` -> (N,) float32."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    box = np.asarray(box, dtype=np.float32)
    iw = np.clip(np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]), 0, None)
    ih = np.clip(np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]), 0, None)
    inter = iw * ih
    area = lambda b: np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area(boxes) + area(box) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a (2, 3) affine `matrix` to (N, 2) `points` -> (N, 2) float64."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(matrix, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]



def crop_square(
    image: np.ndarray,
    box: np.ndarray | Sequence[float],
    *,
    scale: float = 1.25,
    size_mode: Literal["max", "sum"] = "max",
    center_shift_y: float = 0.0,
    output_size: int = 256,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: float = 0,
) -> CropResult:
    """Cut a square around `box` and resize it to `output_size` x `output_size`.

    The square is centered on the box center, shifted vertically by `center_shift_y * h`
    (negative moves it up), and has side `scale * face_size` source pixels, where
    face_size is `max(w, h)` ("max") or `w + h` ("sum"). Scaling is isotropic. Integer
    coordinates are pixel centers (OpenCV convention): the square's center maps to
    `((output_size - 1) / 2, (output_size - 1) / 2)`. Parts of the square outside the
    image are filled with `border_value`; the square is never shifted or clipped to fit.
    `image` is not modified.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError(f"image must be a uint8 np.ndarray, got {type(image).__name__} "
                        f"{getattr(image, 'dtype', None)}")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError(f"image must have shape (H, W) or (H, W, 3), got {image.shape}")

    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,):
        raise ValueError(f"box must have 4 values (x1, y1, x2, y2), got shape {box.shape}")
    x1, y1, x2, y2 = box
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"box must satisfy x2 > x1 and y2 > y1, got {box}")

    output_size = int(output_size)

    w, h = x2 - x1, y2 - y1
    if size_mode == "max":
        face_size = max(w, h)
    elif size_mode == "sum":
        face_size = w + h

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 + center_shift_y * h
    side = scale * face_size
    s = output_size / side
    c_out = (output_size - 1) / 2

    matrix = np.array([[s, 0.0, c_out - s * cx],
                       [0.0, s, c_out - s * cy]], dtype=np.float64)
    inverse = cv2.invertAffineTransform(matrix).astype(np.float64)

    # A bare scalar borderValue only fills the first channel in OpenCV; broadcast it.
    border = (float(border_value),) * 4
    crop = cv2.warpAffine(
        np.ascontiguousarray(image), matrix, (output_size, output_size),
        flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )
    matrix.setflags(write=False)
    inverse.setflags(write=False)
    return CropResult(image=crop, matrix=matrix, inverse=inverse)
