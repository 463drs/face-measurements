"""Box helpers. Boxes are `x1, y1, x2, y2` in continuous pixel coordinates."""
import numpy as np


def points_to_box(points: np.ndarray) -> np.ndarray:
    """(N, 2) points -> (4,) float32 tight bounding box."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("cannot build a box from zero points")
    return np.concatenate([points.min(0), points.max(0)])


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
