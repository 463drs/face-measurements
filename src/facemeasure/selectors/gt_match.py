import numpy as np

from ..geometry import box_iou, points_to_box
from ..structures import Detection, Selection, SelectionStatus
from .base import TargetSelector


class GTMatchSelector(TargetSelector):
    """Picks the detection whose box has the highest IoU with the GT points' bounding box.

    The GT box is the min/max of the annotated points, so it is tighter than a detector box
    (e.g. ibug68 has no forehead points) and IoUs of correct matches are well below 1:
    keep `iou_threshold` moderate.
    """

    requires_gt = True

    def __init__(self, iou_threshold: float = 0.3):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        self.iou_threshold = iou_threshold

    def _select(self, detections: list[Detection], gt_points: np.ndarray) -> Selection:
        target_box = points_to_box(gt_points)
        if not detections:
            return Selection(SelectionStatus.NO_DETECTION, None, n_detections=0, target_box=target_box)

        ious = box_iou(np.stack([d.box for d in detections]), target_box)
        best = int(np.argmax(ious))  # ties -> earliest, i.e. highest score
        best_iou = float(ious[best])
        matched = best_iou >= self.iou_threshold
        return Selection(
            status=SelectionStatus.OK if matched else SelectionStatus.NO_MATCH,
            detection=detections[best] if matched else None,
            n_detections=len(detections),
            best_index=best,
            best_iou=best_iou,
            target_box=target_box,
        )
