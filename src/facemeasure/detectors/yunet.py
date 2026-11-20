"""OpenCV YuNet face detector (https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)."""
from pathlib import Path

import cv2
import numpy as np

from ..config import PROJECT_ROOT
from ..structures import Detection
from .base import Detector

# Fetched (pinned commit + checksum) by `python -m scripts.download_models yunet`.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


class YuNetDetector(Detector):
    """Wraps `cv2.FaceDetectorYN`. Not thread-safe: the underlying net keeps per-input-size state.

    Keypoints follow YuNet's order. "Right"/"left" are the subject's, so the right eye
    usually has the smaller x in the image.
    """

    keypoint_schema = "yunet5"
    keypoint_names = ("right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner")

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_side: int | None = None,
    ):
        """`model_path=None` uses `models/face_detection_yunet_2023mar.onnx`.

        `max_side`: images whose longer side exceeds it are downscaled (keeping aspect ratio)
        before detection; outputs are mapped back to original-image pixels. None disables it.
        """
        if max_side is not None and max_side <= 0:
            raise ValueError(f"max_side must be positive or None, got {max_side}")
        self.max_side = max_side
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not path.is_file():
            hint = " (run `python -m scripts.download_models yunet`)" if path == DEFAULT_MODEL_PATH else ""
            raise FileNotFoundError(f"YuNet model not found at {path}{hint}")
        # The input size is a placeholder; it's set per image in `_detect`.
        self._net = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._input_size = (320, 320)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Downscale so the longer side is at most `max_side`; never upscale."""
        h, w = image.shape[:2]
        if self.max_side is None or max(h, w) <= self.max_side:
            return image
        scale = self.max_side / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))  # cv2 takes (w, h)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def _detect(self, image: np.ndarray) -> list[Detection]:
        orig_h, orig_w = image.shape[:2]
        image = self._resize(image)
        h, w = image.shape[:2]
        if (w, h) != self._input_size:
            self._net.setInputSize((w, h))
            self._input_size = (w, h)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # YuNet was trained on BGR
        _, faces = self._net.detect(bgr)
        if faces is None:
            return []
        # Each row: x, y, w, h, 5 x (kx, ky), score
        faces = faces[np.argsort(-faces[:, 14])]
        # Per-axis factors back to original pixels (rounding makes them differ slightly).
        sx, sy = orig_w / w, orig_h / h
        return [
            Detection(
                box=np.array([f[0] * sx, f[1] * sy, (f[0] + f[2]) * sx, (f[1] + f[3]) * sy], dtype=np.float32),
                score=float(f[14]),
                keypoints=(f[4:14].reshape(5, 2) * (sx, sy)).astype(np.float32),
                keypoint_schema=self.keypoint_schema,
            )
            for f in faces
        ]
