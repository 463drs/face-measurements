"""2D-FAN landmark model (Bulat & Tzimiropoulos, ICCV 2017) from the `face-alignment` package.

Only the network and its heatmap decoder are taken from the package. Cropping and the mapping
back to image pixels are done here, in float, with an exact inverse affine: the package's own
`get_landmarks_from_image` rounds its crop and its output points to whole pixels.
See docs/decisions.md "Use only the FAN network, with our own crop and float output".
"""
from pathlib import Path

import numpy as np
import torch
from face_alignment.models import FAN
from face_alignment.utils import flip, get_preds_fromhm

from ..config import PROJECT_ROOT
from ..geometry import apply_affine, crop_square
from ..structures import CropResult, Detection, Landmarks
from .base import LandmarkModel

# Fetched (checksum-verified) by `python -m scripts.download_models fan2d`.
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models" / "2DFAN4-11f355bf06.pth.tar"

NUM_STACKS = 4
CROP_SIZE = 256
"""FAN input resolution."""
HEATMAP_SIZE = 64
"""FAN output heatmap resolution; one heatmap cell spans CROP_SIZE / HEATMAP_SIZE = 4 crop pixels."""

# Crop geometry of the `face-alignment` package: center moved up by 0.12 * box height, square side
DEFAULT_CROP_SCALE = 200 / 195
DEFAULT_CENTER_SHIFT_Y = -0.12


def resolve_device(device: str) -> torch.device:
    """`"auto"` -> CUDA if available, else CPU. Otherwise any torch device string (`"cpu"`, `"cuda:1"`)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available")
    return dev


class FaceAlignmentModel(LandmarkModel):
    """2D-FAN-4, 68 ibug points following the visible face contour. One face per call.

    Not thread-safe (shares one network). Points outside the crop are still returned (FAN places
    them at the heatmap border); low `scores` flag them.
    """

    schema = "ibug68"
    crop_size = CROP_SIZE

    def __init__(
        self,
        device: str = "auto",
        flip_test: bool = True,
        crop_scale: float = DEFAULT_CROP_SCALE,
        center_shift_y: float = DEFAULT_CENTER_SHIFT_Y,
        weights_path: str | Path | None = None,
    ):
        """
        `device`: "auto" | "cpu" | "cuda" | "cuda:N".
        `flip_test`: also run on the horizontally flipped crop and average the heatmaps
            (~2x slower, slightly more accurate; on in the `face-alignment` package by default).
        `crop_scale`, `center_shift_y`: passed to `crop_square` (with `size_mode="sum"`); defaults
            200/195 and -0.12 (the package's crop uses 200/165 for YuNet boxes).
        `weights_path`: None uses `models/2DFAN4-11f355bf06.pth.tar`.
        """
        if crop_scale <= 0:
            raise ValueError(f"crop_scale must be positive, got {crop_scale}")
        self.device = resolve_device(device)
        self.flip_test = flip_test
        self.crop_scale = crop_scale
        self.center_shift_y = center_shift_y

        path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
        if not path.is_file():
            hint = " (run `python -m scripts.download_models fan2d`)" if path == DEFAULT_WEIGHTS_PATH else ""
            raise FileNotFoundError(f"FAN weights not found at {path}{hint}")
        net = FAN(NUM_STACKS)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self._net = net.to(self.device).eval()

    def _heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x)
        return out[-1] if isinstance(out, list) else out  # last hourglass stack is the prediction

    def crop(self, image: np.ndarray, detection: Detection) -> CropResult:
        """The exact `CROP_SIZE` x `CROP_SIZE` crop the network receives for `detection`."""
        return crop_square(
            image, detection.box,
            scale=self.crop_scale, size_mode="sum", center_shift_y=self.center_shift_y,
            output_size=CROP_SIZE,
        )

    @torch.inference_mode()
    def _predict(self, image: np.ndarray, detection: Detection) -> Landmarks:
        crop = self.crop(image, detection)
        # Same preprocessing as the package: RGB, CHW, float in [0, 1], no mean/std normalization.
        x = torch.from_numpy(crop.image).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.float32) / 255.0
        hm = self._heatmaps(x)
        if self.flip_test:
            # `is_label=True` also swaps left/right point channels. Averaging (not summing, as the
            # package does) leaves the argmax unchanged and keeps scores on the same scale either way.
            hm = (hm + flip(self._heatmaps(flip(x)), is_label=True)) / 2
        hm = hm.float().cpu().numpy()

        # Package decoder: argmax + quarter-cell shift towards the higher neighbour, then -0.5.
        # Without center/scale it returns float heatmap coordinates where cell i spans [i, i + 1].
        pts_hm, _, scores = get_preds_fromhm(hm)
        # Heatmap edges [0, 64] <-> crop edges; crop pixel centers are integers -> crop edges are
        # [-0.5, 255.5]. Hence crop = hm * 4 - 0.5.
        pts_crop = pts_hm[0] * (CROP_SIZE / HEATMAP_SIZE) - 0.5
        pts = apply_affine(crop.inverse, pts_crop)
        return Landmarks(
            points=pts.astype(np.float32),
            schema=self.schema,
            scores=scores[0].astype(np.float32),
            crop=crop,
        )
