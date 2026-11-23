from abc import ABC, abstractmethod

import numpy as np

from ..structures import Detection, Landmarks
from ..validation import check_image


class LandmarkModel(ABC):
    schema: str
    """Point layout of the returned `Landmarks` (e.g. "ibug68"); must be a key of `schemas.SCHEMA_NUM_POINTS`."""

    def predict(self, image: np.ndarray, detection: Detection) -> Landmarks:
        """Landmarks of the face `detection` in RGB uint8 (H, W, 3) `image`, in `image` pixels."""
        check_image(image)
        result = self._predict(image, detection)
        self._check_result(result)
        return result

    __call__ = predict

    @abstractmethod
    def _predict(self, image: np.ndarray, detection: Detection) -> Landmarks:
        """`image` is already validated. Must return `Landmarks` with `self.schema`, in original-image pixels."""

    def _check_result(self, result: Landmarks) -> None:
        # Shape/dtype/finiteness are checked by `Landmarks.__post_init__`; only model-specific checks here.
        name = type(self).__name__
        if not isinstance(result, Landmarks):
            raise TypeError(f"{name}._predict must return Landmarks, got {type(result).__name__}")
        if result.schema != self.schema:
            raise ValueError(f"{name}: result schema {result.schema!r} != model schema {self.schema!r}")
