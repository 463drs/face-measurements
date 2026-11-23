from .detectors.yunet import YuNetDetector
from .landmarks.face_alignment import FaceAlignmentModel
from .selectors.gt_match import GTMatchSelector
from .config import ComponentConfig

DETECTORS = {
    "yunet": YuNetDetector,
}

LANDMARK_MODELS = {
    "face_alignment": FaceAlignmentModel,
}

TARGET_SELECTORS = {
    "gt_match": GTMatchSelector,
}

def build(registry: dict, spec: ComponentConfig):
    if spec.name not in registry:
        raise KeyError(f"unknown component {spec.name!r}; available: {sorted(registry)}")
    return registry[spec.name](**spec.params)

def build_detector(spec: ComponentConfig):
    return build(DETECTORS, spec)

def build_landmark_model(spec: ComponentConfig):
    return build(LANDMARK_MODELS, spec)

def build_selector(spec: ComponentConfig):
    return build(TARGET_SELECTORS, spec)
