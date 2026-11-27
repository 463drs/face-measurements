"""Component registries: config name -> "module:Class", imported only when the component is built.

Lazy on purpose: `import facemeasure` (config, data, detectors) must not load torch, which only the
landmark models need.
"""
from importlib import import_module

from .config import ComponentConfig

DETECTORS = {
    "yunet": "facemeasure.detectors.yunet:YuNetDetector",
}

LANDMARK_MODELS = {
    "face_alignment": "facemeasure.landmarks.face_alignment:FaceAlignmentModel",
}

TARGET_SELECTORS = {
    "gt_match": "facemeasure.selectors.gt_match:GTMatchSelector",
}


def resolve(registry: dict[str, str], name: str) -> type:
    """Import and return the class registered under `name`."""
    if name not in registry:
        raise KeyError(f"unknown component {name!r}; available: {sorted(registry)}")
    module, _, attr = registry[name].partition(":")
    return getattr(import_module(module), attr)


def build(registry: dict[str, str], spec: ComponentConfig):
    return resolve(registry, spec.name)(**spec.params)


def build_detector(spec: ComponentConfig):
    return build(DETECTORS, spec)


def build_landmark_model(spec: ComponentConfig):
    return build(LANDMARK_MODELS, spec)


def build_selector(spec: ComponentConfig):
    return build(TARGET_SELECTORS, spec)
