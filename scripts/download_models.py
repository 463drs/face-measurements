"""Download model weights pinned to a source commit and verify their SHA-256.

Usage (from the project root):
    python -m scripts.download_models            # all models
    python -m scripts.download_models yunet      # selected models
"""
import argparse
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.facemeasure.detectors.yunet import DEFAULT_MODEL_PATH as YUNET_PATH


@dataclass(frozen=True)
class ModelFile:
    url: str
    sha256: str
    path: Path


# opencv_zoo keeps weights in Git LFS; media.githubusercontent.com serves the LFS object as of
# the given commit. The sha256 is the `oid` in the LFS pointer file at that commit.
OPENCV_ZOO_COMMIT = "f12e12798e8314f7c074a6656816c048dcc95b7a"  # "update yunet to v2 (#151)"

MODELS = {
    "yunet": ModelFile(
        url=(
            f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{OPENCV_ZOO_COMMIT}/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        path=YUNET_PATH,
    ),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, model: ModelFile) -> None:
    if model.path.is_file():
        if sha256_of(model.path) == model.sha256:
            print(f"{name}: up to date at {model.path}")
            return
        print(f"{name}: checksum mismatch in existing {model.path}, re-downloading")

    model.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model.path.with_suffix(model.path.suffix + ".part")
    print(f"{name}: downloading {model.url}")
    try:
        urllib.request.urlretrieve(model.url, tmp)
        actual = sha256_of(tmp)
        if actual != model.sha256:
            raise RuntimeError(f"{name}: checksum mismatch: expected {model.sha256}, got {actual}")
        tmp.replace(model.path)  # only a verified file ever lands at the final path
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{name}: saved to {model.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"models to download, any of {sorted(MODELS)} (default: all)")
    args = parser.parse_args()
    unknown = set(args.names) - set(MODELS)
    if unknown:
        parser.error(f"unknown models {sorted(unknown)}; available: {sorted(MODELS)}")
    for name in args.names or MODELS:
        download(name, MODELS[name])
