"""Convert raw 300W (images/ + pts/ with 68 ibug points per face) into the project parquet format.

Usage (from the project root):
    python -m scripts.prepare_300w
    python -m scripts.prepare_300w --root data/raw/300W_FIXED --out data/processed/300w.parquet
"""
import argparse
from pathlib import Path

import numpy as np

from facemeasure import write_parquet

SCHEMA = "ibug68"


def read_pts(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = lines.index("{") + 1
    end = lines.index("}")
    return np.array([line.split() for line in lines[start:end]], dtype=np.float32)


def rows(root: Path):
    for pts_path in sorted((root / "pts").glob("*.pts")):
        pts = read_pts(pts_path)
        assert pts.shape == (68, 2), pts_path
        image_path = root / "images" / f"{pts_path.stem}.jpg"
        yield {
            "image_id": f"300w/{pts_path.stem}",
            "image_path": image_path,
            "points": pts,
            "schema": SCHEMA,
            "attributes": {"dataset": "300W_FIXED"},
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/raw/300W_FIXED"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/300w.parquet"))
    args = parser.parse_args()
    n = write_parquet(rows(args.root), args.out)
    print(f"wrote {n} samples to {args.out}")
