"""Download the 300W test set (indoor + outdoor) from Activeloop Hub into images/ + pts/.

Same source and output layout as https://github.com/PamanGie/300w-Dataset, which is what
scripts/prepare_300w.py reads. Needs deeplake 3.x (`pip install "deeplake<4"`).

Usage (from the project root):
    python -m scripts.download_300w
    python -m scripts.download_300w --out data/raw/300W_FIXED
"""
import argparse
import shutil
from pathlib import Path

import deeplake

HUB_PATH = "hub://activeloop/300w"
N_POINTS = 68


def write_pts(path: Path, points) -> None:
    lines = ["version: 1", f"n_points: {len(points)}", "{", *(f"{x:.6f} {y:.6f}" for x, y in points), "}"]
    path.write_text("\n".join(lines) + "\n")


def is_complete(root: Path, n: int) -> bool:
    images = {p.stem for p in (root / "images").glob("*.jpg")}
    pts = {p.stem for p in (root / "pts").glob("*.pts")}
    return len(images) == n and images == pts


def download(out: Path) -> None:
    ds = deeplake.load(HUB_PATH, read_only=True, verbose=False)
    n = len(ds)
    if out.is_dir():
        if is_complete(out, n):
            print(f"300w: up to date at {out} ({n} samples)")
            return
        raise SystemExit(f"300w: {out} exists but is incomplete; delete it and rerun")

    tmp = out.with_name(out.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "images").mkdir(parents=True)
    (tmp / "pts").mkdir()
    print(f"300w: downloading {n} samples from {HUB_PATH}")
    try:
        for i in range(n):
            # keypoints_coco: flat (x, y, visibility) per point, stored as int32 pixel coords
            points = ds.keypoints[i].numpy().reshape(N_POINTS, 3)[:, :2]
            stem = f"img_{i:05d}"
            # the stored JPEG bytes as-is: no decode/re-encode, grayscale images stay grayscale
            (tmp / "images" / f"{stem}.jpg").write_bytes(ds.images[i].tobytes())
            write_pts(tmp / "pts" / f"{stem}.pts", points)
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f"300w: {i + 1}/{n}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(out)  # only a complete download ever lands at the final path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"300w: saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw/300W_FIXED"))
    args = parser.parse_args()
    download(args.out)
