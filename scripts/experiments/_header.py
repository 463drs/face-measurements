"""Header printed at the top of every experiment's output, so a saved result file says how it was made.

Results go to stdout (redirect it to scripts/experiments/results/), progress to stderr. Paths in the header
are relative to the project root, so result files carry no local absolute paths.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from facemeasure import Config, FaceDataset
from facemeasure.config import PROJECT_ROOT


def git_commit() -> str:
    """Short HEAD hash, with "-dirty" if the working tree has uncommitted changes; "unknown" outside git."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    try:
        return git("rev-parse", "--short", "HEAD").strip() + ("-dirty" if git("status", "--porcelain").strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rel(path: str | Path) -> str:
    """`path` relative to the project root, or just its file name if it lies outside."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.name


def gt_convention(ds: FaceDataset) -> str:
    """Coordinate convention recorded by the data preparation script (in each sample's attributes)."""
    values = {ds[i].attributes.get("coordinates", "unknown") for i in (0, len(ds) - 1)}
    return ", ".join(sorted(values))


def print_header(script: str, config: str | Path, cfg: Config, ds: FaceDataset, n_images: int, **params) -> None:
    """Print the header block: date, commit, config and component params, extra `params`, images, GT."""
    config_name = rel(config) if Path(config).suffix in (".yaml", ".yml") else str(config)
    lines = [
        f"script: {script}",
        f"date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"git commit: {git_commit()}",
        f"config: {config_name}",
        f"data: {', '.join(rel(p) for p in cfg.data.paths)}",
        f"detector: {cfg.detector.name} {cfg.detector.params}",
        f"target_selector: {cfg.target_selector.name} {cfg.target_selector.params}",
        f"landmark_model: {cfg.landmark_model.name} {cfg.landmark_model.params}",
        *(f"{k}: {v}" for k, v in params.items()),
        f"images: first {n_images} of {len(ds)}",
        f"GT convention: {gt_convention(ds)} (points as stored in the parquet)",
    ]
    print("\n".join(f"# {line}" for line in lines))
    print(flush=True)


def progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
