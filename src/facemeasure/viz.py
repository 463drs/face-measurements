"""Matplotlib drawing helpers

Coordinates are image pixels, as everywhere in the package; axes are expected to show an image
via `show` (y axis pointing down).
"""
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .geometry import apply_affine
from .schemas import IBUG68_REGIONS
from .structures import CropResult, Detection

# Fixed categorical order, one color per ibug68 region.
REGION_COLORS: dict[str, str] = dict(zip(IBUG68_REGIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

C_TOP = "#2a78d6"        # highest-score detection
C_GT_BOX = "white"       # GT box (from annotated points)
C_SELECTED = "#1baf7a"   # matched / selected detection
C_REJECTED = "#e34948"   # best candidate that failed the selector's criterion
C_OTHER = "#c3c2b7"      # all other detectidons
C_KPTS = "#6250d6"       # detector keypoints (e.g. YuNet's 5)
C_CROP = "#eda100"       # crop square
C_GT_PTS = "white"       # GT landmarks, when drawn together with predictions
C_PRED = "#e34948"       # predicted landmarks

OUTLINE = [pe.withStroke(linewidth=3, foreground="black", alpha=0.6)]
"""Path effect that keeps thin marks visible on any photo."""


def show(ax: Axes, image: np.ndarray, title: str | None = None) -> None:
    ax.imshow(image)
    ax.set_axis_off()
    if title:
        ax.set_title(title)


def draw_ibug68(ax: Axes, pts: np.ndarray, s: float = 12, lines: bool = True, labels: bool = False,
                color: str | None = None) -> None:
    """68 ibug points colored by region (or all in `color`); `lines` connects each contour,
    `labels` prints point indices."""
    for region, (ranges, closed) in IBUG68_REGIONS.items():
        c = color or REGION_COLORS[region]
        for r in ranges:
            p = pts[list(r)]
            if lines:
                p_line = np.vstack([p, p[:1]]) if closed else p
                ax.plot(*p_line.T, color=c, lw=1, alpha=0.9)
            ax.scatter(*p.T, s=s, color=c, edgecolors="black", linewidths=0.4, zorder=3)
    if labels:
        for i, (x, y) in enumerate(pts):
            ax.annotate(str(i), (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color="white", path_effects=OUTLINE)


def draw_box(ax: Axes, box, color: str, lw: float = 1.5, ls: str = "-",
             label: str | None = None, label_pos: str = "top") -> None:
    """`x1, y1, x2, y2` box; `label` is drawn on a filled tag at the box's top or bottom edge."""
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
                           lw=lw, ls=ls, path_effects=OUTLINE if lw >= 1.5 else None))
    if label:
        y, va = (y1, "bottom") if label_pos == "top" else (y2, "top")
        ax.text(x1, y, label, color="black" if color == "white" else "white", fontsize=8, va=va, ha="left",
                bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.85), clip_on=True)


def draw_detection(ax: Axes, det: Detection, color: str, lw: float = 1.5, label: str | None = None,
                   kpts: bool = True, kpt_size: float = 20, kpt_color: str = C_KPTS) -> None:
    """Detection box plus its keypoints (if any and `kpts`)."""
    draw_box(ax, det.box, color, lw=lw, label=label)
    if kpts and det.keypoints is not None:
        ax.scatter(*det.keypoints.T, s=kpt_size, marker="x", color=kpt_color, linewidths=1.5,
                   path_effects=OUTLINE, zorder=4)


def crop_outline(crop: CropResult) -> np.ndarray:
    """(5, 2) closed polyline of the crop's outer pixel edges, in source-image pixels.

    Pixel centers are integers, so the crop's edges are at -0.5 and size - 0.5.
    """
    size = crop.image.shape[0]
    lo, hi = -0.5, size - 0.5
    corners = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi], [lo, lo]])
    return apply_affine(crop.inverse, corners)


def zoom_to(ax: Axes, box, image_shape: tuple[int, ...], margin: float = 0.35) -> None:
    """Set view limits to `box` (+ relative margin), clipped to the image."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    ax.set_xlim(max(0, x1 - mx), min(w, x2 + mx))
    ax.set_ylim(min(h, y2 + my), max(0, y1 - my))  # y axis points down


def region_handles() -> list[Line2D]:
    """Legend handles for the ibug68 region colors."""
    return [Line2D([], [], color=c, marker="o", lw=1, label=r) for r, c in REGION_COLORS.items()]


def legend_below(fig: Figure, handles: list) -> None:
    """Legend under the panels; needs a figure created with layout="compressed" (or "constrained")."""
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)


def region_legend(fig: Figure) -> None:
    legend_below(fig, region_handles())
