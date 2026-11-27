"""Which heatmap->crop offset, crop scale and GT shift remove FAN's systematic offset?

For each of the first N 300W images that the pipeline selects a face for, the network runs once per
reference scale (crop_scale = 200 / ref); every decoding offset and GT shift is then scored on the
same heatmaps. Reports mean / median NME (inter-ocular, outer eye corners) and the mean signed error
without the jaw (pred - GT, % of inter-ocular distance). GT "stored" is the parquet as is; "stored-1"
subtracts another 1 px (a check that the stored convention is right).

Usage (from the project root):
    python -m scripts.experiments.bias_exp 50 > scripts/experiments/results/bias50.txt
"""
import argparse
import time

import numpy as np
import torch
from face_alignment.utils import flip, get_preds_fromhm

from facemeasure import FaceDataset, SelectionStatus, build_detector, build_landmark_model, build_selector, load_config
from facemeasure.geometry import apply_affine, crop_square

from ._header import print_header, progress

OFFSETS = [-0.5, 0.0, 0.5]  # added to decoded heatmap coords (cells); the wrapper uses 0, i.e. crop = 4 * p - 0.5
REFS = [165, 195]           # crop_scale = 200 / ref
GT_SHIFTS = (0, 1)          # px subtracted from the stored GT


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=int, help="number of images (the first N of the dataset)")
    ap.add_argument("--config", default="300w_fan", help="config name or path to a .yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = FaceDataset(cfg.data.paths)
    n = min(args.n, len(ds))
    print_header("scripts.experiments.bias_exp", args.config, cfg, ds, n,
                 refs=REFS, heatmap_offsets=OFFSETS, gt_shifts_px=list(GT_SHIFTS), flip_test="always on")
    det, sel = build_detector(cfg.detector), build_selector(cfg.target_selector)
    lm = build_landmark_model(cfg.landmark_model)
    res = {(r, o, g): ([], []) for r in REFS for o in OFFSETS for g in GT_SHIFTS}
    n_ok = 0
    t0 = time.perf_counter()
    for i in range(n):
        s = ds[i]
        r = sel(det(s.image), gt_points=s.points)
        if r.status != SelectionStatus.OK:
            continue
        n_ok += 1
        iod = np.linalg.norm(s.points[36] - s.points[45])
        for ref in REFS:
            c = crop_square(s.image, r.detection.box, scale=200 / ref, size_mode="sum",
                            center_shift_y=lm.center_shift_y, output_size=lm.crop_size)
            x = torch.from_numpy(c.image).permute(2, 0, 1)[None].to(lm.device, torch.float32) / 255
            with torch.inference_mode():
                hm = (lm._heatmaps(x) + flip(lm._heatmaps(flip(x)), is_label=True)) / 2
            p, _, _ = get_preds_fromhm(hm.float().cpu().numpy())
            for o in OFFSETS:
                pts = apply_affine(c.inverse, (p[0] + o) * 4 - 0.5)
                for g in GT_SHIFTS:
                    e = pts - (s.points - g)
                    res[(ref, o, g)][0].append(np.linalg.norm(e, axis=1).mean() / iod)
                    res[(ref, o, g)][1].append(e[17:].mean(0) / iod)
        if i % 10 == 0:
            progress(f"{i} {time.perf_counter() - t0:.0f}s")

    print(f"selected (status ok): {n_ok} of {n}\n")
    print(f"{'ref':>4} {'off':>5} {'GT':>8} {'meanNME%':>9} {'medNME%':>8}  bias/IOD% (x, y) w/o jaw")
    for (ref, o, g), (nm, b) in res.items():
        print(f"{ref:>4} {o:>5} {'stored-1' if g else 'stored':>8} {np.mean(nm) * 100:9.2f} "
              f"{np.median(nm) * 100:8.2f}  {np.round(np.mean(b, 0) * 100, 2)}")


if __name__ == "__main__":
    main()
