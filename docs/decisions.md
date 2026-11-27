# Decisions

Short log of pipeline decisions, newest last. Values are checked against the code and config at the time
of writing; "pending" means decided but not applied yet. Experiment scripts and saved outputs live in
`scripts/experiments/`. All 300W numbers are on `data/processed/300w.parquet` (599 images).


## 2026-11-23: 300W GT is 1-based: subtract 1 px in prepare_300w.py (applied 2026-11-27)
- Why: 300W `.pts` follow the Matlab 1-based convention; our pixel centers are 0-based, so stored GT sat +1 px right/down.
- Evidence: first 50 images (all selected), crop 200/195: mean NME 6.46% → 5.99% with GT − 1; bias without jaw (x, y) −1.70 / −1.51 → −0.76 / −0.58 % of inter-ocular distance. Same direction at 200/165 (7.43% → 6.90%).
- Applied: parquet regenerated (all 599 samples exactly −1 px, `coordinates` attribute added); rerun with GT as stored, max_side 1280, reproduces 5.99% mean / 5.47% median at 200/195.
- Source: `scripts/experiments/bias_exp.py`, output before the fix `scripts/experiments/results/bias50.txt`.

## 2026-11-23: FAN crop_scale 200/195 instead of 200/165 with YuNet boxes
- Why: 200/165 is the package's setting for YuNet boxes, but with our YuNet boxes the 195 crop is more accurate.
- Evidence: first 50 images, GT 1-based: mean NME 5.99% (195) vs 6.90% (165), median 5.47% vs 6.20%; GT as stored: 6.46% vs 7.43%.
- Source: `scripts/experiments/bias_exp.py`, output before the GT fix `scripts/experiments/results/bias50.txt`.

## Open questions
- **Multi-scale YuNet** (scales `[1.0, 0.5]`): retrying at half size when nothing passes 0.6 recovered 3 of the 5 failures (00534, 00242, 00357) at scores 0.78–0.90; 00101 and 00413 still fail. Not implemented.
- **Tune crop_scale on a separate set** (300W train subsets: LFPW / HELEN / AFW trainsets) instead of the test images it is evaluated on.
- **Residual upward-left bias** of about −0.8 / −0.6 % of inter-ocular distance (x, y, without jaw) remains with crop 200/195 and 1-based GT (bias50: −0.76 / −0.58; pkg50 variant b: −0.80 / −0.55).
