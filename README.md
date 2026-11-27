Measuring face features using computer vision methods.

## Setup

```bash
# 1. PyTorch first, matching your hardware (see https://pytorch.org/get-started/locally/), e.g. CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 2. The `facemeasure` package and its dependencies (pyproject.toml), as an editable install:
pip install -e .
# 3. Model weights into models/ (checksum-verified):
python -m scripts.download_models
# 4. (optional) 300W test set into data/raw/300W_FIXED, then convert it to data/processed/300w.parquet:
pip install -e ".[data]"
python -m scripts.download_300w
python -m scripts.prepare_300w
```
