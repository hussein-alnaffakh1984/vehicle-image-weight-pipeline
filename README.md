# Vehicle Image Weight Pipeline

Source code for image preprocessing, reproducible data splitting, model training,
and evaluation of a two-branch CNN + MobileNetV2 feature-fusion regressor for
category-level vehicle weight estimation and camera-based overload pre-screening.

## Environment

- Python 3.10+
- TensorFlow 2.x

## Installation

```bash
pip install -r requirements.txt
```

## Repository layout

```
src/
  config.py          # central configuration (seeds, paths, hyperparameters)
  deduplicate.py     # perceptual-hash (pHash) duplicate detection
  generate_splits.py # weight labeling + stratified 70/15/15 split
  data.py            # image loading helpers
  models.py          # six architectures (single branches + four fusions)
  train.py           # trains all variants over seeds, with fine-tuning
  baselines.py       # oracle and classifier-to-class-mean baselines
  metrics.py         # regression + screening metrics (incl. WAPE)
  evaluate.py        # aggregates raw results into mean/SD tables
splits/
  labels_master.csv  # per-image class, weight, and train/val/test assignment
results/             # written by the scripts
```

## Data

Image files are not redistributed. The dataset sources are listed in the
associated manuscript. Place the images under `data/Dataset/{train,test}/<class>/`
to match `splits/labels_master.csv`, or regenerate the labels with
`generate_splits.py`.

## Reproducing the experiments

```bash
# 1. (optional) regenerate labels and splits from the images
python src/generate_splits.py

# 2. baselines (oracle + classifier-to-class-mean)
python src/baselines.py

# 3. train all variants over 3 seeds, with fine-tuning of the proposed model
python src/train.py

# 4. aggregate into mean/SD tables
python src/evaluate.py
```

Fixed random seeds (Python, NumPy, TensorFlow) are used for reproducibility.
Full bitwise determinism on GPU is not enforced, so results may vary by a small
margin across runs; report metrics as mean +/- SD over the provided seeds.

## License

Released under the MIT License (see `LICENSE`).
