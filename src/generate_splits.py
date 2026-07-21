"""Generate reproducible weight labels and a stratified 70/15/15 split."""
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import WEIGHT_RANGES, LABEL_SEED, DATASET_ROOT, LABELS_CSV
from deduplicate import list_images, deduplicate


def assign_weights(df, seed=LABEL_SEED):
    """Sample one weight per image, uniform within its category range (fixed seed)."""
    rng = random.Random(seed)
    df = df.copy()
    df["weight"] = [round(rng.uniform(*WEIGHT_RANGES[c]), 2) for c in df["class"]]
    return df


def noise_floor():
    """Per-category and overall theoretical MAE floor = range/4."""
    return {c: (hi - lo) / 4 for c, (lo, hi) in WEIGHT_RANGES.items()}


def make_splits(df, seed=42):
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df["class"], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["class"], random_state=seed)
    df = df.copy()
    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[te.index, "split"] = "test"
    return df


if __name__ == "__main__":
    df = list_images(DATASET_ROOT)
    df, _ = deduplicate(df)
    df = assign_weights(df)
    df = make_splits(df)
    df["relpath"] = df["path"].str.replace(DATASET_ROOT + "/", "", regex=False)
    df[["relpath", "orig_split", "class", "weight", "split"]].to_csv(LABELS_CSV, index=False)

    floors = noise_floor()
    counts = df["class"].value_counts()
    overall = sum(floors[c] * counts[c] for c in floors) / counts.sum()
    print(f"wrote {LABELS_CSV} ({len(df)} images)")
    print(f"overall theoretical MAE floor: {overall:.0f} kg")
    print(df.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())
