"""Data loading from the labels file."""
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from config import IMG_SIZE, LABELS_CSV, DATASET_ROOT


def load_labels(labels_csv=LABELS_CSV, dataset_root=DATASET_ROOT):
    df = pd.read_csv(labels_csv)
    df["path"] = dataset_root + "/" + df["relpath"]
    return df


def load_split(df):
    """Return (X, y, classes) for a split subframe."""
    X = np.stack([img_to_array(load_img(p, target_size=IMG_SIZE)) for p in df["path"]])
    return X.astype("float32"), df["weight"].to_numpy("float32"), df["class"].to_numpy()


def load_all(labels_csv=LABELS_CSV):
    df = load_labels(labels_csv)
    return (load_split(df[df.split == "train"]),
            load_split(df[df.split == "val"]),
            load_split(df[df.split == "test"]))
