"""Mandatory baselines: oracle class-mean and classifier-to-class-mean."""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from config import SEEDS, IMG_SIZE, BATCH_SIZE, MAX_EPOCHS, PATIENCE, WEIGHT_RANGES, RESULTS_DIR
from data import load_labels, load_split
from generate_splits import noise_floor
from metrics import evaluate


def oracle(train_df, test_df):
    """Replace each test image's category by the training mean of that category."""
    means = train_df.groupby("class")["weight"].mean()
    y_true = test_df["weight"].to_numpy()
    y_pred = test_df["class"].map(means).to_numpy()
    return evaluate(y_true, y_pred, "baseline_oracle_class_mean", "-")


def classifier_to_class_mean(train_df, val_df, test_df, seed=42):
    classes = sorted(WEIGHT_RANGES)
    idx = {c: i for i, c in enumerate(classes)}
    means = train_df.groupby("class")["weight"].mean()

    def load(df):
        Xtr, ytr, ctr = load_split(df)
        return Xtr, np.array([idx[c] for c in ctr]), df["weight"].to_numpy(), ctr

    Xtr, ytr, _, _ = load(train_df)
    Xva, yva, _, _ = load(val_df)
    Xte, yte, wte, cte = load(test_df)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    inp = Input(shape=(*IMG_SIZE, 3))
    x = layers.Lambda(preprocess_input)(inp)
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = False
    z = layers.Dense(128, activation="relu")(layers.GlobalAveragePooling2D()(base.output))
    out = layers.Dense(len(classes), activation="softmax")(z)
    clf = models.Model(inp, out)
    clf.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    clf.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
            verbose=0, callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    pred = clf.predict(Xte, verbose=0).argmax(1)
    y_pred = np.array([means[classes[i]] for i in pred])
    return evaluate(wte, y_pred, "baseline_classifier_mean", seed)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_labels()
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]

    floors = noise_floor()
    counts = df["class"].value_counts()
    overall = sum(floors[c] * counts[c] for c in floors) / counts.sum()
    print(f"theoretical noise floor (overall): {overall:.0f} kg")

    rows = [oracle(tr, te)]
    for seed in SEEDS:
        rows.append(classifier_to_class_mean(tr, va, te, seed))
        tf.keras.backend.clear_session()
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/baseline_results.csv", index=False)
    print(pd.DataFrame(rows)[["model", "seed", "MAE", "R2", "F1", "AUC"]].to_string(index=False))
