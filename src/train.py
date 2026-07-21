"""Train all model variants over multiple seeds, with optional fine-tuning."""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (SEEDS, BATCH_SIZE, MAX_EPOCHS, PATIENCE, HUBER_DELTA,
                    FINETUNE_LR, FINETUNE_UNFREEZE_LAYERS, RESULTS_DIR)
from data import load_all
from models import build_model
from metrics import evaluate, per_class

FUSIONS = ["cnn_only", "mobilenet_only", "concat", "mul", "concat_mul", "attention"]


def set_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def _callbacks(patience=PATIENCE):
    return [EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau("val_loss", factor=0.5, patience=5, min_lr=1e-6)]


def train_variant(fusion, seed, data):
    (Xtr, ytr, ctr), (Xva, yva, cva), (Xte, yte, cte) = data
    set_seeds(seed)
    model = build_model(fusion)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE, verbose=0, callbacks=_callbacks())
    y_pred = model.predict(Xte, verbose=0).flatten()
    return evaluate(yte, y_pred, fusion, seed), per_class(yte, y_pred, cte, fusion, seed)


def train_proposed_finetuned(seed, data):
    """Proposed variant with augmentation and a second fine-tuning phase."""
    (Xtr, ytr, ctr), (Xva, yva, cva), (Xte, yte, cte) = data
    set_seeds(seed)
    model, base = build_model("concat_mul", return_base=True)
    # phase 1: frozen backbone
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=BATCH_SIZE,
              verbose=0, callbacks=_callbacks(patience=8))
    # phase 2: unfreeze deepest layers, keep BatchNorm frozen
    base.trainable = True
    for layer in base.layers[:-FINETUNE_UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=Adam(FINETUNE_LR),
                  loss=tf.keras.losses.Huber(delta=HUBER_DELTA), metrics=["mae"])
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=BATCH_SIZE,
              verbose=0, callbacks=_callbacks())
    y_pred = model.predict(Xte, verbose=0).flatten()
    return evaluate(yte, y_pred, "proposed_finetuned", seed), per_class(yte, y_pred, cte, "proposed_finetuned", seed)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_all()
    results, pcs = [], []
    for fusion in FUSIONS:
        for seed in SEEDS:
            r, pc = train_variant(fusion, seed, data)
            results.append(r); pcs.append(pc)
            print(f"{fusion} seed={seed}: MAE={r['MAE']:.0f} R2={r['R2']:.3f} F1={r['F1']:.3f}")
            tf.keras.backend.clear_session()
    for seed in SEEDS:
        r, pc = train_proposed_finetuned(seed, data)
        results.append(r); pcs.append(pc)
        print(f"proposed_finetuned seed={seed}: MAE={r['MAE']:.0f} F1={r['F1']:.3f}")
        tf.keras.backend.clear_session()

    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/results_all_runs.csv", index=False)
    pd.concat(pcs).to_csv(f"{RESULTS_DIR}/results_per_class.csv", index=False)
    print(f"\nwrote results to {RESULTS_DIR}/")
