"""Regression and overload-screening metrics."""
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

from config import THRESHOLD_KG


def wape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum() * 100


def evaluate(y_true, y_pred, name, seed):
    """Return a dict of regression + screening metrics."""
    yt = (y_true > THRESHOLD_KG).astype(int)
    yp = (y_pred > THRESHOLD_KG).astype(int)
    fpr, tpr, _ = roc_curve(yt, y_pred)
    return {
        "model": name, "seed": seed,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "WAPE": float(wape(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Acc": accuracy_score(yt, yp),
        "Prec": precision_score(yt, yp, zero_division=0),
        "Rec": recall_score(yt, yp, zero_division=0),
        "F1": f1_score(yt, yp, zero_division=0),
        "AUC": auc(fpr, tpr),
    }


def per_class(y_true, y_pred, classes, name, seed):
    import pandas as pd
    rows = []
    for c in sorted(set(classes)):
        m = classes == c
        rows.append({"model": name, "seed": seed, "class": c,
                     "MAE": float(np.abs(y_true[m] - y_pred[m]).mean()),
                     "MAPE": float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100),
                     "WAPE": float(wape(y_true[m], y_pred[m]))})
    return pd.DataFrame(rows)
