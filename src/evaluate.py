"""Summarize raw results into mean +/- SD tables."""
import pandas as pd

from config import RESULTS_DIR

METRICS = ["MAE", "RMSE", "MAPE", "WAPE", "R2", "Acc", "Prec", "Rec", "F1", "AUC"]


def summarize(results_csv=f"{RESULTS_DIR}/results_all_runs.csv",
              baseline_csv=f"{RESULTS_DIR}/baseline_results.csv"):
    frames = [pd.read_csv(results_csv)]
    try:
        frames.append(pd.read_csv(baseline_csv))
    except FileNotFoundError:
        pass
    runs = pd.concat(frames, ignore_index=True)
    cols = [m for m in METRICS if m in runs.columns]
    mean = runs.groupby("model")[cols].mean().round(3)
    std = runs.groupby("model")[cols].std().round(3)
    mean.to_csv(f"{RESULTS_DIR}/summary_mean.csv")
    std.to_csv(f"{RESULTS_DIR}/summary_std.csv")
    return mean, std


if __name__ == "__main__":
    mean, std = summarize()
    print("=== mean over seeds ===")
    print(mean.to_string())
    print("\n=== std over seeds ===")
    print(std.to_string())
