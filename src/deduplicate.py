"""Perceptual-hash deduplication of the merged image pool."""
import os
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

from config import WEIGHT_RANGES, PHASH_DISTANCE, DATASET_ROOT


def list_images(root):
    """Enumerate all images in a deterministic order."""
    rows = []
    for split in ("train", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cdir = os.path.join(split_dir, cls)
            if not os.path.isdir(cdir) or cls not in WEIGHT_RANGES:
                continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({"path": os.path.join(cdir, f),
                                 "orig_split": split, "class": cls})
    return pd.DataFrame(rows)


def _popcount64(x):
    b = x.astype(np.uint64).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1).sum(1)


def deduplicate(df, dist=PHASH_DISTANCE):
    """Greedy first-kept deduplication. Returns (clean_df, report_df)."""
    hashes = []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                hashes.append(int(str(imagehash.phash(im.convert("RGB"))), 16))
        except Exception:
            hashes.append(-1)
    df = df.assign(phash=hashes)
    df = df[df["phash"] >= 0].reset_index(drop=True)

    kept_idx, kept = [], np.empty(0, dtype=np.uint64)
    dropped = []
    for i in df.index:
        h = np.uint64(df.at[i, "phash"])
        if len(kept):
            d = _popcount64(np.bitwise_xor(kept, h))
            j = int(np.argmin(d))
            if d[j] <= dist:
                dropped.append({"dropped": df.at[i, "path"],
                                "kept": df.at[kept_idx[j], "path"],
                                "hamming": int(d[j]),
                                "cross_split": df.at[i, "orig_split"] != df.at[kept_idx[j], "orig_split"],
                                "cross_class": df.at[i, "class"] != df.at[kept_idx[j], "class"]})
                continue
        kept_idx.append(i)
        kept = np.append(kept, h)
    return df.loc[kept_idx].reset_index(drop=True), pd.DataFrame(dropped)


if __name__ == "__main__":
    df = list_images(DATASET_ROOT)
    clean, report = deduplicate(df)
    print(f"kept {len(clean)} / {len(df)}; removed {len(report)} duplicates")
    if len(report):
        print(f"  cross-split leaks: {int(report['cross_split'].sum())}")
        print(f"  cross-class: {int(report['cross_class'].sum())}")
    report.to_csv("results/dedup_report.csv", index=False)
