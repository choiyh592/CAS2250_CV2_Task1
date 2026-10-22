# -*- coding: utf-8 -*-
"""Section 2 (EDA) of the report: dataset statistics and Figure 1. Run with no arguments.

    results/eda.json                          every number in Section 2
    results/figures/fig00_mnist_samples.png   Figure 1 (SAMPLES_PER_CLASS per class, seed SEED)

Works on the raw uint8 images (0-255), before any normalization.
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# ---------------------------------------------------------------- settings
DATA_ROOT = "./data"
OUT_JSON = Path("results/eda.json")
OUT_FIG = Path("results/figures/fig00_mnist_samples.png")
SEED = 2250                    # Figure 1 sample selection
SAMPLES_PER_CLASS = 3
FG_THRESHOLD = 127             # foreground = pixel value > 127


def load(train):
    ds = datasets.MNIST(DATA_ROOT, train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy()


def pct(c, n):
    """Percentage rounded half-up on the exact value (5,949 / 60,000 = 9.915% -> 9.92)."""
    return float((Decimal(int(c)) * 100 / Decimal(int(n))).quantize(Decimal("0.01"), ROUND_HALF_UP))


def fg_width(x):
    """Horizontal extent (in pixels) of the foreground in each image; 0 if none."""
    cols = (x > FG_THRESHOLD).any(1)                       # (N, 28): column has foreground
    has = cols.any(1)
    left = np.argmax(cols, 1)
    right = cols.shape[1] - 1 - np.argmax(cols[:, ::-1], 1)
    return np.where(has, right - left + 1, 0)


def stats(x, y):
    counts = np.bincount(y, minlength=10)
    f = x / 255.0
    fg = x > FG_THRESHOLD
    w = fg_width(x)
    flat = x.reshape(len(x), -1)
    return {
        "n_images": int(len(x)),
        "image_shape": list(x.shape[1:]),
        "all_images_28x28": bool(x.shape[1:] == (28, 28)),
        "class_counts": counts.tolist(),
        "class_percent": [pct(c, len(y)) for c in counts],
        "class_count_std_population": round(float(counts.std()), 2),
        "pixel_mean_0to1": round(float(f.mean()), 6),
        "pixel_std_0to1": round(float(f.std()), 6),
        "zero_pixel_percent": round(100 * float((x == 0).mean()), 2),
        "all_black_images": int((flat.max(1) == 0).sum()),
        "exact_duplicate_images": int(len(x) - len({r.tobytes() for r in flat})),
        f"foreground_percent_per_class_gt{FG_THRESHOLD}": [round(100 * float(fg[y == k].mean()), 2) for k in range(10)],
        f"foreground_width_median_per_class_gt{FG_THRESHOLD}": [float(np.median(w[y == k])) for k in range(10)],
    }


def figure(x, y):
    rng = np.random.default_rng(SEED)
    picks = [rng.choice(np.flatnonzero(y == k), SAMPLES_PER_CLASS, replace=False) for k in range(10)]
    fig, ax = plt.subplots(SAMPLES_PER_CLASS, 10, figsize=(10, SAMPLES_PER_CLASS + 0.4))
    for k in range(10):
        for r, i in enumerate(picks[k]):
            ax[r, k].imshow(x[i], cmap="gray"); ax[r, k].axis("off")
        ax[0, k].set_title(str(k))
    fig.suptitle(f"MNIST training samples ({SAMPLES_PER_CLASS} per class, seed {SEED})", y=1.02)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight"); plt.close(fig)
    return {str(k): picks[k].tolist() for k in range(10)}


if __name__ == "__main__":
    xtr, ytr = load(True)
    xte, yte = load(False)
    out = {"train": stats(xtr, ytr), "test": stats(xte, yte),
           "figure1_seed": SEED, "figure1_train_indices": figure(xtr, ytr)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: out[k] for k in ("train", "test")}, indent=2))
    print(f"-> {OUT_JSON}, {OUT_FIG}")
