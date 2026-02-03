"""
Visualization utilities for IMDB sentiment project.

Produces:
- label distribution plot
- review length histogram
- confusion matrix plot (from metrics.json)
- run comparison bar charts (from outputs/summary.json)

All plots are saved to disk (PNG).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_label_distribution(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    counts = df["label"].value_counts().sort_index()
    labels = ["neg", "pos"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Label distribution")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_review_length_hist(df: pd.DataFrame, out_path: str | Path, bins: int = 50) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    lengths = df["text"].astype(str).map(lambda s: len(s.split())).to_numpy()

    plt.figure()
    plt.hist(lengths, bins=bins)
    plt.title("Review length (words)")
    plt.xlabel("words per review")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: Sequence[Sequence[int]], out_path: str | Path, title: str = "Confusion Matrix") -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    arr = np.array(cm, dtype=int)

    plt.figure()
    plt.imshow(arr)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["neg", "pos"])
    plt.yticks([0, 1], ["neg", "pos"])

    for (i, j), v in np.ndenumerate(arr):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def plot_run_metrics_bars(summary_json: str | Path, out_dir: str | Path) -> None:
    """
    Uses outputs/summary.json (written by compare.py) to create bar charts
    for acc and f1 across runs.
    """
    summary_json = Path(summary_json)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    data = read_json(summary_json)
    rows = data.get("rows", [])
    if not rows:
        raise ValueError(f"No rows found in {summary_json}")

    # Keep order as given (already sorted by compare.py)
    names = [r["run"] for r in rows]
    acc = [float(r["test_accuracy"]) for r in rows]
    f1 = [float(r["test_f1"]) for r in rows]

    # ACC plot
    plt.figure()
    plt.bar(range(len(names)), acc)
    plt.title("Test Accuracy by Run")
    plt.xlabel("run")
    plt.ylabel("accuracy")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "runs_accuracy.png", dpi=160)
    plt.close()

    # F1 plot
    plt.figure()
    plt.bar(range(len(names)), f1)
    plt.title("Test F1 by Run")
    plt.xlabel("run")
    plt.ylabel("f1")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "runs_f1.png", dpi=160)
    plt.close()
