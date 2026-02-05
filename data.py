"""
Dataset utilities for IMDB sentiment CSV from Kaggle.

Expected CSV columns:
- review (text)
- sentiment ('positive'/'negative')

This module normalizes to:
- text
- label (1 for positive, 0 for negative)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    x_train: list[str]
    x_val: list[str]
    x_test: list[str]
    y_train: list[int]
    y_val: list[int]
    y_test: list[int]


def load_imdb_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load Kaggle IMDB Dataset.csv and normalize to ['text', 'label'].

    Raises:
        ValueError: if required columns are missing.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, got {set(df.columns)}")

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["text"] = df["review"].astype(str)
    df["label"] = (df["sentiment"].astype(str).str.lower() == "positive").astype(int)
    return df[["text", "label"]]


def make_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Split:
    """
    Create stratified train/val/test split.

    Note: split is done as:
    - train_temp/test
    - train/val from train_temp
    """
    x = df["text"].tolist()
    y = df["label"].tolist()

    x_train_temp, x_test, y_train_temp, y_test = train_test_split(
        x, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    val_size = val_ratio / (1.0 - test_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_temp, y_train_temp, test_size=val_size, random_state=seed, stratify=y_train_temp
    )

    return Split(
        x_train=x_train, x_val=x_val, x_test=x_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )

