"""
Dataset sanity checks for IMDB Kaggle CSV.

Run:
  python -m src.check_dataset --csv "/content/imdb_sentiment/data/IMDB Dataset.csv"
"""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter

import pandas as pd

from src.data import load_imdb_csv


POS_WORDS = {
    "good","great","excellent","amazing","awesome","love","loved","like","liked",
    "best","wonderful","brilliant","fantastic","enjoy","enjoyed","perfect",
    "superb","favorite","fun","masterpiece","beautiful","positive"
}
NEG_WORDS = {
    "bad","terrible","awful","worst","boring","hate","hated","dislike","poor",
    "waste","stupid","dull","annoying","horrible","disappoint","disappointed",
    "mess","garbage","ridiculous","negative"
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", str(text).lower())


def polarity_score(text: str) -> int:
    toks = tokenize(text)
    c = Counter(toks)
    pos = sum(c[w] for w in POS_WORDS if w in c)
    neg = sum(c[w] for w in NEG_WORDS if w in c)
    return pos - neg


def predict_from_score(score: int, threshold: int = 0) -> int:
    return 1 if score > threshold else 0


def show_samples(df: pd.DataFrame, label: int, k: int, seed: int) -> None:
    subset = df[df["label"] == label]
    idxs = subset.sample(n=min(k, len(subset)), random_state=seed).index.tolist()
    name = "POSITIVE" if label == 1 else "NEGATIVE"
    print(f"\n=== {name} samples ({len(idxs)}) ===")
    for i in idxs:
        text = df.loc[i, "text"]
        s = polarity_score(text)
        print(f"\n--- idx={i} lex_score={s} ---")
        print(text[:500].replace("\n", " ") + ("..." if len(text) > 500 else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_suspicious", type=int, default=15)
    args = p.parse_args()

    df = load_imdb_csv(args.csv)
    print("Rows:", len(df))
    print("Label distribution:", df["label"].value_counts().to_dict())

    show_samples(df, label=0, k=args.samples, seed=args.seed)
    show_samples(df, label=1, k=args.samples, seed=args.seed + 1)

    scores = df["text"].map(polarity_score)
    pred = scores.map(lambda s: predict_from_score(int(s), threshold=0))

    agree = (pred.values == df["label"].values).mean()
    neg_mask = df["label"].values == 0
    pos_mask = df["label"].values == 1
    neg_agree = (pred.values[neg_mask] == 0).mean() if neg_mask.any() else float("nan")
    pos_agree = (pred.values[pos_mask] == 1).mean() if pos_mask.any() else float("nan")

    print("\n=== Lexicon sanity check (heuristic) ===")
    print(f"Overall agreement: {agree:.4f}")
    print(f"Negative agreement: {neg_agree:.4f}")
    print(f"Positive agreement: {pos_agree:.4f}")

    signed = scores.copy()
    signed[df["label"] == 0] = -signed[df["label"] == 0]
    suspicious = signed.sort_values(ascending=True).head(args.top_suspicious)

    print(f"\n=== Top {args.top_suspicious} suspicious rows (heuristic) ===")
    for idx in suspicious.index:
        label = int(df.loc[idx, "label"])
        raw_score = int(scores.loc[idx])
        print(f"\nidx={idx} label={'pos' if label==1 else 'neg'} lex_score={raw_score}")
        t = df.loc[idx, "text"]
        print(t[:500].replace("\n", " ") + ("..." if len(t) > 500 else ""))


if __name__ == "__main__":
    main()
