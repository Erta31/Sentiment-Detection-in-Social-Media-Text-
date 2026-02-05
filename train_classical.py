"""
Classical baseline: TF-IDF + LogisticRegression.

Run:
  python /content/imdb_sentiment/src/train_classical.py --csv "/content/imdb_sentiment/data/IMDB Dataset.csv"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data import load_imdb_csv, make_split
from metrics import evaluate_binary, save_confusion_matrix_png


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1)),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to IMDB Dataset.csv")
    p.add_argument("--out_dir", default="/content/imdb_sentiment/outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load_imdb_csv(args.csv)
    split = make_split(df, seed=args.seed)

    model = build_model()
    model.fit(split.x_train, split.y_train)

    val_pred = model.predict(split.x_val).tolist()
    test_pred = model.predict(split.x_test).tolist()

    val_res = evaluate_binary(split.y_val, val_pred)
    test_res = evaluate_binary(split.y_test, test_pred)

    run_dir = Path(args.out_dir) / f"classical_logreg_{_now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.joblib")

    payload = {
        "kind": "classical",
        "model": "logreg",
        "seed": args.seed,
        "val": {
            "accuracy": val_res.accuracy,
            "f1": val_res.f1,
            "precision": val_res.precision,
            "recall": val_res.recall,
            "report": val_res.report,
            "confusion_matrix": val_res.confusion_matrix,
        },
        "test": {
            "accuracy": test_res.accuracy,
            "f1": test_res.f1,
            "precision": test_res.precision,
            "recall": test_res.recall,
            "report": test_res.report,
            "confusion_matrix": test_res.confusion_matrix,
        },
    }

    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_confusion_matrix_png(test_res.confusion_matrix, str(run_dir / "confusion_matrix_test.png"))

    print("Saved run to:", run_dir)
    print(f"TEST acc={test_res.accuracy:.4f} f1={test_res.f1:.4f}")


if __name__ == "__main__":
    main()
