"""
Predict + explain sentiment using TF-IDF + LogisticRegression pipeline.

Explanation:
- Uses TF-IDF features (words/ngrams)
- Computes contribution per feature: tfidf_value * coef
- Shows top positive and top negative contributing terms

Run:
  %cd /content/imdb_sentiment
  python -m src.predict_explain --outputs_dir outputs --text "Your review here..." --topk 15
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _find_latest_run(outputs_dir: Path, prefix: str = "classical_") -> Path:
    runs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not runs:
        raise FileNotFoundError(f"No run folders found in {outputs_dir} with prefix {prefix!r}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _label_name(y: int) -> str:
    return "positive" if y == 1 else "negative"


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def explain_logreg_tfidf_pipeline(model, text: str, topk: int = 15) -> dict:
    """
    Works for Pipeline(tfidf -> LogisticRegression).

    Returns dict with:
      pred, p_pos, decision, intercept, top_pos_terms, top_neg_terms
    """
    # Try to locate steps
    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a scikit-learn Pipeline with named_steps (tfidf + clf).")

    tfidf = model.named_steps.get("tfidf")
    clf = model.named_steps.get("clf")

    if tfidf is None or clf is None:
        raise ValueError("Pipeline must have steps named 'tfidf' and 'clf'.")

    if not hasattr(tfidf, "get_feature_names_out"):
        raise TypeError("TF-IDF vectorizer missing get_feature_names_out().")

    # Vectorize
    X = tfidf.transform([text])  # sparse (1, n_features)
    feature_names = tfidf.get_feature_names_out()

    # Decision and probability
    if hasattr(clf, "decision_function"):
        decision = float(clf.decision_function(X)[0])
    else:
        # Fallback: from predict_proba odds
        proba = float(clf.predict_proba(X)[0][1])
        decision = float(np.log(proba / (1.0 - proba + 1e-12) + 1e-12))

    p_pos = float(clf.predict_proba(X)[0][1]) if hasattr(clf, "predict_proba") else _sigmoid(decision)
    pred = 1 if p_pos >= 0.5 else 0

    # Contributions: tfidf_value * coef
    coef = clf.coef_.reshape(-1)  # (n_features,)
    x_row = X.tocoo()
    contrib = {}
    for j, v in zip(x_row.col, x_row.data):
        contrib[j] = float(v) * float(coef[j])

    # Sort contributions
    # Positive contributions push towards positive; negative contributions push towards negative.
    items = [(int(j), c) for j, c in contrib.items()]
    items_sorted = sorted(items, key=lambda t: t[1], reverse=True)

    top_pos = [(feature_names[j], c) for j, c in items_sorted[:topk] if c > 0]
    top_neg = [(feature_names[j], c) for j, c in reversed(items_sorted[-topk:]) if c < 0]
    # The "reversed(items_sorted[-topk:])" ensures most negative first

    intercept = float(getattr(clf, "intercept_", [0.0])[0])

    return {
        "pred": pred,
        "p_pos": p_pos,
        "decision": decision,
        "intercept": intercept,
        "top_pos_terms": top_pos,
        "top_neg_terms": top_neg,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--text", required=True)
    p.add_argument("--topk", type=int, default=15)
    args = p.parse_args()

    import joblib

    outputs_dir = Path(args.outputs_dir)
    run_dir = _find_latest_run(outputs_dir, prefix="classical_")
    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        # If you still have the old single-file save
        alt = outputs_dir / "classical_model.joblib"
        if alt.exists():
            model_path = alt
            run_dir = outputs_dir
        else:
            raise FileNotFoundError(f"Missing {model_path} (and no classical_model.joblib fallback).")

    model = joblib.load(model_path)

    res = explain_logreg_tfidf_pipeline(model, args.text, topk=args.topk)

    print("Using model:", model_path)
    print("\nTEXT:", args.text)
    print(f"\nPRED: {_label_name(res['pred'])}  (p_pos={res['p_pos']:.4f}, decision={res['decision']:.4f})")

    print("\nTop terms pushing POSITIVE:")
    if res["top_pos_terms"]:
        for term, score in res["top_pos_terms"]:
            print(f"  + {term:<20}  {score:+.6f}")
    else:
        print("  (none found)")

    print("\nTop terms pushing NEGATIVE:")
    if res["top_neg_terms"]:
        for term, score in res["top_neg_terms"]:
            print(f"  - {term:<20}  {score:+.6f}")
    else:
        print("  (none found)")


if __name__ == "__main__":
    main()
