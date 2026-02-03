"""
Predict sentiment for a review using the latest saved model run.

Supports:
- classical TF-IDF + LogisticRegression saved as outputs/classical_*/model.joblib
- deep Keras model saved as outputs/deep_*/model.keras (optional)

Run examples:
  %cd /content/imdb_sentiment
  python -m src.predict_review --outputs_dir outputs --use_example
  python -m src.predict_review --outputs_dir outputs --text "This movie was amazing..."

Notes:
- If multiple runs exist, we pick the most recently modified run folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _find_latest_run(outputs_dir: Path, prefix: str) -> Path:
    runs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not runs:
        raise FileNotFoundError(f"No run folders found in {outputs_dir} with prefix {prefix!r}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _predict_classical(run_dir: Path, text: str) -> Tuple[int, float]:
    import joblib

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")

    model = joblib.load(model_path)

    # LogisticRegression pipeline usually supports predict_proba
    if hasattr(model, "predict_proba"):
        proba_pos = float(model.predict_proba([text])[0][1])
    elif hasattr(model, "decision_function"):
        # Map decision score to pseudo-probability via sigmoid
        score = float(model.decision_function([text])[0])
        proba_pos = float(1.0 / (1.0 + np.exp(-score)))
    else:
        # Fallback
        pred = int(model.predict([text])[0])
        return pred, float("nan")

    pred = 1 if proba_pos >= 0.5 else 0
    return pred, proba_pos


def _load_text_vectorizer(vec_dir: Path):
    """
    Rebuild TextVectorization from (config.json + weights.npz).
    """
    import json
    import tensorflow as tf

    cfg_path = vec_dir / "config.json"
    w_path = vec_dir / "weights.npz"
    if not cfg_path.exists() or not w_path.exists():
        raise FileNotFoundError(f"Missing vectorizer files in: {vec_dir}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vec = tf.keras.layers.TextVectorization.from_config(cfg)

    data = np.load(w_path)
    weights = [data[f"arr_{i}"] for i in range(len(data.files))]
    vec.set_weights(weights)
    return vec


def _predict_deep(run_dir: Path, text: str) -> Tuple[int, float]:
    import tensorflow as tf

    model_path = run_dir / "model.keras"
    vec_dir = run_dir / "text_vectorizer"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")

    model = tf.keras.models.load_model(model_path)
    vec = _load_text_vectorizer(vec_dir)

    x = vec(tf.constant([text]))
    proba_pos = float(model.predict(x, verbose=0).reshape(-1)[0])
    pred = 1 if proba_pos >= 0.5 else 0
    return pred, proba_pos


def _label_name(y: int) -> str:
    return "positive" if y == 1 else "negative"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--model_kind", choices=["classical", "deep"], default="classical")
    p.add_argument("--text", default=None)
    p.add_argument("--use_example", action="store_true")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir)

    if args.model_kind == "classical":
        run_dir = _find_latest_run(outputs_dir, "classical_")
        predict_fn = lambda t: _predict_classical(run_dir, t)
    else:
        run_dir = _find_latest_run(outputs_dir, "deep_")
        predict_fn = lambda t: _predict_deep(run_dir, t)

    print("Using run:", run_dir)

    examples = [
        "I absolutely loved this movie. The acting was brilliant and the story was amazing.",
        "This was a complete waste of time. Boring plot, terrible acting, and I hated it.",
    ]

    if args.use_example:
        for t in examples:
            pred, ppos = predict_fn(t)
            print("\nTEXT:", t)
            print(f"PRED: {_label_name(pred)}  (p_pos={ppos:.4f})")
        return

    if not args.text:
        raise SystemExit("Provide --text '...' or use --use_example")

    pred, ppos = predict_fn(args.text)
    print("\nTEXT:", args.text)
    print(f"PRED: {_label_name(pred)}  (p_pos={ppos:.4f})")


if __name__ == "__main__":
    main()
