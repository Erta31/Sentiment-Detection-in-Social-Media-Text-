"""
Deep learning sentiment model (Keras): CNN or LSTM.

Fix: TF+Py3.12 can crash when tf.saved_model.save(TextVectorization). We save vectorizer
as (config.json + weights.npz) instead.

Run:
  python /content/imdb_sentiment/src/train_deep.py --csv "/content/imdb_sentiment/data/IMDB Dataset.csv" --arch cnn
  python /content/imdb_sentiment/src/train_deep.py --csv "/content/imdb_sentiment/data/IMDB Dataset.csv" --arch lstm
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import load_imdb_csv, make_split
from metrics import evaluate_binary, save_confusion_matrix_png


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_vectorizer(x_train: list[str], max_tokens: int, seq_len: int) -> tf.keras.layers.TextVectorization:
    vec = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize=None,
    )
    vec.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(256))
    return vec


def save_text_vectorizer(vec: tf.keras.layers.TextVectorization, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vec.get_config(), indent=2), encoding="utf-8")
    weights = vec.get_weights()
    np.savez(out_dir / "weights.npz", **{f"arr_{i}": w for i, w in enumerate(weights)})


def build_model(arch: str, vocab_size: int, seq_len: int, emb_dim: int, dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len,), dtype=tf.int64)
    x = tf.keras.layers.Embedding(vocab_size, emb_dim)(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)

    if arch == "cnn":
        x = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif arch == "lstm":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:
        raise ValueError("arch must be one of: cnn, lstm")

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def make_ds(texts: list[str], labels: list[int], batch: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((texts, np.array(labels, dtype=np.int32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(20000, len(texts)), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", default="/content/imdb_sentiment/outputs")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--arch", choices=["cnn", "lstm"], default="cnn")
    p.add_argument("--max_tokens", type=int, default=20000)
    p.add_argument("--seq_len", type=int, default=250)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=6)
    args = p.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    df = load_imdb_csv(args.csv)
    split = make_split(df, seed=args.seed)

    vectorizer = make_vectorizer(split.x_train, args.max_tokens, args.seq_len)

    def vectorize_map(x, y):
        return vectorizer(x), tf.cast(y, tf.float32)

    train_ds = make_ds(split.x_train, split.y_train, args.batch, True, args.seed).map(
        vectorize_map, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = make_ds(split.x_val, split.y_val, args.batch, False, args.seed).map(
        vectorize_map, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = make_ds(split.x_test, split.y_test, args.batch, False, args.seed).map(
        vectorize_map, num_parallel_calls=tf.data.AUTOTUNE
    )

    model = build_model(args.arch, args.max_tokens, args.seq_len, args.emb_dim, args.dropout)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    y_val_prob = model.predict(val_ds, verbose=0).reshape(-1)
    y_test_prob = model.predict(test_ds, verbose=0).reshape(-1)
    y_val_pred = (y_val_prob >= 0.5).astype(int).tolist()
    y_test_pred = (y_test_prob >= 0.5).astype(int).tolist()

    val_res = evaluate_binary(split.y_val, y_val_pred)
    test_res = evaluate_binary(split.y_test, y_test_pred)

    run_dir = Path(args.out_dir) / f"deep_{args.arch}_{_now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model.save(run_dir / "model.keras")
    save_text_vectorizer(vectorizer, run_dir / "text_vectorizer")

    payload = {
        "kind": "deep",
        "arch": args.arch,
        "seed": args.seed,
        "hyperparams": {
            "max_tokens": args.max_tokens,
            "seq_len": args.seq_len,
            "emb_dim": args.emb_dim,
            "dropout": args.dropout,
            "batch": args.batch,
            "epochs": args.epochs,
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
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
