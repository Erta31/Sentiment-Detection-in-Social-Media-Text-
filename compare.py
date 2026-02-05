"""
Compare all runs in outputs directory (classical vs deep).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _summarize(run_dir: Path) -> Dict[str, Any]:
    m = _read_json(run_dir / "metrics.json")
    test = m.get("test", {})
    kind = m.get("kind", "unknown")
    name = m.get("model", m.get("arch", "unknown"))

    return {
        "run": run_dir.name,
        "kind": kind,
        "model_or_arch": name,
        "test_accuracy": float(test.get("accuracy", 0.0)),
        "test_f1": float(test.get("f1", 0.0)),
        "test_precision": float(test.get("precision", 0.0)),
        "test_recall": float(test.get("recall", 0.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default="/content/imdb_sentiment/outputs")
    args = p.parse_args()

    out_dir = Path(args.outputs_dir)
    runs = sorted([d for d in out_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists()])

    rows = [_summarize(r) for r in runs]
    rows.sort(key=lambda r: (r["test_f1"], r["test_accuracy"]), reverse=True)

    if not rows:
        print("No runs found. Train models first.")
        return

    print("run\tkind\tmodel_or_arch\tacc\tf1\tprec\trec")
    for r in rows:
        print(
            f"{r['run']}\t{r['kind']}\t{r['model_or_arch']}\t"
            f"{r['test_accuracy']:.4f}\t{r['test_f1']:.4f}\t{r['test_precision']:.4f}\t{r['test_recall']:.4f}"
        )

    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("Wrote:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
