"""
Generate visualizations for dataset + runs.

Run:
  %cd /content/imdb_sentiment
  python -m src.make_plots --csv "/content/imdb_sentiment/data/IMDB Dataset.csv" --outputs_dir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_imdb_csv
from src.viz import (
    plot_label_distribution,
    plot_review_length_hist,
    plot_confusion_matrix,
    plot_run_metrics_bars,
    read_json,
)


def find_latest_run(outputs_dir: Path) -> Path | None:
    runs = [d for d in outputs_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--out_plots", default="outputs/plots")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir)
    plots_dir = Path(args.out_plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Dataset plots
    df = load_imdb_csv(args.csv)
    plot_label_distribution(df, plots_dir / "label_distribution.png")
    plot_review_length_hist(df, plots_dir / "review_length_hist.png")

    # Latest run confusion matrix plot
    latest = find_latest_run(outputs_dir)
    if latest:
        m = read_json(latest / "metrics.json")
        cm = m.get("test", {}).get("confusion_matrix")
        if cm:
            plot_confusion_matrix(cm, plots_dir / f"{latest.name}_confusion_matrix.png", title=f"{latest.name} (test)")
            print("Saved confusion matrix for:", latest.name)

    # Run comparison plots (requires compare.py to be run at least once)
    summary = outputs_dir / "summary.json"
    if summary.exists():
        plot_run_metrics_bars(summary, plots_dir)
        print("Saved run comparison charts from summary.json")
    else:
        print("summary.json not found. Run compare.py first to generate run comparison charts.")

    print("All plots saved to:", plots_dir)


if __name__ == "__main__":
    main()
