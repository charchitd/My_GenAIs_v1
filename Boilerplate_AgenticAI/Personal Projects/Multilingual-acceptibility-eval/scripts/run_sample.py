from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.score_pll import PLLScorer
from src.metrics import compute_metrics
from src.plots import plot_mcc_by_language
from src.data import load_sample_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample_acceptability.csv")
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_sample_csv(args.data)
    scorer = PLLScorer(model_name=args.model)

    scores = []
    for s in tqdm(df["sentence"].tolist(), desc="Scoring sentences"):
        scores.append(scorer.score(s))
    df["pll"] = np.array(scores, dtype=np.float32)

    rows = []
    for lang, g in df.groupby("language"):
        m = compute_metrics(g["label"].values, g["pll"].values)
        rows.append({"language": lang, "dataset": g["dataset"].iloc[0], "n": len(g), "accuracy": m.accuracy, "mcc": m.mcc, "auroc": m.auroc})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir/"metrics_by_language.csv", index=False)

    plot_mcc_by_language(metrics_df.fillna(0.0), str(outdir/"figures/performance_by_language.png"))

    print("Saved metrics:", outdir/"metrics_by_language.csv")
    print("Saved plot:", outdir/"figures/performance_by_language.png")


if __name__ == "__main__":
    main()
