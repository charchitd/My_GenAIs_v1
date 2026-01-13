from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_mcc_by_language(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("language")

    plt.figure()
    plt.bar(df["language"], df["mcc"])
    plt.ylabel("MCC")
    plt.title("Acceptability prediction (PLL threshold) by language")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
