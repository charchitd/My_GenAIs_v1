from __future__ import annotations

import argparse
from pathlib import Path

from src.score_pll import PLLScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default="The cat sat on the mat.")
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--out", type=str, default="outputs/token_contributions.txt")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scorer = PLLScorer(model_name=args.model)
    contribs = scorer.token_logprobs(args.sentence)

    # Sort by most surprising tokens (lowest log-prob)
    contribs_sorted = sorted(contribs, key=lambda x: x[1])

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Sentence: {args.sentence}\n\n")
        f.write("Token log-prob contributions (lower = more surprising to the model)\n")
        f.write("---------------------------------------------------------------\n")
        for tok, lp in contribs_sorted:
            f.write(f"{tok:>12s} : {lp:.4f}\n")

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
