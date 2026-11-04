
import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def analyze(dataset, model, split="test"):
    pretrained_path = (
        RESULTS / f"{dataset}_pretrained_{split}_queries.csv"
    )
    finetuned_path = (
        RESULTS / f"{dataset}_{model}_{split}_queries.csv"
    )

    for path in (pretrained_path, finetuned_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing results: {path}")

    pretrained = pd.read_csv(
        pretrained_path,
        dtype={"query_id": str, "top_candidate_id": str},
    )

    finetuned = pd.read_csv(
        finetuned_path,
        dtype={"query_id": str, "top_candidate_id": str},
    )

    comparison = pretrained.merge(
        finetuned,
        on="query_id",
        suffixes=("_pretrained", "_finetuned"),
        validate="one_to_one",
    )

    if (
        len(comparison) != len(pretrained)
        or len(comparison) != len(finetuned)
    ):
        raise ValueError("The evaluated query sets do not match.")

    comparison["rank_change"] = (
        comparison["first_relevant_rank_pretrained"]
        - comparison["first_relevant_rank_finetuned"]
    )

    comparison["rr_change"] = (
        comparison["reciprocal_rank_finetuned"]
        - comparison["reciprocal_rank_pretrained"]
    )

    improved = int((comparison["rank_change"] > 0).sum())
    worsened = int((comparison["rank_change"] < 0).sum())
    unchanged = int((comparison["rank_change"] == 0).sum())

    print("=" * 55)
    print("Duplicate Bug Report Error Analysis")
    print("=" * 55)
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Split: {split}")
    print(f"Queries compared: {len(comparison)}")

    print("\nRanking changes")
    print("-" * 35)
    print(f"Improved: {improved}")
    print(f"Worsened: {worsened}")
    print(f"Unchanged: {unchanged}")

    print("\nPaired retrieval metrics")
    print("-" * 35)

    for metric in (
        "hit_at_1",
        "hit_at_5",
        "hit_at_10",
        "reciprocal_rank",
    ):
        before = comparison[f"{metric}_pretrained"].mean()
        after = comparison[f"{metric}_finetuned"].mean()

        print(
            f"{metric}: "
            f"{before:.4f} -> {after:.4f} "
            f"(difference: {after - before:+.4f})"
        )

    print("\nExact McNemar tests")
    print("-" * 35)

    for k in (1, 5, 10):
        old = comparison[f"hit_at_{k}_pretrained"]
        new = comparison[f"hit_at_{k}_finetuned"]

        both_success = int(((old == 1) & (new == 1)).sum())
        lost = int(((old == 1) & (new == 0)).sum())
        gained = int(((old == 0) & (new == 1)).sum())
        both_failed = int(((old == 0) & (new == 0)).sum())

        table = [
            [both_success, lost],
            [gained, both_failed],
        ]

        test = mcnemar(table, exact=True)

        print(f"\nHit@{k}")
        print(f"New successes: {gained}")
        print(f"Lost successes: {lost}")
        print(f"Net successes: {gained - lost:+d}")
        print(f"p-value: {test.pvalue:.6f}")

    print("\nLargest improvements")
    print("-" * 35)

    columns = [
        "query_id",
        "first_relevant_rank_pretrained",
        "first_relevant_rank_finetuned",
        "rr_change",
    ]

    print(
        comparison.nlargest(10, "rr_change")[
            columns
        ].to_string(index=False)
    )

    print("\nLargest regressions")
    print("-" * 35)

    print(
        comparison.nsmallest(10, "rr_change")[
            columns
        ].to_string(index=False)
    )

    output_path = (
        RESULTS / f"{dataset}_{model}_{split}_error_comparison.csv"
    )

    comparison.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )

    args = parser.parse_args()

    analyze(
        dataset=args.dataset,
        model=args.model,
        split=args.split,
    )


if __name__ == "__main__":
    main()
