
import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )
    args = parser.parse_args()

    dataset = args.dataset

    comparison = pd.read_csv(
        RESULTS
        / f"{dataset}_{dataset}-pilot_test_error_comparison.csv",
        dtype={"query_id": str},
    )

    issues = pd.read_csv(
        PROCESSED / f"{dataset}_issues_grouped.csv",
        dtype={"issue_id": str},
    )

    issues = issues.drop_duplicates("issue_id")

    queries = comparison.merge(
        issues[["issue_id", "text"]],
        left_on="query_id",
        right_on="issue_id",
        how="left",
        validate="one_to_one",
    )

    if queries["text"].isna().any():
        raise ValueError("Some query texts were not found.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Calculating token lengths...")

    queries["token_length"] = [
        len(
            tokenizer.encode(
                str(text),
                add_special_tokens=True,
                truncation=False,
            )
        )
        for text in queries["text"]
    ]

    queries["truncated"] = (
        queries["token_length"] > MAX_TOKENS
    )

    queries["rank_change"] = (
        queries["first_relevant_rank_pretrained"]
        - queries["first_relevant_rank_finetuned"]
    )

    queries["ranking_outcome"] = "unchanged"

    queries.loc[
        queries["rank_change"] > 0, "ranking_outcome"
    ] = "improved"

    queries.loc[
        queries["rank_change"] < 0, "ranking_outcome"
    ] = "worsened"

    print("\nToken length summary")
    print(
        queries["token_length"].describe(
            percentiles=[0.5, 0.75, 0.95]
        )
    )

    print("\nTruncation counts")
    print(queries["truncated"].value_counts())

    print("\nPerformance by query truncation")
    for truncated, group in queries.groupby("truncated"):
        print(
            f"\nTruncated: {truncated} "
            f"(n={len(group)})"
        )

        for metric in ("hit_at_1", "hit_at_5", "hit_at_10"):
            before = group[f"{metric}_pretrained"].mean()
            after = group[f"{metric}_finetuned"].mean()

            print(
                f"{metric}: {before:.4f} -> {after:.4f} "
                f"(delta={after-before:+.4f})"
            )

        print(
            "Ranking outcomes:",
            group["ranking_outcome"].value_counts().to_dict()
        )

    output = (
        RESULTS / f"{dataset}_truncation_analysis.csv"
    )
    queries.to_csv(output, index=False)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
