
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"


def main(args):
    prefix = (
        f"{args.dataset}_{args.model}"
        "_fusion_validation"
    )

    baseline = pd.read_csv(
        RESULTS / f"{prefix}_minilm_queries.csv",
        dtype={"query_id": str},
    )

    hybrid = pd.read_csv(
        RESULTS / f"{prefix}_{args.config}_queries.csv",
        dtype={"query_id": str},
    )

    issues = pd.read_csv(
        PROCESSED / f"{args.dataset}_issues_grouped.csv",
        dtype={"issue_id": str, "group_id": str},
        low_memory=False,
    )

    links = pd.read_csv(
        PROCESSED / f"{args.dataset}_direct_links.csv",
        dtype={"source_id": str, "target_id": str},
    )

    assert baseline["query_id"].is_unique
    assert hybrid["query_id"].is_unique
    assert issues["issue_id"].is_unique

    comparison = baseline.merge(
        hybrid,
        on="query_id",
        suffixes=("_minilm", "_hybrid"),
        validate="one_to_one",
    )

    if len(comparison) != len(baseline) or len(comparison) != len(hybrid):
        raise ValueError("Query sets do not match.")

    comparison["rank_change"] = (
        comparison["first_relevant_rank_minilm"]
        - comparison["first_relevant_rank_hybrid"]
    )

    comparison["mrr_change"] = (
        comparison["reciprocal_rank_hybrid"]
        - comparison["reciprocal_rank_minilm"]
    )

    comparison["outcome"] = np.select(
        [
            comparison["rank_change"] > 0,
            comparison["rank_change"] < 0,
        ],
        ["improved", "regressed"],
        default="unchanged",
    )

    print("\nHybrid retrieval error analysis")
    print("=" * 50)
    print("Dataset:", args.dataset)
    print("Configuration:", args.config)
    print("Queries:", len(comparison))

    counts = comparison["outcome"].value_counts()

    for category in (
        "improved",
        "regressed",
        "unchanged",
    ):
        n = counts.get(category, 0)
        print(
            f"{category.capitalize()}: "
            f"{n} ({n / len(comparison):.2%})"
        )

    print(
        "\nMean MRR difference:",
        f"{comparison['mrr_change'].mean():+.5f}",
    )

    lookup = issues.set_index("issue_id")
    adjacency = {}

    for source, target in links[
        ["source_id", "target_id"]
    ].itertuples(index=False, name=None):
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    timestamps = pd.to_datetime(
        issues["created_at"],
        utc=True,
        errors="coerce",
    )

    created = dict(zip(issues["issue_id"], timestamps))

    examples = []

    for category in ("improved", "regressed"):
        subset = comparison[
            comparison["outcome"] == category
        ]

        if category == "improved":
            subset = subset.sort_values(
                "mrr_change", ascending=False
            )
        else:
            subset = subset.sort_values(
                "mrr_change", ascending=True
            )

        for row in subset.head(args.examples).itertuples(index=False):
            query_id = row.query_id

            if query_id not in lookup.index:
                continue

            query_time = created[query_id]

            earlier_duplicates = [
                other
                for other in adjacency.get(query_id, set())
                if other in created
                and pd.notna(query_time)
                and pd.notna(created[other])
                and created[other] < query_time
            ]

            example = {
                "outcome": category,
                "query_id": query_id,
                "minilm_rank": row.first_relevant_rank_minilm,
                "hybrid_rank": row.first_relevant_rank_hybrid,
                "rank_change": row.rank_change,
                "mrr_change": row.mrr_change,
                "query_text": str(
                    lookup.loc[query_id, "text"]
                )[:1500],
                "earlier_direct_duplicates": ", ".join(
                    sorted(earlier_duplicates)
                ),
            }

            if earlier_duplicates:
                duplicate_id = sorted(
                    earlier_duplicates
                )[0]

                example["example_duplicate_id"] = duplicate_id
                example["example_duplicate_text"] = str(
                    lookup.loc[duplicate_id, "text"]
                )[:1500]
            else:
                example["example_duplicate_id"] = ""
                example["example_duplicate_text"] = ""

            examples.append(example)

    RESULTS.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        RESULTS
        / f"{args.dataset}_{args.config}_rank_changes.csv"
    )

    examples_path = (
        RESULTS
        / f"{args.dataset}_{args.config}_error_examples.csv"
    )

    comparison.to_csv(comparison_path, index=False)
    pd.DataFrame(examples).to_csv(
        examples_path, index=False
    )

    print("\nRepresentative examples")
    for example in examples:
        print(
            f"{example['outcome']}: "
            f"{example['query_id']} "
            f"rank {example['minilm_rank']}"
            f" -> {example['hybrid_rank']}"
        )

    print("\nSaved:")
    print(comparison_path)
    print(examples_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=["issuedupes-pilot", "eclipse-pilot"],
        required=True,
    )

    parser.add_argument(
        "--config",
        default="full_w95",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=5,
    )

    main(parser.parse_args())
