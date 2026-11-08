
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"

METRICS = [
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "reciprocal_rank",
]


def load_comparison(dataset, model):
    path = (
        RESULTS
        / f"{dataset}_{model}_test_error_comparison.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    comparison = pd.read_csv(
        path,
        dtype={"query_id": str},
    )

    if comparison["query_id"].duplicated().any():
        raise ValueError("Duplicate query IDs found.")

    return comparison


def add_clusters(comparison, dataset, cluster_type):
    issues_path = (
        PROCESSED / f"{dataset}_issues_grouped.csv"
    )

    issues = pd.read_csv(
        issues_path,
        dtype={"issue_id": str, "group_id": str},
        usecols=["issue_id", "group_id"],
    )

    issues = issues.drop_duplicates("issue_id")

    merged = comparison.merge(
        issues,
        left_on="query_id",
        right_on="issue_id",
        how="left",
        validate="one_to_one",
    )

    if merged["group_id"].isna().any():
        raise ValueError("Some query groups are missing.")

    if cluster_type == "group":
        merged["cluster_id"] = merged["group_id"]

    elif cluster_type == "repository":
        if dataset != "issuedupes":
            raise ValueError(
                "Repository clustering is only supported "
                "for IssueDupesLite."
            )

        merged["cluster_id"] = (
            merged["query_id"].str.rsplit(
                "#", n=1
            ).str[0]
        )

        if (
            merged["cluster_id"].isna().any()
            or not merged["query_id"].str.contains(
                "#", regex=False
            ).all()
        ):
            raise ValueError(
                "Unexpected IssueDupesLite issue ID format."
            )

    return merged


def bootstrap_metric(
    dataframe,
    metric,
    iterations,
    rng,
):
    before_col = f"{metric}_pretrained"
    after_col = f"{metric}_finetuned"

    dataframe = dataframe.copy()

    dataframe["difference"] = (
        dataframe[after_col] - dataframe[before_col]
    )

    cluster_summary = dataframe.groupby(
        "cluster_id"
    )["difference"].agg(
        ["sum", "count"]
    )

    sums = cluster_summary["sum"].to_numpy(
        dtype=float
    )

    counts = cluster_summary["count"].to_numpy(
        dtype=float
    )

    n_clusters = len(cluster_summary)

    observed = dataframe["difference"].mean()

    estimates = np.empty(iterations)

    for i in range(iterations):
        sampled = rng.integers(
            low=0,
            high=n_clusters,
            size=n_clusters,
        )

        estimates[i] = (
            sums[sampled].sum()
            / counts[sampled].sum()
        )

    lower, upper = np.percentile(
        estimates,
        [2.5, 97.5],
    )

    return observed, lower, upper


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--cluster",
        choices=["group", "repository"],
        default="group",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.iterations < 100:
        raise ValueError(
            "Use at least 100 bootstrap iterations."
        )

    model = args.model or f"{args.dataset}-pilot"

    comparison = load_comparison(
        args.dataset,
        model,
    )

    data = add_clusters(
        comparison,
        args.dataset,
        args.cluster,
    )

    print("=" * 60)
    print("Cluster-Aware Retrieval Uncertainty Analysis")
    print("=" * 60)

    print("Dataset:", args.dataset)
    print("Model:", model)
    print("Cluster type:", args.cluster)
    print("Queries:", len(data))
    print("Clusters:", data["cluster_id"].nunique())
    print("Bootstrap iterations:", args.iterations)

    results = []

    for metric in METRICS:
        rng = np.random.default_rng(args.seed)

        observed, lower, upper = bootstrap_metric(
            data,
            metric,
            args.iterations,
            rng,
        )

        results.append({
            "metric": metric,
            "observed_difference": observed,
            "ci_lower": lower,
            "ci_upper": upper,
        })

    results_df = pd.DataFrame(results)

    print("\nBootstrap results")
    print("-" * 60)

    for row in results_df.itertuples(index=False):
        print(
            f"{row.metric}: "
            f"{row.observed_difference:+.4f} "
            f"95% CI [{row.ci_lower:+.4f}, "
            f"{row.ci_upper:+.4f}]"
        )

    output_path = (
        RESULTS
        / f"{args.dataset}_{model}_{args.cluster}_bootstrap.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
