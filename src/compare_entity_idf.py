
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASET = "eclipse"
BASE = "jaccard_hybrid_w0.3"
EXPERIMENT = "idf_hybrid_w0.3"

N_BOOTSTRAPS = 5000
SEED = 42


def main():
    results = pd.read_csv(
    RESULTS / "eclipse_entity_idf_queries.csv",
    dtype={"query_id": str},
)

    issues = pd.read_csv(
        ROOT / "data/processed/eclipse_issues_grouped.csv",
        usecols=["issue_id", "group_id"],
        dtype={"issue_id": str, "group_id": str},
    )

    a = (
        results[results["method"] == BASE]
        .set_index("query_id")
    )

    b = (
        results[results["method"] == EXPERIMENT]
        .set_index("query_id")
    )

    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("Duplicate query-method records.")

    if set(a.index) != set(b.index):
        raise ValueError("Query sets differ.")

    a = a.sort_index()
    b = b.reindex(a.index)

    metadata = (
        issues.drop_duplicates("issue_id")
        .set_index("issue_id")
    )

    groups = metadata.reindex(a.index)["group_id"]

    if groups.isna().any():
        raise ValueError("Some queries lack group IDs.")

    ranks_a = a["first_relevant_rank"].to_numpy()
    ranks_b = b["first_relevant_rank"].to_numpy()

    improved = ranks_b < ranks_a
    regressed = ranks_b > ranks_a
    unchanged = ranks_b == ranks_a

    print("Queries:", len(a))
    print("Improved:", improved.sum())
    print("Regressed:", regressed.sum())
    print("Unchanged:", unchanged.sum())

    group_codes, unique_groups = pd.factorize(
        groups.to_numpy(),
        sort=True,
    )

    n_groups = len(unique_groups)
    group_sizes = np.bincount(
        group_codes,
        minlength=n_groups,
    )

    rng = np.random.default_rng(SEED)

    metrics = {
        "Hit@1": "hit_at_1",
        "Hit@5": "hit_at_5",
        "Hit@10": "hit_at_10",
        "MRR": "reciprocal_rank",
    }

    print("\nPaired cluster bootstrap")
    print("-" * 60)
    print("Duplicate groups:", n_groups)

    for label, column in metrics.items():
        delta = (
            b[column].to_numpy(dtype=float)
            - a[column].to_numpy(dtype=float)
        )

        group_sums = np.bincount(
            group_codes,
            weights=delta,
            minlength=n_groups,
        )

        estimates = np.empty(
            N_BOOTSTRAPS,
            dtype=float,
        )

        for i in range(N_BOOTSTRAPS):
            sampled = rng.integers(
                0, n_groups, size=n_groups
            )

            estimates[i] = (
                group_sums[sampled].sum()
                / group_sizes[sampled].sum()
            )

        lower, upper = np.percentile(
            estimates, [2.5, 97.5]
        )

        print(
            f"{label}: "
            f"delta={delta.mean():+.5f}, "
            f"95% CI=[{lower:+.5f}, {upper:+.5f}]"
        )


if __name__ == "__main__":
    main()
