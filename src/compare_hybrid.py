
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

prefix = "eclipse_eclipse-pilot_fusion_validation"

baseline = pd.read_csv(
    RESULTS / f"{prefix}_minilm_queries.csv",
    dtype={"query_id": str},
)

hybrid = pd.read_csv(
    RESULTS / f"{prefix}_full_w95_queries.csv",
    dtype={"query_id": str},
)

issues = pd.read_csv(
    ROOT / "data/processed/eclipse_issues_grouped.csv",
    usecols=["issue_id", "group_id"],
    dtype={"issue_id": str, "group_id": str},
)

assert baseline["query_id"].is_unique
assert hybrid["query_id"].is_unique

joined = baseline.merge(
    hybrid,
    on="query_id",
    suffixes=("_baseline", "_hybrid"),
    validate="one_to_one",
)

assert len(joined) == len(baseline) == len(hybrid)

joined = joined.merge(
    issues.rename(columns={"issue_id": "query_id"}),
    on="query_id",
    validate="many_to_one",
)

assert joined["group_id"].notna().all()

metrics = [
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "reciprocal_rank",
]

rng = np.random.default_rng(42)

group_indices = list(
    joined.groupby("group_id", sort=True).indices.values()
)

n_groups = len(group_indices)
n_bootstrap = 5000

print("Queries:", len(joined))
print("Duplicate groups:", n_groups)

for metric in metrics:
    a = joined[f"{metric}_baseline"].to_numpy()
    b = joined[f"{metric}_hybrid"].to_numpy()

    differences = b - a
    observed = differences.mean()

    samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        selected = rng.integers(
            0, n_groups, size=n_groups
        )

        sampled_indices = np.concatenate([
            group_indices[j] for j in selected
        ])

        samples[i] = differences[sampled_indices].mean()

    low, high = np.percentile(samples, [2.5, 97.5])

    print(
        f"{metric}: "
        f"delta={observed:+.5f}, "
        f"95% CI=[{low:+.5f}, {high:+.5f}]"
    )

print("\nNote: CIs are exploratory because the")
print("hybrid configuration was selected on")
print("this same validation dataset.")
