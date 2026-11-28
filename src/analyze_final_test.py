
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_test"

SEED = 42
N_BOOTSTRAPS = 5000

COMPARISONS = [
    ("eclipse", "unrestricted",
     "hybrid", "minilm"),

    ("eclipse", "unrestricted",
     "idf_hybrid_w0.3", "hybrid"),

    ("eclipse", "unrestricted",
     "idf_hybrid_w0.3", "jaccard_hybrid_w0.3"),

    ("issuedupes", "unrestricted",
     "hybrid", "minilm"),

    ("issuedupes", "unrestricted",
     "idf_hybrid_w0.3", "hybrid"),

    ("issuedupes", "same_repository",
     "idf_hybrid_w0.3", "hybrid"),

    ("issuedupes", "same_repository",
     "hybrid", "minilm"),

    ("issuedupes", "same_repository",
     "hybrid", "unrestricted:hybrid"),
]


METRICS = {
    "Hit@1": "hit_at_1",
    "Hit@5": "hit_at_5",
    "Hit@10": "hit_at_10",
    "MRR": "reciprocal_rank",
}


def compare(df, dataset, mode, treatment, control):
    subset = df[df["dataset"] == dataset].copy()

    if ":" in control:
        control_mode, control_method = control.split(":", 1)
    else:
        control_mode, control_method = mode, control

    a = subset[
        (subset["mode"] == control_mode)
        & (subset["method"] == control_method)
    ].set_index("query_id")

    b = subset[
        (subset["mode"] == mode)
        & (subset["method"] == treatment)
    ].set_index("query_id")

    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("Duplicate query records.")

    if set(a.index) != set(b.index):
        raise ValueError(
            f"Query-set mismatch: {dataset}, {mode}"
        )

    a = a.sort_index()
    b = b.reindex(a.index)

    if not (
        a["group_id"].astype(str).to_numpy()
        == b["group_id"].astype(str).to_numpy()
    ).all():
        raise ValueError("Group IDs differ.")

    groups = a["group_id"].astype(str).to_numpy()
    group_codes, unique_groups = pd.factorize(
        groups, sort=True
    )

    n_groups = len(unique_groups)

    sizes = np.bincount(
        group_codes, minlength=n_groups
    )

    rng = np.random.default_rng(SEED)

    rank_a = a["first_relevant_rank"].to_numpy()
    rank_b = b["first_relevant_rank"].to_numpy()

    print(
        f"\n{dataset} | {mode} | "
        f"{treatment} vs {control}"
    )

    print(
        f"Queries={len(a)}, Groups={n_groups}, "
        f"Improved={(rank_b < rank_a).sum()}, "
        f"Regressed={(rank_b > rank_a).sum()}, "
        f"Unchanged={(rank_b == rank_a).sum()}"
    )

    rows = []

    for label, column in METRICS.items():
        delta = (
            b[column].to_numpy(dtype=float)
            - a[column].to_numpy(dtype=float)
        )

        group_sums = np.bincount(
            group_codes,
            weights=delta,
            minlength=n_groups,
        )

        estimates = np.empty(N_BOOTSTRAPS)

        for i in range(N_BOOTSTRAPS):
            sampled = rng.integers(
                0, n_groups, size=n_groups
            )

            estimates[i] = (
                group_sums[sampled].sum()
                / sizes[sampled].sum()
            )

        lower, upper = np.percentile(
            estimates, [2.5, 97.5]
        )

        print(
            f"{label}: "
            f"{delta.mean():+.5f}, "
            f"95% CI [{lower:+.5f}, {upper:+.5f}]"
        )

        rows.append({
            "dataset": dataset,
            "mode": mode,
            "treatment": treatment,
            "control": control,
            "metric": label,
            "queries": len(a),
            "groups": n_groups,
            "delta": float(delta.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        })

    return rows


def main():
    frames = []

    for dataset in ("eclipse", "issuedupes"):
        path = RESULTS / f"{dataset}_final_queries.csv"

        frame = pd.read_csv(
            path,
            dtype={
                "query_id": str,
                "group_id": str,
            },
        )

        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    results = []

    for comparison in COMPARISONS:
        results.extend(compare(df, *comparison))

    output = RESULTS / "final_test_bootstrap.csv"

    pd.DataFrame(results).to_csv(
        output, index=False
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
