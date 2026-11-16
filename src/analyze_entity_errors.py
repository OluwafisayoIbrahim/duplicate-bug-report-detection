
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import prepare_benchmark


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / "models" / "entity_cache"

BASELINE = "hybrid"
EXPERIMENT = "entity_hybrid_w0.3"


def load_entity_cache(issues):
    path = CACHE / "eclipse_entities_v1.json"

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    signature = hashlib.sha256()

    for issue_id, text in zip(
        issues["issue_id"].astype(str),
        issues["text"].fillna(""),
    ):
        signature.update(issue_id.encode("utf-8"))
        signature.update(b"\0")
        signature.update(str(text).encode("utf-8"))
        signature.update(b"\0")

    expected = signature.hexdigest()

    if data.get("signature") != expected:
        raise ValueError(
            "Entity cache does not match benchmark data."
        )

    entity_lists = data["entities"]

    if len(entity_lists) != len(issues):
        raise ValueError("Entity cache length mismatch.")

    return [set(x) for x in entity_lists]


def jaccard(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def excerpt(text, limit=250):
    return " ".join(str(text).split())[:limit]


def main():
    issues, queries, eligible, relevant = (
        prepare_benchmark("eclipse", "validation")
    )

    issues = issues.copy()
    issues["issue_id"] = issues["issue_id"].astype(str)

    entity_sets = load_entity_cache(issues)

    results = pd.read_csv(
        RESULTS / "eclipse_entity_retrieval_queries.csv",
        dtype={"query_id": str},
    )

    results = results[
        results["method"].isin(
            [BASELINE, EXPERIMENT]
        )
    ].copy()

    if results.duplicated(
        ["query_id", "method"]
    ).any():
        raise ValueError(
            "Duplicate query-method records."
        )

    ranks = results.pivot(
        index="query_id",
        columns="method",
        values="first_relevant_rank",
    )

    scores = results.pivot(
        index="query_id",
        columns="method",
        values="reciprocal_rank",
    )

    query_has_entities = results[
        results["method"] == BASELINE
    ].set_index("query_id")[
        "query_has_entities"
    ]

    comparison = pd.DataFrame({
        "baseline_rank": ranks[BASELINE],
        "entity_rank": ranks[EXPERIMENT],
        "baseline_rr": scores[BASELINE],
        "entity_rr": scores[EXPERIMENT],
        "query_has_entities": query_has_entities,
    })

    comparison["rank_change"] = (
        comparison["baseline_rank"]
        - comparison["entity_rank"]
    )

    comparison["delta_rr"] = (
        comparison["entity_rr"]
        - comparison["baseline_rr"]
    )

    comparison["outcome"] = np.select(
        [
            comparison["rank_change"] > 0,
            comparison["rank_change"] < 0,
        ],
        ["improved", "regressed"],
        default="unchanged",
    )

    issue_ids = issues["issue_id"].to_numpy()

    id_to_idx = {
        issue_id: idx
        for idx, issue_id in enumerate(issue_ids)
    }

    eligible_ids = set(
        eligible["issue_id"].astype(str)
    )

    if set(comparison.index) != eligible_ids:
        raise ValueError(
            "Results and eligible query sets differ."
        )

    records = []

    for query_id, row in comparison.iterrows():
        query_idx = id_to_idx[query_id]
        query_entities = entity_sets[query_idx]

        known_ids = sorted(
            str(rid)
            for rid in relevant[query_id]
            if str(rid) in id_to_idx
        )

        candidates = []

        for duplicate_id in known_ids:
            duplicate_idx = id_to_idx[
                duplicate_id
            ]

            duplicate_entities = entity_sets[
                duplicate_idx
            ]

            overlap = (
                query_entities & duplicate_entities
            )

            candidates.append({
                "duplicate_id": duplicate_id,
                "similarity": jaccard(
                    query_entities,
                    duplicate_entities,
                ),
                "shared": sorted(overlap),
            })

        best = max(
            candidates,
            key=lambda item: (
                item["similarity"],
                len(item["shared"]),
            ),
            default={
                "duplicate_id": "",
                "similarity": 0.0,
                "shared": [],
            },
        )

        records.append({
            "query_id": query_id,
            "group_id": issues.iloc[
                query_idx
            ]["group_id"],
            "outcome": row["outcome"],
            "baseline_rank": row[
                "baseline_rank"
            ],
            "entity_rank": row[
                "entity_rank"
            ],
            "rank_change": row[
                "rank_change"
            ],
            "delta_rr": row["delta_rr"],
            "query_has_entities": bool(
                query_entities
            ),
            "query_entity_count": len(
                query_entities
            ),
            "known_duplicate_count": len(
                known_ids
            ),
            "best_overlap_duplicate_id": best[
                "duplicate_id"
            ],
            "best_duplicate_jaccard": best[
                "similarity"
            ],
            "shared_entity_count": len(
                best["shared"]
            ),
            "shared_entities": json.dumps(
                best["shared"]
            ),
            "query_entities": json.dumps(
                sorted(query_entities)
            ),
            "query_excerpt": excerpt(
                issues.iloc[query_idx]["text"]
            ),
        })

    analysis = pd.DataFrame(records)

    coverage = (
        analysis.groupby(
            "query_has_entities"
        )
        .agg(
            queries=("query_id", "size"),
            improved=(
                "outcome",
                lambda x: (x == "improved").sum(),
            ),
            regressed=(
                "outcome",
                lambda x: (x == "regressed").sum(),
            ),
            unchanged=(
                "outcome",
                lambda x: (x == "unchanged").sum(),
            ),
            baseline_mrr=(
                "baseline_rank",
                lambda x: (1.0 / x).mean(),
            ),
            entity_mrr=(
                "entity_rank",
                lambda x: (1.0 / x).mean(),
            ),
            mean_delta_rr=("delta_rr", "mean"),
            mean_best_duplicate_jaccard=(
                "best_duplicate_jaccard",
                "mean",
            ),
        )
        .reset_index()
    )
    
    group_summary = (
        analysis.groupby("group_id")
        .agg(
            queries=("query_id", "size"),
            improved=(
                "outcome",
                lambda x: (x == "improved").sum(),
            ),
            regressed=(
                "outcome",
                lambda x: (x == "regressed").sum(),
            ),
            total_delta_rr=("delta_rr", "sum"),
        )
        .reset_index()
    )

    group_summary["changed"] = (
        group_summary["improved"]
        + group_summary["regressed"]
    )

    group_summary = group_summary.sort_values(
        ["changed", "queries"],
        ascending=False,
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis.to_csv(
        RESULTS / "eclipse_entity_error_analysis.csv",
        index=False,
    )

    coverage.to_csv(
        RESULTS / "eclipse_entity_coverage_analysis.csv",
        index=False,
    )

    group_summary.to_csv(
        RESULTS / "eclipse_entity_group_analysis.csv",
        index=False,
    )

    print("\nEntity-aware error analysis")
    print("-" * 60)

    print(
        analysis["outcome"].value_counts().to_string()
    )

    print("\nEntity coverage comparison")
    print("-" * 60)

    print(
        coverage.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nQueries sharing at least one entity")
    print("with a known earlier duplicate:")

    shared = (
        analysis["shared_entity_count"] > 0
    )

    print(
        f"{shared.sum()} / {len(analysis)} "
        f"({shared.mean():.2%})"
    )

    print("\nTop improvements")
    print("-" * 60)

    improved = analysis[
        analysis["outcome"] == "improved"
    ].sort_values(
        "delta_rr",
        ascending=False,
    )

    print(
        improved[
            [
                "query_id",
                "baseline_rank",
                "entity_rank",
                "delta_rr",
                "shared_entity_count",
                "best_overlap_duplicate_id",
            ]
        ].head(15).to_string(index=False)
    )

    print("\nTop regressions")
    print("-" * 60)

    regressed = analysis[
        analysis["outcome"] == "regressed"
    ].sort_values(
        "delta_rr"
    )

    print(
        regressed[
            [
                "query_id",
                "baseline_rank",
                "entity_rank",
                "delta_rr",
                "shared_entity_count",
                "best_overlap_duplicate_id",
            ]
        ].head(15).to_string(index=False)
    )

    print("\nTop duplicate groups by changed queries")
    print("-" * 60)

    print(
        group_summary.head(10).to_string(
            index=False
        )
    )

    print("\nSaved three CSV reports to results/.")


if __name__ == "__main__":
    main()
