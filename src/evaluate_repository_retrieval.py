
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from src.evaluate import prepare_benchmark
from src.evaluate_hybrid import (
    CACHE,
    fingerprint,
    get_ranks,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASET = "issuedupes"
MODEL = "issuedupes-pilot"
SEMANTIC_WEIGHT = 0.95
RRF_K = 60


def get_repository(issue_id):
    issue_id = str(issue_id)

    if "#" not in issue_id:
        raise ValueError(f"Invalid GitHub issue ID: {issue_id}")

    return issue_id.rsplit("#", 1)[0].lower()


def ranking_metrics(ranks, relevant_indices):
    best_rank = int(np.min(ranks[relevant_indices]))

    return {
        "first_relevant_rank": best_rank,
        "hit_at_1": int(best_rank <= 1),
        "hit_at_5": int(best_rank <= 5),
        "hit_at_10": int(best_rank <= 10),
        "reciprocal_rank": 1.0 / best_rank,
    }


def rank_methods(lexical_scores, semantic_scores):
    lexical_ranks = get_ranks(lexical_scores)
    semantic_ranks = get_ranks(semantic_scores)

    fused_scores = (
        (1.0 - SEMANTIC_WEIGHT)
        / (RRF_K + lexical_ranks)
        + SEMANTIC_WEIGHT
        / (RRF_K + semantic_ranks)
    )

    hybrid_ranks = get_ranks(fused_scores)

    return {
        "tfidf": lexical_ranks,
        "minilm": semantic_ranks,
        "hybrid": hybrid_ranks,
    }


def summarize(frame, subset):
    rows = []

    for (mode, method), group in frame.groupby(
        ["mode", "method"]
    ):
        rows.append({
            "subset": subset,
            "mode": mode,
            "method": method,
            "queries": len(group),
            "Hit@1": group["hit_at_1"].mean(),
            "Hit@5": group["hit_at_5"].mean(),
            "Hit@10": group["hit_at_10"].mean(),
            "MRR": group["reciprocal_rank"].mean(),
            "mean_candidates": group["candidate_count"].mean(),
        })

    return rows


def main():
    issues, queries, eligible, relevant = prepare_benchmark(
        DATASET, "validation"
    )

    issues = issues.copy()

    issues["repository"] = (
        issues["issue_id"].map(get_repository)
    )

    issue_ids = issues["issue_id"].to_numpy()
    repositories = issues["repository"].to_numpy()

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    id_to_idx = {
        issue_id: idx
        for idx, issue_id in enumerate(issue_ids)
    }

    print("Dataset:", DATASET)
    print("Model:", MODEL)
    print("Validation queries:", len(eligible))

    covered_queries = set()
    coverage_records = []

    for query in eligible.itertuples(index=False):
        query_repo = get_repository(query.issue_id)

        same_repo_relevant = {
            duplicate_id
            for duplicate_id in relevant[query.issue_id]
            if get_repository(duplicate_id) == query_repo
        }

        covered = bool(same_repo_relevant)

        if covered:
            covered_queries.add(query.issue_id)

        coverage_records.append({
            "query_id": query.issue_id,
            "repository": query_repo,
            "known_earlier_duplicates": len(
                relevant[query.issue_id]
            ),
            "same_repo_duplicates": len(
                same_repo_relevant
            ),
            "same_repo_covered": covered,
        })

    coverage = pd.DataFrame(coverage_records)

    print("\nRepository duplicate coverage")
    print("-" * 45)

    print("Eligible queries:", len(coverage))
    print("Covered queries:", len(covered_queries))
    print(
        "Uncovered queries:",
        len(coverage) - len(covered_queries),
    )
    print(
        "Coverage rate:",
        f"{len(covered_queries) / len(coverage):.2%}",
    )

    # Load existing MiniLM embeddings.
    signature = fingerprint(issues, MODEL)

    cache_path = (
        CACHE
        / f"{DATASET}_{MODEL}_validation_{signature}.npy"
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing embedding cache: {cache_path}"
        )

    embeddings = np.load(
        cache_path,
        allow_pickle=False,
        mmap_mode="r",
    )

    if embeddings.shape != (len(issues), 384):
        raise ValueError("Unexpected embedding shape.")

    print("\nLoaded embeddings:", embeddings.shape)

    # Fit TF-IDF only on the inner fitting period.
    validation_start = queries["created_at"].min()

    fitting = issues[
        issues["created_at"] < validation_start
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )

    vectorizer.fit(fitting["text"].tolist())

    matrix = vectorizer.transform(
        issues["text"].tolist()
    )

    print("TF-IDF matrix:", matrix.shape)

    records = []

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc="Repository retrieval",
    ):
        query_id = query.issue_id
        query_idx = id_to_idx[query_id]
        query_repo = repositories[query_idx]

        cutoff = np.searchsorted(
            timestamps,
            np.datetime64(
                query.created_at.to_datetime64()
            ),
            side="left",
        )

        if cutoff <= 0:
            raise ValueError(
                f"No earlier candidates for {query_id}"
            )

        all_indices = np.arange(cutoff)

        same_repo_indices = all_indices[
            repositories[:cutoff] == query_repo
        ]

        relevant_indices = {
            id_to_idx[rid]
            for rid in relevant[query_id]
            if rid in id_to_idx
            and id_to_idx[rid] < cutoff
        }

        if not relevant_indices:
            raise ValueError(
                f"No relevant candidates for {query_id}"
            )

        candidate_pools = {
            "unrestricted": all_indices,
        }

        if query_id in covered_queries:
            candidate_pools["same_repository"] = (
                same_repo_indices
            )

        for mode, candidate_indices in candidate_pools.items():
            relevant_local = np.flatnonzero(
                np.isin(
                    candidate_indices,
                    list(relevant_indices),
                )
            )

            if len(relevant_local) == 0:
                raise ValueError(
                    f"Missing relevant candidate: "
                    f"{query_id}, {mode}"
                )

            lexical_scores = (
                matrix[candidate_indices]
                @ matrix[query_idx].T
            ).toarray().ravel()

            semantic_scores = np.asarray(
                embeddings[candidate_indices]
                @ embeddings[query_idx]
            )

            ranked_methods = rank_methods(
                lexical_scores,
                semantic_scores,
            )

            for method, ranks in ranked_methods.items():
                result = ranking_metrics(
                    ranks, relevant_local
                )

                top_local_idx = int(np.argmin(ranks))
                top_global_idx = candidate_indices[
                    top_local_idx
                ]

                records.append({
                    "query_id": query_id,
                    "repository": query_repo,
                    "mode": mode,
                    "method": method,
                    "candidate_count": len(
                        candidate_indices
                    ),
                    "top_candidate_id": issue_ids[
                        top_global_idx
                    ],
                    "same_repo_covered": (
                        query_id in covered_queries
                    ),
                    **result,
                })

    frame = pd.DataFrame(records)

    comparable = frame[
        frame["same_repo_covered"]
    ].copy()

    summary_rows = summarize(
        frame[frame["mode"] == "unrestricted"],
        "all_eligible_queries",
    )

    summary_rows.extend(
        summarize(comparable, "covered_queries")
    )

    summary = pd.DataFrame(summary_rows)

    RESULTS.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        RESULTS / "issuedupes_repository_summary.csv",
        index=False,
    )

    frame.to_csv(
        RESULTS / "issuedupes_repository_queries.csv",
        index=False,
    )

    coverage.to_csv(
        RESULTS / "issuedupes_repository_coverage.csv",
        index=False,
    )

    print("\nRetrieval comparison")
    print("-" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nPaired differences on covered queries")
    print("-" * 45)

    paired = summary[
        summary["subset"] == "covered_queries"
    ]

    for method in ("tfidf", "minilm", "hybrid"):
        base = paired[
            (paired["method"] == method)
            & (paired["mode"] == "unrestricted")
        ].iloc[0]

        restricted = paired[
            (paired["method"] == method)
            & (paired["mode"] == "same_repository")
        ].iloc[0]

        print(
            f"{method}: "
            f"MRR delta="
            f"{restricted['MRR'] - base['MRR']:+.5f}, "
            f"Hit@1 delta="
            f"{restricted['Hit@1'] - base['Hit@1']:+.5f}"
        )

    print("\nSaved repository analysis files.")


if __name__ == "__main__":
    main()
