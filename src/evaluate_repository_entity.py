
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from src.evaluate import prepare_benchmark
from src.evaluate_hybrid import CACHE, fingerprint, get_ranks
from src.evaluate_entity_retrieval import load_entities
from src.evaluate_entity_idf import (
    build_idf_weights,
    weighted_jaccard,
    rerank_with_scores,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASET = "issuedupes"
MODEL = "issuedupes-pilot"

SEMANTIC_WEIGHT = 0.95
RRF_K = 60
ENTITY_WEIGHT = 0.30
TOP_K = 50


def get_repository(issue_id):
    issue_id = str(issue_id)

    if "#" not in issue_id:
        raise ValueError(f"Invalid issue ID: {issue_id}")

    return issue_id.rsplit("#", 1)[0].lower()


def calculate_metrics(ranks, relevant_indices):
    best = int(np.min(ranks[relevant_indices]))

    return {
        "first_relevant_rank": best,
        "hit_at_1": int(best <= 1),
        "hit_at_5": int(best <= 5),
        "hit_at_10": int(best <= 10),
        "reciprocal_rank": 1.0 / best,
    }


def summarize(frame):
    return (
        frame.groupby(["mode", "method"], sort=False)
        .agg(
            queries=("query_id", "size"),
            Hit1=("hit_at_1", "mean"),
            Hit5=("hit_at_5", "mean"),
            Hit10=("hit_at_10", "mean"),
            MRR=("reciprocal_rank", "mean"),
            mean_candidates=("candidate_count", "mean"),
        )
        .reset_index()
    )


def main():
    issues, queries, eligible, relevant = (
        prepare_benchmark(DATASET, "validation")
    )

    issues = issues.copy()

    ids = issues["issue_id"].astype(str).to_numpy()

    repositories = np.array(
        [get_repository(issue_id) for issue_id in ids]
    )

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    id_to_idx = {
        issue_id: idx
        for idx, issue_id in enumerate(ids)
    }

    print("Dataset:", DATASET)
    print("Model:", MODEL)
    print("Validation queries:", len(eligible))

    signature = fingerprint(issues, MODEL)

    embedding_path = (
        CACHE
        / f"{DATASET}_{MODEL}_validation_{signature}.npy"
    )

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Missing embedding cache: {embedding_path}"
        )

    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if embeddings.shape != (len(issues), 384):
        raise ValueError("Unexpected embedding shape.")

    print("Loaded embeddings:", embeddings.shape)

    validation_start = queries["created_at"].min()

    fitting_indices = np.flatnonzero(
        (
            issues["created_at"] < validation_start
        ).to_numpy()
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )

    vectorizer.fit(
        issues.iloc[fitting_indices]["text"]
        .fillna("")
        .tolist()
    )

    tfidf = vectorizer.transform(
        issues["text"].fillna("").tolist()
    )

    print("TF-IDF matrix:", tfidf.shape)

    entities = load_entities(
        issues,
        dataset=DATASET,
    )

    idf_weights, unseen_weight = build_idf_weights(
        entities,
        fitting_indices,
    )

    records = []

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc="Repository entity retrieval",
    ):
        query_id = str(query.issue_id)
        query_idx = id_to_idx[query_id]
        query_repo = repositories[query_idx]

        cutoff = np.searchsorted(
            timestamps,
            np.datetime64(
                query.created_at.to_datetime64()
            ),
            side="left",
        )

        candidate_indices = np.flatnonzero(
            repositories[:cutoff] == query_repo
        )

        if len(candidate_indices) == 0:
            raise ValueError(
                f"No same-repository candidates: {query_id}"
            )

        known_relevant = set(
            str(rid) for rid in relevant[query_id]
        )

        relevant_local = np.array(
            [
                local_idx
                for local_idx, global_idx
                in enumerate(candidate_indices)
                if ids[global_idx] in known_relevant
            ],
            dtype=np.int64,
        )

        if len(relevant_local) == 0:
            raise ValueError(
                f"No same-repository duplicate: {query_id}"
            )

        lexical_scores = (
            tfidf[candidate_indices]
            @ tfidf[query_idx].T
        ).toarray().ravel()

        semantic_scores = np.asarray(
            embeddings[candidate_indices]
            @ embeddings[query_idx]
        ).ravel()

        lexical_ranks = get_ranks(lexical_scores)
        semantic_ranks = get_ranks(semantic_scores)

        fused_scores = (
            (1.0 - SEMANTIC_WEIGHT)
            / (RRF_K + lexical_ranks)
            + SEMANTIC_WEIGHT
            / (RRF_K + semantic_ranks)
        )

        hybrid_ranks = get_ranks(fused_scores)

        n = min(TOP_K, len(candidate_indices))

        top_local_indices = np.argsort(
            hybrid_ranks,
            kind="stable",
        )[:n]

        query_entities = entities[query_idx]

        entity_scores = np.array(
            [
                weighted_jaccard(
                    query_entities,
                    entities[
                        int(candidate_indices[local_idx])
                    ],
                    idf_weights,
                    unseen_weight,
                )
                for local_idx in top_local_indices
            ],
            dtype=np.float64,
        )

        entity_ranks = rerank_with_scores(
            hybrid_ranks,
            entity_scores,
            ENTITY_WEIGHT,
        )

        rankings = {
            "minilm": semantic_ranks,
            "hybrid": hybrid_ranks,
            "idf_hybrid": entity_ranks,
        }

        for method, ranks in rankings.items():
            result = calculate_metrics(
                ranks,
                relevant_local,
            )

            top_local = int(np.argmin(ranks))

            records.append({
                "query_id": query_id,
                "repository": query_repo,
                "mode": "same_repository",
                "method": method,
                "candidate_count": len(
                    candidate_indices
                ),
                "query_has_entities": bool(
                    query_entities
                ),
                "top_candidate_id": ids[
                    candidate_indices[top_local]
                ],
                **result,
            })

    restricted = pd.DataFrame(records)

    unrestricted_path = (
        RESULTS / "issuedupes_entity_idf_queries.csv"
    )

    unrestricted = pd.read_csv(
        unrestricted_path,
        dtype={"query_id": str},
    )

    unrestricted = unrestricted[
        unrestricted["method"].isin(
            ["minilm", "hybrid", "idf_hybrid_w0.3"]
        )
    ].copy()

    unrestricted["method"] = unrestricted[
        "method"
    ].replace({
        "idf_hybrid_w0.3": "idf_hybrid",
    })

    unrestricted["mode"] = "unrestricted"

    expected_ids = set(
        eligible["issue_id"].astype(str)
    )

    for name, frame in (
        ("restricted", restricted),
        ("unrestricted", unrestricted),
    ):
        if set(frame["query_id"]) != expected_ids:
            raise ValueError(
                f"{name} query set does not match."
            )

        if frame.duplicated(
            ["query_id", "method"]
        ).any():
            raise ValueError(
                f"Duplicate query-method rows: {name}"
            )

    combined = pd.concat(
        [unrestricted, restricted],
        ignore_index=True,
    )

    summary = summarize(combined)

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        RESULTS
        / "issuedupes_repository_entity_queries.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS
        / "issuedupes_repository_entity_summary.csv",
        index=False,
    )

    print("\nRepository-aware entity comparison")
    print("-" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nEntity effect within each candidate setting")
    print("-" * 80)

    for mode in ("unrestricted", "same_repository"):
        subset = summary[
            summary["mode"] == mode
        ].set_index("method")

        base = subset.loc["hybrid"]
        entity = subset.loc["idf_hybrid"]

        print(
            f"{mode}: "
            f"Delta MRR="
            f"{entity['MRR'] - base['MRR']:+.5f}, "
            f"Delta Hit@1="
            f"{entity['Hit1'] - base['Hit1']:+.5f}"
        )

    print("\nSame-repository query changes")
    print("-" * 80)

    same_repo = restricted.pivot(
        index="query_id",
        columns="method",
        values="first_relevant_rank",
    )

    improved = (
        same_repo["idf_hybrid"]
        < same_repo["hybrid"]
    )

    regressed = (
        same_repo["idf_hybrid"]
        > same_repo["hybrid"]
    )

    print("Improved:", int(improved.sum()))
    print("Regressed:", int(regressed.sum()))
    print(
        "Unchanged:",
        int((~improved & ~regressed).sum()),
    )

    full = unrestricted.pivot(
        index="query_id",
        columns="method",
        values="first_relevant_rank",
    )

    full_regressed = (
    full["idf_hybrid"] < full["hybrid"]
) == False

    full_regressed = (
        full["idf_hybrid"] > full["hybrid"]
    )

    print("\nPreviously regressed unrestricted queries:",
          int(full_regressed.sum()))

    print(
        "Of these, improved under repository restriction:",
        int((full_regressed & improved).sum()),
    )

    print(
        "Of these, still regressed under restriction:",
        int((full_regressed & regressed).sum()),
    )

    print(
        "Of these, unchanged under restriction:",
        int(
            (
                full_regressed
                & ~improved
                & ~regressed
            ).sum()
        ),
    )

    print("\nSaved results to results/.")


if __name__ == "__main__":
    main()
