
import argparse
import json
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
from src.extract_technical_entities import extract_entities


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ENTITY_CACHE = ROOT / "models" / "entity_cache"

DATASET = "eclipse"
MODEL = "eclipse-pilot"

SEMANTIC_WEIGHT = 0.95
RRF_K = 60
TOP_K = 50

WEIGHTS = [0.05, 0.15, 0.30]


def build_entity_set(text):
    extracted = extract_entities(text)

    tokens = set()

    qualified = set(
        extracted.get("qualified_identifiers", [])
    )

    placeholders = {
        "a.b.c",
        "foo.bar.baz",
        "com.example.test",
    }

    for category, values in extracted.items():
        category = (
            "code_identifiers"
            if category == "config_keys"
            else category
        )

        for value in values:
            if value in placeholders:
                continue

            if category == "file_names":
                if value in qualified:
                    continue

            tokens.add(f"{category}:{value}")

    return tokens


def load_entities(issues):
    ENTITY_CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        ENTITY_CACHE
        / f"{DATASET}_entities_v1.json"
    )

    ids = issues["issue_id"].astype(str).tolist()

    import hashlib

    signature = hashlib.sha256()

    for issue_id, text in zip(
        ids, issues["text"].fillna("")
    ):
        signature.update(issue_id.encode("utf-8"))
        signature.update(b"\0")
        signature.update(str(text).encode("utf-8"))
        signature.update(b"\0")

    digest = signature.hexdigest()

    if cache_path.exists():
        with cache_path.open(
            "r", encoding="utf-8"
        ) as f:
            cached = json.load(f)

        if (
            cached.get("signature") == digest
            and len(cached.get("entities", []))
            == len(issues)
        ):
            print("Loaded cached entities.")
            return [
                set(values)
                for values in cached["entities"]
            ]

    entities = [
        build_entity_set(text)
        for text in tqdm(
            issues["text"].fillna(""),
            desc="Extracting entities",
        )
    ]

    with cache_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "signature": digest,
                "entities": [
                    sorted(values)
                    for values in entities
                ],
            },
            f,
        )

    print("Saved entity cache.")
    return entities


def jaccard(a, b):
    if not a or not b:
        return 0.0

    intersection = len(a & b)

    if intersection == 0:
        return 0.0

    return intersection / len(a | b)


def rerank_top_k(
    base_ranks,
    entity_sets,
    query_idx,
    weight,
    top_k=TOP_K,
):
    """
    Rerank the original top-K candidates.

    Candidates outside top-K retain their
    original ranks.

    Score = normalized base rank
            + weight * entity Jaccard.
    """

    new_ranks = base_ranks.copy()

    n = min(top_k, len(base_ranks))

    top_indices = np.argsort(
        base_ranks,
        kind="stable",
    )[:n]

    query_entities = entity_sets[query_idx]

    similarities = np.array(
        [
            jaccard(
                query_entities,
                entity_sets[int(idx)],
            )
            for idx in top_indices
        ],
        dtype=np.float32,
    )

    if n > 1:
        normalized_rank_scores = (
            1.0
            - (base_ranks[top_indices] - 1)
            / (n - 1)
        )
    else:
        normalized_rank_scores = np.ones(1)

    scores = (
        normalized_rank_scores
        + weight * similarities
    )

    new_order = np.argsort(
        -scores,
        kind="stable",
    )

    ordered_indices = top_indices[new_order]

    new_ranks[ordered_indices] = np.arange(
        1, n + 1
    )

    return new_ranks


def calculate_metrics(ranks, relevant_indices):
    best = int(
        np.min(ranks[relevant_indices])
    )

    return {
        "first_relevant_rank": best,
        "hit_at_1": int(best <= 1),
        "hit_at_5": int(best <= 5),
        "hit_at_10": int(best <= 10),
        "reciprocal_rank": 1.0 / best,
    }


def save_summary(frame):
    summary = (
        frame.groupby("method", sort=False)
        .agg(
            queries=("query_id", "size"),
            Hit1=("hit_at_1", "mean"),
            Hit5=("hit_at_5", "mean"),
            Hit10=("hit_at_10", "mean"),
            MRR=("reciprocal_rank", "mean"),
        )
        .reset_index()
    )

    return summary


def main():
    issues, queries, eligible, relevant = (
        prepare_benchmark(
            DATASET,
            "validation",
        )
    )

    issues = issues.copy()

    issue_ids = issues["issue_id"].astype(
        str
    ).to_numpy()

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    id_to_idx = {
        issue_id: idx
        for idx, issue_id
        in enumerate(issue_ids)
    }

    print("Dataset:", DATASET)
    print("Model:", MODEL)
    print("Validation queries:", len(eligible))

    signature = fingerprint(issues, MODEL)

    cache_path = (
        CACHE
        / f"{DATASET}_{MODEL}_validation_{signature}.npy"
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}"
        )

    embeddings = np.load(
        cache_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if embeddings.shape != (len(issues), 384):
        raise ValueError(
            "Unexpected embedding matrix shape."
        )

    print("Loaded embeddings:", embeddings.shape)

    validation_start = queries[
        "created_at"
    ].min()

    fitting = issues[
        issues["created_at"]
        < validation_start
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )

    vectorizer.fit(
        fitting["text"].fillna("").tolist()
    )

    matrix = vectorizer.transform(
        issues["text"].fillna("").tolist()
    )

    print("TF-IDF matrix:", matrix.shape)

    entity_sets = load_entities(issues)

    print(
        "Reports with technical entities:",
        sum(bool(x) for x in entity_sets),
    )

    rows = []

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc="Entity-aware retrieval",
    ):
        query_id = str(query.issue_id)
        query_idx = id_to_idx[query_id]

        cutoff = np.searchsorted(
            timestamps,
            np.datetime64(
                query.created_at.to_datetime64()
            ),
            side="left",
        )

        if cutoff <= 0:
            raise ValueError(
                f"No candidates for {query_id}"
            )

        relevant_indices = np.array(
            sorted(
                id_to_idx[duplicate_id]
                for duplicate_id in relevant[
                    query_id
                ]
                if duplicate_id in id_to_idx
                and id_to_idx[
                    duplicate_id
                ] < cutoff
            ),
            dtype=np.int64,
        )

        if len(relevant_indices) == 0:
            raise ValueError(
                f"No earlier duplicate for {query_id}"
            )

        lexical_scores = (
            matrix[:cutoff]
            @ matrix[query_idx].T
        ).toarray().ravel()

        semantic_scores = np.asarray(
            embeddings[:cutoff]
            @ embeddings[query_idx]
        ).ravel()

        lexical_ranks = get_ranks(
            lexical_scores
        )

        semantic_ranks = get_ranks(
            semantic_scores
        )

        fused_scores = (
            (1.0 - SEMANTIC_WEIGHT)
            / (RRF_K + lexical_ranks)
            + SEMANTIC_WEIGHT
            / (RRF_K + semantic_ranks)
        )

        hybrid_ranks = get_ranks(
            fused_scores
        )

        rankings = {
            "minilm": semantic_ranks,
            "hybrid": hybrid_ranks,
        }

        for weight in WEIGHTS:
            rankings[
                f"entity_minilm_w{weight}"
            ] = rerank_top_k(
                semantic_ranks,
                entity_sets,
                query_idx,
                weight,
            )

            rankings[
                f"entity_hybrid_w{weight}"
            ] = rerank_top_k(
                hybrid_ranks,
                entity_sets,
                query_idx,
                weight,
            )

        for method, ranks in rankings.items():
            metrics = calculate_metrics(
                ranks,
                relevant_indices,
            )

            rows.append({
                "query_id": query_id,
                "method": method,
                "query_has_entities": bool(
                    entity_sets[query_idx]
                ),
                **metrics,
            })

    frame = pd.DataFrame(rows)

    summary = save_summary(frame)

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        RESULTS
        / "eclipse_entity_retrieval_queries.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS
        / "eclipse_entity_retrieval_summary.csv",
        index=False,
    )

    print("\nEntity-aware retrieval results")
    print("-" * 65)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nChange relative to baseline")
    print("-" * 65)

    for base in ("minilm", "hybrid"):
        baseline = summary[
            summary["method"] == base
        ].iloc[0]

        for weight in WEIGHTS:
            name = (
                f"entity_{base}_w{weight}"
            )

            result = summary[
                summary["method"] == name
            ].iloc[0]

            print(
                f"{name}: "
                f"Delta MRR="
                f"{result['MRR'] - baseline['MRR']:+.5f}, "
                f"Delta Hit@1="
                f"{result['Hit1'] - baseline['Hit1']:+.5f}"
            )

    print("\nOutput saved to results/.")


if __name__ == "__main__":
    main()
