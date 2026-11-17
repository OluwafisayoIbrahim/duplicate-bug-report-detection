
from collections import Counter
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
from src.evaluate_entity_retrieval import (
    load_entities,
    jaccard,
    rerank_top_k,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASET = "issuedupes"
MODEL = "issuedupes-pilot"

SEMANTIC_WEIGHT = 0.95
RRF_K = 60
TOP_K = 50
WEIGHTS = [0.05, 0.15, 0.30]

INSPECT_CASES = [
    "175595",
    "161930",
    "174759",
    "172575",
]


def build_idf_weights(entities, fitting_indices):
    """
    Calculate IDF using fitting-period
    reports only.
    """
    document_frequency = Counter()

    for idx in fitting_indices:
        document_frequency.update(entities[int(idx)])

    n_documents = len(fitting_indices)

    weights = {
        entity: (
            np.log(
                (n_documents + 1)
                / (frequency + 1)
            ) + 1.0
        )
        for entity, frequency
        in document_frequency.items()
    }

    unseen_weight = np.log(
        n_documents + 1
    ) + 1.0

    print("Fitting reports:", n_documents)
    print("Unique fitting entities:", len(weights))

    return weights, unseen_weight


def weighted_jaccard(
    query_entities,
    candidate_entities,
    weights,
    unseen_weight,
):
    if not query_entities or not candidate_entities:
        return 0.0

    intersection = (
        query_entities & candidate_entities
    )

    if not intersection:
        return 0.0

    union = query_entities | candidate_entities

    numerator = sum(
        weights.get(entity, unseen_weight)
        for entity in intersection
    )

    denominator = sum(
        weights.get(entity, unseen_weight)
        for entity in union
    )

    return (
        numerator / denominator
        if denominator > 0
        else 0.0
    )


def rerank_with_scores(
    original_ranks,
    similarity_scores,
    weight,
):
    """
    Rerank only the original top 50.

    similarity_scores must correspond to
    the original top-50 candidate order.
    """
    n = min(TOP_K, len(original_ranks))

    top_indices = np.argsort(
        original_ranks,
        kind="stable",
    )[:n]

    new_ranks = original_ranks.copy()

    if n > 1:
        rank_scores = (
            1.0
            - (original_ranks[top_indices] - 1)
            / (n - 1)
        )
    else:
        rank_scores = np.ones(n)

    combined_scores = (
        rank_scores
        + weight * similarity_scores
    )

    order = np.argsort(
        -combined_scores,
        kind="stable",
    )

    new_ranks[top_indices[order]] = (
        np.arange(1, n + 1)
    )

    return new_ranks


def metrics(ranks, relevant_indices):
    best_rank = int(
        np.min(ranks[relevant_indices])
    )

    return {
        "first_relevant_rank": best_rank,
        "hit_at_1": int(best_rank <= 1),
        "hit_at_5": int(best_rank <= 5),
        "hit_at_10": int(best_rank <= 10),
        "reciprocal_rank": 1.0 / best_rank,
    }


def main():
    issues, queries, eligible, relevant = (
        prepare_benchmark(
            DATASET, "validation"
        )
    )

    issues = issues.copy()

    ids = issues["issue_id"].astype(
        str
    ).to_numpy()

    id_to_idx = {
        issue_id: idx
        for idx, issue_id in enumerate(ids)
    }

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    print("Dataset:", DATASET)
    print("Eligible queries:", len(eligible))

    signature = fingerprint(issues, MODEL)

    embedding_path = (
        CACHE
        / f"{DATASET}_{MODEL}_validation_{signature}.npy"
    )

    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if embeddings.shape != (len(issues), 384):
        raise ValueError(
            "Unexpected embedding cache shape."
        )

    print("Loaded embeddings:", embeddings.shape)

    validation_start = (
        queries["created_at"].min()
    )

    fitting_mask = (
        issues["created_at"] < validation_start
    ).to_numpy()

    fitting_indices = np.flatnonzero(
        fitting_mask
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )

    vectorizer.fit(
        issues.iloc[fitting_indices][
            "text"
        ].fillna("").tolist()
    )

    tfidf = vectorizer.transform(
        issues["text"].fillna("").tolist()
    )

    print("TF-IDF matrix:", tfidf.shape)

    entities = load_entities(issues, dataset=DATASET)

    idf_weights, unseen_weight = (
        build_idf_weights(
            entities,
            fitting_indices,
        )
    )

    rows = []

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc="Evaluating entity IDF",
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

        relevant_indices = np.array(
            sorted(
                id_to_idx[rid]
                for rid in relevant[query_id]
                if rid in id_to_idx
                and id_to_idx[rid] < cutoff
            ),
            dtype=np.int64,
        )

        if len(relevant_indices) == 0:
            raise ValueError(
                f"No relevant candidates: {query_id}"
            )

        lexical_scores = (
            tfidf[:cutoff]
            @ tfidf[query_idx].T
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

        query_entities = entities[query_idx]

        for base_name, base_ranks in (
            ("minilm", semantic_ranks),
            ("hybrid", hybrid_ranks),
        ):
            n = min(TOP_K, len(base_ranks))

            top_indices = np.argsort(
                base_ranks,
                kind="stable",
            )[:n]

            ordinary_scores = np.array(
                [
                    jaccard(
                        query_entities,
                        entities[int(idx)],
                    )
                    for idx in top_indices
                ],
                dtype=np.float64,
            )

            weighted_scores = np.array(
                [
                    weighted_jaccard(
                        query_entities,
                        entities[int(idx)],
                        idf_weights,
                        unseen_weight,
                    )
                    for idx in top_indices
                ],
                dtype=np.float64,
            )

            for weight in WEIGHTS:
                ordinary_name = (
                    f"jaccard_{base_name}_w{weight}"
                )

                idf_name = (
                    f"idf_{base_name}_w{weight}"
                )

                rankings[ordinary_name] = (
                    rerank_top_k(
                        base_ranks,
                        entities,
                        query_idx,
                        weight,
                    )
                )

                rankings[idf_name] = (
                    rerank_with_scores(
                        base_ranks,
                        weighted_scores,
                        weight,
                    )
                )

        for name, ranks in rankings.items():
            result = metrics(
                ranks,
                relevant_indices,
            )

            rows.append({
                "query_id": query_id,
                "method": name,
                "query_has_entities": bool(
                    query_entities
                ),
                **result,
            })

    frame = pd.DataFrame(rows)

    summary = (
        frame.groupby(
            "method",
            sort=False,
        )
        .agg(
            queries=("query_id", "size"),
            Hit1=("hit_at_1", "mean"),
            Hit5=("hit_at_5", "mean"),
            Hit10=("hit_at_10", "mean"),
            MRR=("reciprocal_rank", "mean"),
        )
        .reset_index()
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
    RESULTS / f"{DATASET}_entity_idf_queries.csv",
    index=False,
)

    summary.to_csv(
    RESULTS / f"{DATASET}_entity_idf_summary.csv",
    index=False,
)

    print("\nEntity-IDF evaluation")
    print("-" * 75)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nIDF versus ordinary Jaccard")
    print("-" * 75)

    for base in ("minilm", "hybrid"):
        for weight in WEIGHTS:
            ordinary = (
                f"jaccard_{base}_w{weight}"
            )

            weighted = (
                f"idf_{base}_w{weight}"
            )

            j_row = summary[
                summary["method"] == ordinary
            ].iloc[0]

            i_row = summary[
                summary["method"] == weighted
            ].iloc[0]

            print(
                f"{base}, weight={weight}: "
                f"Delta MRR="
                f"{i_row['MRR'] - j_row['MRR']:+.5f}, "
                f"Delta Hit@1="
                f"{i_row['Hit1'] - j_row['Hit1']:+.5f}"
            )

    print("\nKnown case comparisons")
    print("-" * 75)

    for query_id in INSPECT_CASES:
        subset = frame[
            frame["query_id"] == query_id
        ].set_index("method")

        if subset.empty:
            continue

        print(f"\nQuery: {query_id}")

        for method in (
            "hybrid",
            "jaccard_hybrid_w0.3",
            "idf_hybrid_w0.3",
        ):
            print(
                f"{method}: "
                f"rank={subset.loc[method, 'first_relevant_rank']}"
            )

    print("\nSaved results to results/.")


if __name__ == "__main__":
    main()
