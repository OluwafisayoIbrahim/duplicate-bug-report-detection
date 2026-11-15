
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import prepare_benchmark
from src.evaluate_hybrid import CACHE, fingerprint, get_ranks
from src.evaluate_entity_retrieval import (
    DATASET,
    MODEL,
    RRF_K,
    SEMANTIC_WEIGHT,
    load_entities,
    jaccard,
    rerank_top_k,
)


CASES = ["175595", "161930", "174759", "172575"]
WEIGHT = 0.30


def main():
    issues, queries, eligible, relevant = prepare_benchmark(
        DATASET, "validation"
    )

    ids = issues["issue_id"].astype(str).to_numpy()
    id_to_idx = {
        issue_id: i for i, issue_id in enumerate(ids)
    }

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    signature = fingerprint(issues, MODEL)
    path = (
        CACHE
        / f"{DATASET}_{MODEL}_validation_{signature}.npy"
    )

    embeddings = np.load(
        path, mmap_mode="r", allow_pickle=False
    )

    entities = load_entities(issues)

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

    vectorizer.fit(fitting["text"].fillna("").tolist())

    tfidf = vectorizer.transform(
        issues["text"].fillna("").tolist()
    )

    eligible_by_id = {
        str(row.issue_id): row
        for row in eligible.itertuples(index=False)
    }

    for query_id in CASES:
        query = eligible_by_id[query_id]
        query_idx = id_to_idx[query_id]

        cutoff = np.searchsorted(
            timestamps,
            np.datetime64(
                query.created_at.to_datetime64()
            ),
            side="left",
        )

        lexical = (
            tfidf[:cutoff] @ tfidf[query_idx].T
        ).toarray().ravel()

        semantic = np.asarray(
            embeddings[:cutoff] @ embeddings[query_idx]
        ).ravel()

        lexical_ranks = get_ranks(lexical)
        semantic_ranks = get_ranks(semantic)

        fused_scores = (
            (1 - SEMANTIC_WEIGHT)
            / (RRF_K + lexical_ranks)
            + SEMANTIC_WEIGHT
            / (RRF_K + semantic_ranks)
        )

        baseline_ranks = get_ranks(fused_scores)

        entity_ranks = rerank_top_k(
            baseline_ranks,
            entities,
            query_idx,
            WEIGHT,
        )

        known = {
            str(issue_id)
            for issue_id in relevant[query_id]
        }

        print("\n" + "=" * 75)
        print("QUERY:", query_id)
        print("Known earlier duplicates:", sorted(known))

        # Inspect the union of both top-5 lists,
        # plus relevant reports found within top 50.
        baseline_top = set(
            np.argsort(baseline_ranks)[:5]
        )

        entity_top = set(
            np.argsort(entity_ranks)[:5]
        )

        relevant_top50 = {
            id_to_idx[rid]
            for rid in known
            if rid in id_to_idx
            and id_to_idx[rid] < cutoff
            and baseline_ranks[id_to_idx[rid]] <= 50
        }

        inspect_indices = (
            baseline_top | entity_top | relevant_top50
        )

        print("\nCandidate comparisons")
        print("-" * 75)

        for idx in sorted(
            inspect_indices,
            key=lambda i: entity_ranks[i],
        ):
            candidate_id = ids[idx]
            shared = sorted(
                entities[query_idx] & entities[idx]
            )

            print("\nCandidate:", candidate_id)
            print("Known duplicate:", candidate_id in known)
            print("Baseline rank:", baseline_ranks[idx])
            print("Entity rank:", entity_ranks[idx])
            print(
                "Entity Jaccard:",
                round(
                    jaccard(
                        entities[query_idx],
                        entities[idx],
                    ),
                    4,
                ),
            )
            print("Shared entities:", shared[:20])
            print(
                "Candidate text:",
                " ".join(
                    str(issues.iloc[idx]["text"]).split()
                )[:200],
            )


if __name__ == "__main__":
    main()
