
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import prepare_benchmark
from src.evaluate_hybrid import CACHE, fingerprint, get_ranks


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def inspect_query(
    query_id,
    issues,
    tfidf_matrix,
    embeddings,
    adjacency,
    top_k=5,
    semantic_weight=0.95,
    rrf_k=60,
):
    ids = issues["issue_id"].to_numpy()
    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    id_to_idx = {
        issue_id: i
        for i, issue_id in enumerate(ids)
    }

    if query_id not in id_to_idx:
        raise ValueError(f"Unknown query: {query_id}")

    query_idx = id_to_idx[query_id]
    query_time = timestamps[query_idx]

    cutoff = np.searchsorted(
        timestamps,
        query_time,
        side="left",
    )

    if cutoff == 0:
        raise ValueError("No earlier candidate reports.")

    # Same scoring as evaluate_hybrid.py
    lexical_scores = (
        tfidf_matrix[:cutoff]
        @ tfidf_matrix[query_idx].T
    ).toarray().ravel()

    semantic_scores = np.asarray(
        embeddings[:cutoff] @ embeddings[query_idx]
    )

    lexical_ranks = get_ranks(lexical_scores)
    semantic_ranks = get_ranks(semantic_scores)

    fused_scores = (
        (1.0 - semantic_weight)
        / (rrf_k + lexical_ranks)
        + semantic_weight
        / (rrf_k + semantic_ranks)
    )

    fused_ranks = get_ranks(fused_scores)

    relevant_ids = {
        other
        for other in adjacency.get(query_id, set())
        if other in id_to_idx
        and timestamps[id_to_idx[other]] < query_time
    }

    query_text = str(issues.iloc[query_idx]["text"])

    print("\n" + "=" * 70)
    print("QUERY:", query_id)
    print("=" * 70)
    print("Description:", query_text[:600])
    print("Earlier candidates:", cutoff)
    print("Known earlier direct duplicates:", sorted(relevant_ids))

    methods = {
        "TF-IDF": (lexical_ranks, lexical_scores),
        "MiniLM": (semantic_ranks, semantic_scores),
        "Weighted RRF": (fused_ranks, fused_scores),
    }

    records = []

    for method, (ranks, scores) in methods.items():
        print("\n" + "-" * 70)
        print(method.upper())
        print("-" * 70)

        top_indices = np.argsort(
            ranks, kind="stable"
        )[:top_k]

        for idx in top_indices:
            candidate_id = ids[idx]
            is_duplicate = candidate_id in relevant_ids

            candidate_text = str(
                issues.iloc[idx]["text"]
            )

            record = {
                "query_id": query_id,
                "method": method,
                "rank": int(ranks[idx]),
                "candidate_id": candidate_id,
                "direct_duplicate": is_duplicate,
                "score": float(scores[idx]),
                "tfidf_rank": int(lexical_ranks[idx]),
                "minilm_rank": int(semantic_ranks[idx]),
                "hybrid_rank": int(fused_ranks[idx]),
                "tfidf_score": float(lexical_scores[idx]),
                "minilm_score": float(semantic_scores[idx]),
                "hybrid_score": float(fused_scores[idx]),
                "candidate_text": candidate_text[:1200],
            }

            records.append(record)

            label = "DIRECT DUPLICATE" if is_duplicate else "Other"

            print(
                f"\nRank {record['rank']}: "
                f"{candidate_id} [{label}]"
            )

            print(
                f"Score: {record['score']:.6f} | "
                f"TF-IDF rank: {record['tfidf_rank']} | "
                f"MiniLM rank: {record['minilm_rank']} | "
                f"Hybrid rank: {record['hybrid_rank']}"
            )

            print("Text:", candidate_text[:250])

    return records


def main(args):
    if not 0 <= args.semantic_weight <= 1:
        raise ValueError("Weight must be between 0 and 1.")

    issues, queries, eligible, relevant = prepare_benchmark(
        args.dataset, "validation"
    )

    eligible_ids = set(eligible["issue_id"])

    for query_id in args.queries:
        if query_id not in eligible_ids:
            raise ValueError(
                f"Query {query_id} is not an eligible "
                "validation query."
            )

    signature = fingerprint(issues, args.model)

    cache_path = (
        CACHE
        / f"{args.dataset}_{args.model}_validation_{signature}.npy"
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}\n"
            "Run evaluate_hybrid.py for this dataset first."
        )

    embeddings = np.load(
        cache_path,
        allow_pickle=False,
        mmap_mode="r",
    )

    if embeddings.shape != (len(issues), 384):
        raise ValueError("Embedding cache shape mismatch.")

    print("Loaded cached embeddings:", embeddings.shape)

    # Fit the identical TF-IDF configuration
    # used by evaluate_hybrid.py.
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
    tfidf_matrix = vectorizer.transform(
        issues["text"].tolist()
    )

    links = pd.read_csv(
        PROCESSED / f"{args.dataset}_direct_links.csv",
        dtype={"source_id": str, "target_id": str},
    )

    adjacency = {}

    for source, target in links[
        ["source_id", "target_id"]
    ].itertuples(index=False, name=None):
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    all_records = []

    for query_id in args.queries:
        records = inspect_query(
            query_id=query_id,
            issues=issues,
            tfidf_matrix=tfidf_matrix,
            embeddings=embeddings,
            adjacency=adjacency,
            top_k=args.top_k,
            semantic_weight=args.semantic_weight,
            rrf_k=args.rrf_k,
        )

        all_records.extend(records)

    RESULTS.mkdir(parents=True, exist_ok=True)

    output = (
        RESULTS
        / f"{args.dataset}_hybrid_candidate_inspection.csv"
    )

    pd.DataFrame(all_records).to_csv(
        output, index=False
    )

    print("\nSaved candidate inspection:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=["issuedupes-pilot", "eclipse-pilot"],
        required=True,
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )

    main(parser.parse_args())
